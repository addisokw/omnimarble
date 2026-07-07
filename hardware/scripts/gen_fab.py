"""Generate the JLCPCB fab/assembly package from the final board.

Outputs (under hardware/fab/jlc/):
  gerbers/                      Gerber + Excellon drill files
  omnimarble-driver-gerbers.zip zipped gerbers+drill (upload for fabrication)
  bom_jlc.csv                   JLC assembly BOM (machine-placed SMD parts)
  cpl_jlc.csv                   JLC pick-and-place / CPL (machine-placed parts)
  drc_driver.json               copy of the schematic-parity DRC report
  ORDER_NOTES.md                what JLC places vs. what is hand-installed

Machine-placed = has an LCSC number, not DNP, not excluded-from-pos (mounting
holes / fiducials), and its footprint library is not in HAND_FP_LIBS. The
hand-install set (connectors, fuse, THT caps, net-ties, test points) is listed
in ORDER_NOTES for the user to source and solder; JLC does not place it.

Run with KiCad 10's bundled python:
  "C:\\Program Files\\KiCad\\10.0\\bin\\python.exe" hardware/scripts/gen_fab.py
"""

import csv
import re
import subprocess
import sys
import zipfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import pcbnew

# JLCPCB CPL rotation corrections: KiCad's footprint orientation convention
# differs from JLC's EasyEDA library for whole families, so the raw exported
# rotation needs a per-footprint offset added. Table = the community
# fabrication-toolkit transformations.csv (authoritative). Specific patterns
# must precede general ones (first match wins). Footprints with no match pass
# through unchanged — that deliberately covers connectors/relays/THT (JLC places
# those by the hole pattern), whose rotation is verified in the JLC preview/DFM.
JLC_ROT_CORRECTIONS = [
    (r"^Bosch_LGA-", 90), (r"^CP_EIA-", 180), (r"^CP_Elec_", 180),
    (r"^C_Elec_", 180), (r"^DFN-", 270), (r"^D_SOT-23", 180),
    (r"^HTSSOP-", 270), (r"^JST_GH_SM", 180), (r"^JST_PH_S", 180),
    (r"^LQFP-", 270), (r"^MSOP-", 270), (r"^PowerPAK_SO-8_Single", 270),
    (r"^QFN-", 90), (r"^qfn-", 90), (r"^R_Array_Concave_", 90),
    (r"^R_Array_Convex_", 90), (r"^SC-74-6", 180),
    (r"^SOIC127P798X216-8N", -90), (r"^SOIC-", 270),
    (r"^SOP-18_", 0), (r"^SOP-4_", 0), (r"^SOP-", 270),
    (r"^SOT-143", 180), (r"^SOT-223", 180), (r"^SOT-23", 180),
    (r"^SOT-353", 180), (r"^SOT-363", 180), (r"^SOT-89", 180),
    (r"^SSOP-", 270), (r"^SW_SPST_B3", 90), (r"^TDSON-8-1", 270),
    (r"^TO-277", 90), (r"^TQFP-", 270), (r"^TSOT-23", 180),
    (r"^TSSOP-", 270), (r"^UDFN-10", 270), (r"^USON-10", 270),
    (r"^VSON-8_", 270), (r"^VSSOP-10_", 270), (r"^VSSOP-8_", 270),
]


def jlc_rotation(fp_name, raw_rot):
    """Apply the JLC rotation offset for a footprint; unmatched -> unchanged."""
    for pat, deg in JLC_ROT_CORRECTIONS:
        if re.match(pat, fp_name):
            return round((raw_rot + deg) % 360, 4)
    return round(raw_rot % 360, 4)

HW = Path(__file__).resolve().parent.parent
KICLI = r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe"

# footprint libs that are hand-soldered / not JLC-placed (mirror of validate.py)
HAND_FP_LIBS = {"Connector_PinSocket_2.54mm", "Connector_PinHeader_2.54mm",
                "Fuse", "TestPoint", "NetTie", "Capacitor_THT"}

LAYERS_4 = ("F.Cu,In1.Cu,In2.Cu,B.Cu,F.Paste,B.Paste,"
            "F.Silkscreen,B.Silkscreen,F.Mask,B.Mask,Edge.Cuts")
LAYERS_2 = ("F.Cu,B.Cu,F.Paste,B.Paste,"
            "F.Silkscreen,B.Silkscreen,F.Mask,B.Mask,Edge.Cuts")


@dataclass
class BoardCfg:
    """Per-board fab-package settings. The driver config reproduces the exact
    prior single-board output; the rail config adds 2-layer gerbers + per-variant
    BOM/CPL for the EMIT/RECV populations."""
    name: str
    pcb: Path
    drc: Path
    out: Path
    zip_name: str
    gerber_layers: str
    copper_th: dict            # gbrjob copper-layer name -> mm thickness
    bom_stem: str              # e.g. "bom_jlc" -> bom_jlc.csv / bom_rail_emit.csv
    cpl_stem: str
    notes: str                 # which ORDER_NOTES writer ("driver" / "rail")
    rev: str = "r1"            # stable design revision, stamped into the gbrjob
    variants: tuple = ()       # () = one package; else per-variant BOM+CPL

    @property
    def gerb(self):
        return self.out / "gerbers"


DRIVER = BoardCfg(
    name="driver",
    pcb=HW / "omnimarble-driver" / "omnimarble-driver.kicad_pcb",
    drc=HW / "fab" / "drc_driver.json",
    out=HW / "fab" / "jlc",
    zip_name="omnimarble-driver-gerbers.zip",
    gerber_layers=LAYERS_4,
    copper_th={"F.Cu": 0.07, "B.Cu": 0.07, "In1.Cu": 0.035, "In2.Cu": 0.035},
    bom_stem="bom_jlc", cpl_stem="cpl_jlc", notes="driver",
)

RAIL = BoardCfg(
    name="rail",
    pcb=HW / "gate-rail" / "gate-rail.kicad_pcb",
    drc=HW / "fab" / "drc_rail.json",
    out=HW / "fab" / "rail",
    zip_name="gate-rail-gerbers.zip",
    gerber_layers=LAYERS_2,
    copper_th={"F.Cu": 0.035, "B.Cu": 0.035},   # 2-layer, 1 oz both sides
    bom_stem="bom_rail", cpl_stem="cpl_rail", notes="rail",
    variants=("EMIT", "RECV"),
)

VBENCH = BoardCfg(
    name="vbench",
    pcb=HW / "vbench-pwr" / "vbench-pwr.kicad_pcb",
    drc=HW / "fab" / "drc_vbench.json",
    out=HW / "fab" / "vbench",
    zip_name="vbench-gerbers.zip",
    gerber_layers=LAYERS_2,
    copper_th={"F.Cu": 0.07, "B.Cu": 0.07},   # 2-layer, 2 oz both (pulse path)
    bom_stem="bom_vbench", cpl_stem="cpl_vbench", notes="driver",  # bare board: gerbers are the deliverable
)

BOARDS = {"driver": DRIVER, "rail": RAIL, "vbench": VBENCH}


def lcsc_of(fp):
    for f in fp.GetFields():
        if f.GetName().upper() in ("LCSC", "JLCPCB", "JLC"):
            return f.GetText().strip()
    return ""


def classify(fp):
    if fp.GetAttributes() & pcbnew.FP_EXCLUDE_FROM_POS_FILES:
        return "feature"                       # mounting holes, fiducials
    if fp.IsDNP():
        return "dnp"
    lib = fp.GetFPIDAsString().split(":")[0]
    if lib in HAND_FP_LIBS:
        return "hand"
    if not lcsc_of(fp):
        return "nolcsc"
    return "placed"


def is_tht(fp):
    return any(p.GetAttribute() in (pcbnew.PAD_ATTRIB_PTH, pcbnew.PAD_ATTRIB_NPTH)
               for p in fp.Pads())


def variant_of(fp):
    for f in fp.GetFields():
        if f.GetName() == "Variant":
            return f.GetText().strip()
    return ""


def variant_ok(fp, variant):
    """True if fp is populated in this build variant. variant=None -> all parts
    (single-package boards). Otherwise a part is placed iff its Variant field is
    the built variant, COMMON, or unset (mirrors validate.py's variant BOMs)."""
    if variant is None:
        return True
    return variant_of(fp) in ("", "COMMON", variant)


def run_cli(*args):
    subprocess.run([KICLI, *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _table(rows):
    return "\n".join(f"| {r} | {v} | {extra} |" for r, v, extra in rows)


def write_driver_notes(out, hand, feature, dnp, tht, n_bom, n_placed, n_cpl):
    hand_t = _table((r, v, f"{lc or '—'} ({pk})") for r, v, pk, lc in
                    sorted(hand))
    tht_t = _table((r, v, f"{lc} ({pk})") for r, v, pk, lc in sorted(tht))
    feat_t = "\n".join(f"| {r} | {v} |" for r, v in sorted(feature))
    dnp_t = "\n".join(f"| {r} | {v} |" for r, v in sorted(dnp)) or "| — | none |"
    txt = f"""# OmniMarble Driver — JLCPCB Order Notes

Assembly package for `omnimarble-driver.kicad_pcb`. **Scope: SMT + THT
assembly** (JLC places surface-mount *and* the common through-hole parts).

## Board
- 220 × 150 mm, **4-layer, 2 oz outer / 1 oz inner** copper, **ENIG** finish
  (F.Cu / In1.GND / In2.GND / B.Cu). The stackup + gerber job metadata state this.
- Min track/space 0.15 mm; min via 0.4 mm / 0.2 mm drill
- Electrical DRC: **0** (0 unconnected / shorts / clearance / parity). 16 cosmetic
  silk warnings only. Report: `drc_driver.json`.

## Files in this package
| File | Upload to |
|---|---|
| `omnimarble-driver-gerbers.zip` | JLC **fabrication** (gerbers + drill) |
| `bom_jlc.csv` | JLC **assembly** — BOM ({n_bom} line items) |
| `cpl_jlc.csv` | JLC **assembly** — pick-and-place ({n_cpl} placements) |
| `drc_driver.json` | reference (DRC evidence) |

Summary: **{n_placed} machine-placed** parts, **{len(hand)} hand-installed**,
{len(feature)} board features, {len(dnp)} do-not-populate.

## When ordering (read first)
1. **Set 4-layer, 2 oz OUTER copper + ENIG finish.** The thermal/current design
   assumes 2 oz outer; verify the order form matches (the gerber stackup states it,
   but the copper weight is a paid order-form selection). ENIG is recommended for
   the fine-pitch U10.
2. **Enable "Assemble top side" + turn ON through-hole assembly** — {len(tht)} of the
   placed parts are THT (below). Without THT assembly JLC will skip them.
3. **Verify component rotations in JLC's preview.** KiCad and JLC differ on some
   part rotations (polarised caps, some ICs/connectors). **Check K1/K2 especially**
   — both relays are at rotation 0 in the CPL and JLC's relay origin convention
   often differs from KiCad's; a wrong relay rotation swaps its terminals. Also
   confirm J1 (barrel jack) and the polarised caps. Fix any flagged in JLC's
   online CPL editor before paying.
4. **Let JLC's DFM be the final arbiter.** Inspect DFM warnings around **U10**
   (fine-pitch ADS7042 escape), **U2**, the **ISNS_P/N traces** (0.15 mm width —
   intentional, 2 oz min; note these are *not* a tight matched pair — see bring-up
   item 5), and **J1** (barrel jack is now an **edge-mount** — its body overhangs
   the bottom outline ~6 mm by design; pads are on-board).

## Machine-placed through-hole parts (need JLC THT assembly) — {len(tht)}
| Ref | Value | LCSC (package) |
|---|---|---|
{tht_t}

## Hand-installed by you (NOT on the CPL) — {len(hand)}
Order these separately (LCSC # given where known) and solder after JLC assembly.
| Ref | Value | LCSC (package) |
|---|---|---|
{hand_t}

## Board features (no placement) — {len(feature)}
| Ref | Value |
|---|---|
{feat_t}

## Do-not-populate — {len(dnp)}
| Ref | Value |
|---|---|
{dnp_t}

## Bring-up checklist (physical checks DRC/parity CANNOT catch)
1. **Ohmmeter each relay (K1, K2) de-energized: COM↔NC continuity.** The HF3FF
   footprint's NO/NC were transposed and fixed by a pad-number swap — parity is
   green *by construction*, so a meter is the ONLY confirmation the dump/charge
   logic isn't inverted. De-energized, COM must read to NC (not NO).
2. **OVP is a *throttle*, not a hard stop — the authoritative over-voltage cut
   is firmware.** U5's TL431 cathode sits on the UC3843 COMP pin and can only
   clamp it to ~2–2.5 V, which *reduces* the charge current but does not reach
   zero duty (that needs COMP < 1.4 V). So on a bench supply the observable is a
   **charge-rate knee near ~59–62 V with continued slow creep**, NOT a clean
   plateau — a tester expecting a hard stop could mis-judge a working part.
   **Firmware MUST treat VBANK_SENSE > threshold as a hard `BOOST_EN_N` inhibit**
   (drives Q1, which saturates COMP to ~0.1 V and has full authority). Verify
   *that* interlock, and the analog knee, before the first bank charge.
   *(Rev-2: give the TL431 full authority via a PNP off BST_VREF onto COMP/Q1.)*
3. **Fan-cool U2 (78L12, SOT-89) and watch its temp** during a sustained charge
   — it runs hot (~0.5–0.85 W); the DPAK thermal upgrade was deferred.
4. Stage first power at the **25 V floor**, then up to 55 V, verifying boost
   regulation before installing max bank capacitance or firing into the coil.
5. **Current-sense rise-phase caveat (ISNS).** The re-route left ISNS_P and
   ISNS_N on different corridors (not a tight pair), so during the fast dI/dt at
   pulse onset expect ~5–10 A of transient pickup error on the *rising* edge;
   **peak and energy readings are unaffected.** For first-spin sim-to-real
   comparison, trust peak/energy; treat the rise-phase waveform as indicative.
   (Rev-2 fix: pre-author the ISNS pair as locked critical copper.)

---
*Generated by `hardware/scripts/gen_fab.py` from the committed board.*
"""
    (out / "ORDER_NOTES.md").write_text(txt, encoding="utf-8")


def write_rail_notes(out, hand, feature, dnp, summary):
    """ORDER_NOTES for the gate-rail sensor board (2-layer, THT, EMIT/RECV)."""
    def vtable(variant):
        v, nb, npl, ncpl, parts = next(s for s in summary if s[0] == variant)
        rows = "\n".join(
            f"| {r} | {val} | {lc} ({pk}) | {'THT' if t else 'SMD'} |"
            for r, val, pk, lc, t in sorted(
                parts, key=lambda x: (x[0][0],
                                      int(x[0][1:]) if x[0][1:].isdigit() else 0)))
        return npl, ncpl, rows
    e_npl, e_cpl, e_rows = vtable("EMIT")
    r_npl, r_cpl, r_rows = vtable("RECV")
    hand_t = "\n".join(f"| {r} | {v} | {lc or '—'} ({pk}) |"
                       for r, v, pk, lc in sorted(hand)) or "| — | none | — |"
    txt = f"""# OmniMarble Gate-Rail (sensor board) — JLCPCB Order Notes

Assembly package for `gate-rail.kicad_pcb` — the IR break-beam sensor rail. **One
PCB, two population variants**; order the bare board once, then assemble as many
**EMIT** and **RECV** boards as your setup needs (a break-beam needs at least one
of each, plugged into the driver's J6 (RAIL-EMIT) / J7 (RAIL-RECV) headers).

## Board
- **224 × 15 mm, 2-layer, 1 oz** copper (low-current IR rail: ~144 mA on +5 V at
  ~24 mA/LED, sub-mA signals — 1 oz is ample). HASL or ENIG both fine.
- 6 sensor stations at z = -60 / -40 / -20 / +5 / +60 / +120 mm from the coil
  fiducial (silk-labelled; spacing asserted to 0.01 mm at board generation).
- IDC-10 (J1): `1=+5V 2=GND 3..8=SIG1..6 9=SIG_SPARE 10=GND` — matches driver **J7**
  pin-for-pin; driver **J6** is power-only (alternating +5V/GND), so a RECV board
  must never plug into J6. Label the ribbons.
- Electrical DRC **0** (0 unconnected / parity); silk warnings cosmetic. `drc_rail.json`.

## Files in this package
| File | Upload to |
|---|---|
| `gate-rail-gerbers.zip` | JLC **fabrication** (gerbers + drill) |
| `bom_rail_emit.csv` + `cpl_rail_emit.csv` | JLC **assembly** — EMIT variant ({e_npl} parts) |
| `bom_rail_recv.csv` + `cpl_rail_recv.csv` | JLC **assembly** — RECV variant ({r_npl} parts) |
| `drc_rail.json` | reference (DRC evidence) |

## When ordering (read first)
1. **Mixed SMD + THT — turn ON THT assembly.** The 150 Ω resistors are 0603 SMD;
   the IDC header, 5 mm IR LEDs and 3 mm phototransistors are through-hole. Without
   THT assembly JLC skips the through-hole parts.
2. **One gerber zip, but two assembly orders.** Fab the shared bare board, then run
   the EMIT CPL/BOM and the RECV CPL/BOM as **separate assembly jobs** — the two
   variants populate different parts on the *same* footprints, so a single CPL
   can't build both. Order however many of each you need.
3. **EMIT** populates D1–6 (IR333C-A) + R1–6 (150 Ω) + J1. **RECV** populates
   Q1–6 (PT204-6B) + J1. D and Q share each station; only one is fitted per build.
4. **Silk marks the built variant** — check the `EMIT [ ] RECV [ ]` box on each
   board so populated boards are distinguishable at assembly.
5. **🔴 CONFIRM OPTO POLARITY IN THE ASSEMBLY PREVIEW (mandatory gate).** The
   Everlight IR333C-A and PT204-6B number pin 1 as anode / emitter — *opposite*
   KiCad's generic `Device:LED` / `Q_Photo` convention. The board is wired
   correctly for **flat-to-flat** insertion (footprint pad 1 = flat-marked side:
   D cathode→GND, anode→LEDA; Q collector→SIG, emitter→GND), so JLC's polarity
   orientation places them right — **but verify it in the preview before paying**:
   each D's flat/cathode faces GND (away from its resistor), each Q's flat/collector
   faces its SIG trace. If the preview shows them reversed, rotate 180° in JLC's
   editor. (Do not have the *generator* swap the nets — that reverses a correct board.)
6. **Verify rotations in JLC's preview** (the IDC header especially), and let JLC's
   DFM be the final arbiter.

## EMIT variant — machine-placed — {e_npl} parts, {e_cpl} placements
| Ref | Value | LCSC (package) | Type |
|---|---|---|---|
{e_rows}

## RECV variant — machine-placed — {r_npl} parts, {r_cpl} placements
| Ref | Value | LCSC (package) | Type |
|---|---|---|---|
{r_rows}

## Hand-installed / not on the CPL
| Ref | Value | LCSC (package) |
|---|---|---|
{hand_t}

---
*Generated by `hardware/scripts/gen_fab.py rail` from the committed board.*
"""
    (out / "ORDER_NOTES.md").write_text(txt, encoding="utf-8")


def main(cfg):
    cfg.out.mkdir(parents=True, exist_ok=True)
    if cfg.gerb.exists():
        shutil.rmtree(cfg.gerb)
    cfg.gerb.mkdir(parents=True)

    board = pcbnew.LoadBoard(str(cfg.pcb))
    placed = []                          # (fp, ref, val, pkg, lcsc)
    hand, feature, nolcsc, dnp = [], [], [], []
    for fp in board.Footprints():
        ref = fp.GetReference()
        val = fp.GetValue()
        pkg = fp.GetFPIDAsString().split(":")[-1]
        lcsc = lcsc_of(fp)
        kind = classify(fp)
        if kind == "placed":
            placed.append((fp, ref, val, pkg, lcsc))
        elif kind == "hand":
            hand.append((ref, val, pkg, lcsc))
        elif kind == "feature":
            feature.append((ref, val))
        elif kind == "dnp":
            dnp.append((ref, val))
        elif kind == "nolcsc":
            nolcsc.append((ref, val, pkg))

    # ---- gerbers / drill / gbrjob / zip / drc (variant-independent) ----------
    run_cli("pcb", "export", "gerbers", "--layers", cfg.gerber_layers,
            "--no-protel-ext", "-o", str(cfg.gerb) + "\\", str(cfg.pcb))
    # kicad-cli drops copper thickness from the gbrjob when an explicit stackup
    # is present; re-assert the board's copper weights + stamp a stable revision
    # (KiCad leaves "rev?" unset; the git hash lags since committing changes it).
    import json
    job = next(cfg.gerb.glob("*.gbrjob"), None)
    if job:
        j = json.loads(job.read_text(encoding="utf-8"))
        for layer in j.get("MaterialStackup", []):
            if layer.get("Type") == "Copper" and layer.get("Name") in cfg.copper_th:
                layer["Thickness"] = cfg.copper_th[layer["Name"]]
        j.setdefault("GeneralSpecs", {}).setdefault("ProjectId", {})["Revision"] = cfg.rev
        job.write_text(json.dumps(j, indent=4), encoding="utf-8")
    run_cli("pcb", "export", "drill", "--format", "excellon",
            "--excellon-units", "mm", "--drill-origin", "absolute",
            "--generate-map", "--map-format", "gerberx2", "-o", str(cfg.gerb) + "\\",
            str(cfg.pcb))
    pos_tmp = cfg.out / "_pos_all.csv"
    run_cli("pcb", "export", "pos", "--format", "csv", "--units", "mm",
            "--side", "both", "-o", str(pos_tmp), str(cfg.pcb))
    with open(pos_tmp, newline="", encoding="utf-8") as fin:
        pos_rows = list(csv.DictReader(fin))
    pos_tmp.unlink()

    zpath = cfg.out / cfg.zip_name
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(cfg.gerb.iterdir()):
            z.write(p, p.name)
    if cfg.drc.exists():
        shutil.copy(cfg.drc, cfg.out / cfg.drc.name)

    # ---- BOM + CPL, per variant (cfg.variants) or a single package ----------
    def refsort(r):
        import re
        m = re.match(r"([A-Za-z]+)(\d+)", r)
        return (m.group(1), int(m.group(2))) if m else (r, 0)

    ref2fp = {ref: pkg for fp, ref, val, pkg, lcsc in placed}   # for JLC rotation
    summary = []
    for variant in (cfg.variants or (None,)):
        suffix = f"_{variant.lower()}" if variant else ""
        groups = defaultdict(list)       # (value, package, lcsc) -> [refs]
        parts, sub_refs = [], set()      # (ref, val, pkg, lcsc, is_tht)
        for fp, ref, val, pkg, lcsc in placed:
            if not variant_ok(fp, variant):
                continue
            groups[(val, pkg, lcsc)].append(ref)
            sub_refs.add(ref)
            parts.append((ref, val, pkg, lcsc, is_tht(fp)))
        bom_rows = []
        for (val, pkg, lcsc), refs in groups.items():
            refs = sorted(refs, key=refsort)
            bom_rows.append((val, ",".join(refs), pkg, lcsc, len(refs)))
        bom_rows.sort(key=lambda r: (-r[4], r[0]))
        with open(cfg.out / f"{cfg.bom_stem}{suffix}.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Comment", "Designator", "Footprint", "LCSC Part #", "Qty"])
            for val, desig, pkg, lcsc, qty in bom_rows:
                w.writerow([val, desig, pkg, lcsc, qty])
        n_cpl = 0
        with open(cfg.out / f"{cfg.cpl_stem}{suffix}.csv", "w", newline="",
                  encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["Designator", "Mid X", "Mid Y", "Layer", "Rotation"])
            for row in pos_rows:
                if row["Ref"] not in sub_refs:
                    continue
                layer = "Top" if row["Side"].lower() in ("top", "front", "f") \
                    else "Bottom"
                rot = jlc_rotation(ref2fp.get(row["Ref"], ""), float(row["Rot"]))
                w.writerow([row["Ref"], row["PosX"], row["PosY"], layer, rot])
                n_cpl += 1
        summary.append((variant, len(bom_rows),
                        sum(len(v) for v in groups.values()), n_cpl, parts))

    # ---- ORDER_NOTES --------------------------------------------------------
    if cfg.notes == "rail":
        write_rail_notes(cfg.out, hand, feature, dnp, summary)
    else:
        _, nb, npl, ncpl, parts = summary[0]
        tht = [(r, v, pk, lc) for r, v, pk, lc, t in parts if t]
        write_driver_notes(cfg.out, hand, feature, dnp, tht, nb, npl, ncpl)

    # ---- report -------------------------------------------------------------
    print(f"[{cfg.name}] gerbers+drill: {len(list(cfg.gerb.iterdir()))} files "
          f"-> {zpath.name}")
    for variant, nb, npl, ncpl, tht in summary:
        print(f"  {variant or 'all'}: BOM {nb} lines / {npl} placed, CPL {ncpl}")
    print(f"  hand-install: {len(hand)}  features: {len(feature)}  "
          f"no-LCSC(check!): {len(nolcsc)}")
    if nolcsc:
        print("  NO-LCSC:", ", ".join(f"{r}({v})" for r, v, _ in nolcsc))
    return hand, feature, nolcsc, summary


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "driver"
    if which not in BOARDS:
        raise SystemExit(f"usage: gen_fab.py [{'|'.join(BOARDS)}]")
    main(BOARDS[which])
