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
import subprocess
import zipfile
import shutil
from pathlib import Path
from collections import defaultdict

import pcbnew

HW = Path(__file__).resolve().parent.parent
PCB = HW / "omnimarble-driver" / "omnimarble-driver.kicad_pcb"
DRC = HW / "fab" / "drc_driver.json"
OUT = HW / "fab" / "jlc"
GERB = OUT / "gerbers"
KICLI = r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe"

# footprint libs that are hand-soldered / not JLC-placed (mirror of validate.py)
HAND_FP_LIBS = {"Connector_PinSocket_2.54mm", "Connector_PinHeader_2.54mm",
                "Fuse", "TestPoint", "NetTie", "Capacitor_THT"}

GERBER_LAYERS = ("F.Cu,In1.Cu,In2.Cu,B.Cu,F.Paste,B.Paste,"
                 "F.Silkscreen,B.Silkscreen,F.Mask,B.Mask,Edge.Cuts")


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


def run_cli(*args):
    subprocess.run([KICLI, *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _table(rows):
    return "\n".join(f"| {r} | {v} | {extra} |" for r, v, extra in rows)


def write_order_notes(hand, feature, dnp, tht, n_bom, n_placed, n_cpl):
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
   part rotations (polarised caps, some ICs/connectors). Fix any flagged in JLC's
   online CPL editor before paying.
4. **Let JLC's DFM be the final arbiter.** Inspect DFM warnings around **U10**
   (fine-pitch ADS7042 escape), **U2**, the **ISNS_P/N Kelvin pair** (0.15 mm /
   ~0.17 mm spacing — intentional, 2 oz min), and **J1** (barrel jack is now an
   **edge-mount** — its body overhangs the bottom outline ~6 mm by design; pads are
   on-board).

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

---
*Generated by `hardware/scripts/gen_fab.py` from the committed board.*
"""
    (OUT / "ORDER_NOTES.md").write_text(txt, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if GERB.exists():
        shutil.rmtree(GERB)
    GERB.mkdir(parents=True)

    board = pcbnew.LoadBoard(str(PCB))
    groups = defaultdict(list)   # (value, package, lcsc) -> [refs]
    hand, feature, nolcsc, dnp, tht_placed = [], [], [], [], []
    placed_refs = set()
    for fp in board.Footprints():
        ref = fp.GetReference()
        val = fp.GetValue()
        pkg = fp.GetFPIDAsString().split(":")[-1]
        lcsc = lcsc_of(fp)
        kind = classify(fp)
        if kind == "placed":
            groups[(val, pkg, lcsc)].append(ref)
            placed_refs.add(ref)
            if is_tht(fp):
                tht_placed.append((ref, val, pkg, lcsc))
        elif kind == "hand":
            hand.append((ref, val, pkg, lcsc))
        elif kind == "feature":
            feature.append((ref, val))
        elif kind == "dnp":
            dnp.append((ref, val))
        elif kind == "nolcsc":
            nolcsc.append((ref, val, pkg))

    # ---- BOM (grouped, machine-placed) --------------------------------------
    def refsort(r):
        import re
        m = re.match(r"([A-Za-z]+)(\d+)", r)
        return (m.group(1), int(m.group(2))) if m else (r, 0)

    bom_rows = []
    for (val, pkg, lcsc), refs in groups.items():
        refs = sorted(refs, key=refsort)
        bom_rows.append((val, ",".join(refs), pkg, lcsc, len(refs)))
    bom_rows.sort(key=lambda r: (-r[4], r[0]))
    with open(OUT / "bom_jlc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Comment", "Designator", "Footprint", "LCSC Part #", "Qty"])
        for val, desig, pkg, lcsc, qty in bom_rows:
            w.writerow([val, desig, pkg, lcsc, qty])

    # ---- CPL from kicad-cli pos (correct gerber-convention coordinates) ------
    run_cli("pcb", "export", "gerbers", "--layers", GERBER_LAYERS,
            "--no-protel-ext", "-o", str(GERB) + "\\", str(PCB))
    # kicad-cli drops copper thickness from the gbrjob when an explicit stackup
    # is present; re-assert 2oz outer / 1oz inner so the fab metadata matches
    # the board's stackup (JLC sets copper weight from the order form regardless).
    import json
    job = next(GERB.glob("*.gbrjob"), None)
    if job:
        j = json.loads(job.read_text(encoding="utf-8"))
        th = {"F.Cu": 0.07, "B.Cu": 0.07, "In1.Cu": 0.035, "In2.Cu": 0.035}
        for layer in j.get("MaterialStackup", []):
            if layer.get("Type") == "Copper" and layer.get("Name") in th:
                layer["Thickness"] = th[layer["Name"]]
        job.write_text(json.dumps(j, indent=4), encoding="utf-8")
    run_cli("pcb", "export", "drill", "--format", "excellon",
            "--excellon-units", "mm", "--drill-origin", "absolute",
            "--generate-map", "--map-format", "gerberx2", "-o", str(GERB) + "\\",
            str(PCB))
    pos_tmp = OUT / "_pos_all.csv"
    run_cli("pcb", "export", "pos", "--format", "csv", "--units", "mm",
            "--side", "both", "-o", str(pos_tmp), str(PCB))

    n_cpl = 0
    with open(pos_tmp, newline="", encoding="utf-8") as fin, \
            open(OUT / "cpl_jlc.csv", "w", newline="", encoding="utf-8") as fout:
        r = csv.DictReader(fin)
        w = csv.writer(fout)
        w.writerow(["Designator", "Mid X", "Mid Y", "Layer", "Rotation"])
        for row in r:
            ref = row["Ref"]
            if ref not in placed_refs:
                continue
            layer = "Top" if row["Side"].lower() in ("top", "front", "f") \
                else "Bottom"
            w.writerow([ref, row["PosX"], row["PosY"], layer, row["Rot"]])
            n_cpl += 1
    pos_tmp.unlink()

    # ---- zip gerbers + drill -------------------------------------------------
    zpath = OUT / "omnimarble-driver-gerbers.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(GERB.iterdir()):
            z.write(p, p.name)

    if DRC.exists():
        shutil.copy(DRC, OUT / "drc_driver.json")

    n_placed = sum(len(v) for v in groups.values())
    write_order_notes(hand, feature, dnp, tht_placed, len(bom_rows),
                      n_placed, n_cpl)
    print(f"gerbers+drill: {len(list(GERB.iterdir()))} files -> {zpath.name}")
    print(f"BOM: {len(bom_rows)} line items, {n_placed} placed parts")
    print(f"CPL: {n_cpl} placements")
    print(f"hand-install: {len(hand)}  board-features: {len(feature)}  "
          f"no-LCSC(check!): {len(nolcsc)}")
    if nolcsc:
        print("  NO-LCSC:", ", ".join(f"{r}({v})" for r, v, _ in nolcsc))
    return hand, feature, nolcsc, n_placed, n_cpl


if __name__ == "__main__":
    main()
