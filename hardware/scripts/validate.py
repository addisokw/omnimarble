"""Schematic-level completeness validation (pre-layout gate).

Fails (exit != 0) on:
  - any placed footprint that doesn't exist in KiCad's bundled libraries
    or the project-local omnimarble.pretty
  - any machine-placed part (not in the hand-install allowlist) with an
    empty LCSC field
  - gate-rail parts missing their Variant field

Also generates the per-variant gate-rail BOMs
(hardware/bom/bom_rail_emit.csv / bom_rail_recv.csv).

Run: uv run python hardware/scripts/validate.py
"""

import csv
import sys
from pathlib import Path

HW_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HW_DIR / "scripts"))
from kicad_sch import KICAD_SHARE, Sym, find, find_all, parse

FP_SHARE = KICAD_SHARE / "footprints"
PROJECTS = {
    "driver": HW_DIR / "omnimarble-driver",
    "rail": HW_DIR / "gate-rail",
}

# Footprint-library prefixes / ref prefixes that are hand-installed or
# non-orderable board features (no LCSC required)
HAND_FP_LIBS = {"Connector_PinSocket_2.54mm", "Connector_PinHeader_2.54mm",
                "Fuse", "TestPoint", "NetTie", "Capacitor_THT"}
NO_PART_SYMBOLS = {"PWR_FLAG"}


def iter_symbols(sch_file):
    tree = parse(sch_file.read_text(encoding="utf-8"))[0]
    for sym in find_all(tree, Sym("symbol")):
        props = {}
        for p in find_all(sym, Sym("property")):
            props[str(p[1])] = str(p[2])
        lib_id = find(sym, Sym("lib_id"))
        dnp = find(sym, Sym("dnp"))
        yield {
            "lib_id": str(lib_id[1]) if lib_id else "",
            "ref": props.get("Reference", "?"),
            "value": props.get("Value", ""),
            "footprint": props.get("Footprint", ""),
            "lcsc": props.get("LCSC", ""),
            "variant": props.get("Variant", ""),
            "dnp": bool(dnp) and str(dnp[1]) == "yes",
            "file": sch_file.name,
        }


def footprint_exists(fp):
    if not fp or ":" not in fp:
        return False
    lib, name = fp.split(":", 1)
    for base in (FP_SHARE, HW_DIR / "omnimarble-driver", HW_DIR / "gate-rail"):
        cand = base / f"{lib}.pretty" / f"{name}.kicad_mod"
        if cand.exists():
            return True
    return False


def main():
    failures = []
    all_syms = {"driver": [], "rail": []}
    for key, proj in PROJECTS.items():
        for sch in sorted(proj.glob("*.kicad_sch")):
            all_syms[key].extend(iter_symbols(sch))

    for key, syms in all_syms.items():
        print(f"\n=== {key}: {len(syms)} symbol instances ===")
        seen_fp_fail, seen_lcsc_fail = set(), set()
        for s in syms:
            if s["ref"].startswith(("#",)) or s["value"] in NO_PART_SYMBOLS:
                continue
            if s["lib_id"].startswith(("power:",)):
                continue
            # footprint existence
            if s["footprint"] and not footprint_exists(s["footprint"]):
                if s["footprint"] not in seen_fp_fail:
                    seen_fp_fail.add(s["footprint"])
                    failures.append(f"{key}: footprint not found: "
                                    f"{s['footprint']} ({s['ref']})")
            if not s["footprint"] and not s["lib_id"].startswith(
                    ("Connector:TestPoint",)):
                failures.append(f"{key}: {s['ref']} has NO footprint")
            # LCSC completeness for machine-placed parts
            fp_lib = s["footprint"].split(":")[0] if s["footprint"] else ""
            hand = fp_lib in HAND_FP_LIBS or s["lib_id"] in (
                "Connector:TestPoint",) or fp_lib == "NetTie"
            if not hand and not s["lcsc"] and not s["dnp"]:
                tag = (s["ref"], s["value"])
                if tag not in seen_lcsc_fail:
                    seen_lcsc_fail.add(tag)
                    failures.append(f"{key}: missing LCSC: {s['ref']} "
                                    f"({s['value']}, {s['footprint']})")
        # duplicate refs (multi-unit shares are same lib_id+ref = OK)
        by_ref = {}
        for s in syms:
            if s["ref"].startswith("#"):
                continue
            by_ref.setdefault(s["ref"], set()).add(s["lib_id"])
        for ref, libids in by_ref.items():
            if len(libids) > 1:
                failures.append(f"{key}: ref {ref} used by different symbols: "
                                f"{sorted(libids)}")

    # Gate-rail: every populated part must carry a Variant (except connector/TP)
    for s in all_syms["rail"]:
        if s["ref"].startswith(("#", "J", "TP")) or s["lib_id"].startswith("power:"):
            continue
        if not s["variant"]:
            failures.append(f"rail: {s['ref']} missing Variant field")

    # Per-variant rail BOMs
    bom_dir = HW_DIR / "bom"
    bom_dir.mkdir(exist_ok=True)
    for variant in ("EMIT", "RECV"):
        rows = [s for s in all_syms["rail"]
                if s["variant"] == variant and not s["ref"].startswith("#")]
        out = bom_dir / f"bom_rail_{variant.lower()}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Reference", "Value", "Footprint", "LCSC"])
            for s in sorted(rows, key=lambda x: x["ref"]):
                w.writerow([s["ref"], s["value"], s["footprint"], s["lcsc"]])
        print(f"wrote {out.name} ({len(rows)} parts)")

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f_ in failures:
            print("  -", f_)
        sys.exit(1)
    print("\nvalidate.py: all completeness checks PASS")


if __name__ == "__main__":
    main()
