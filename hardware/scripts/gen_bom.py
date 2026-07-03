"""Generate hardware/bom/bom.csv (engineering BOM) from parts.py.

The JLC assembly BOM/CPL (bom_jlc.csv, cpl_jlc.csv) are generated later by
validate.py from the actual placed schematic/board so designators always
match the CAD data; this file is the human/order-review BOM.

Run: uv run python hardware/scripts/gen_bom.py
"""

import csv
import sys
from pathlib import Path

HW_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HW_DIR / "scripts"))
from parts import PARTS

OUT = HW_DIR / "bom" / "bom.csv"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "mfr", "mpn", "lcsc", "jlc_class", "package",
                    "qty", "description", "alternate"])
        for key, p in PARTS.items():
            w.writerow([key, p["mfr"], p["mpn"], p.get("lcsc", ""),
                        p["jlc"], p["package"], p["qty"], p["desc"],
                        p.get("alt", "")])
    n = len(PARTS)
    basics = sum(1 for p in PARTS.values() if p["jlc"] == "basic")
    hand = sum(1 for p in PARTS.values() if p["jlc"] == "hand")
    print(f"Wrote {OUT} — {n} line items ({basics} basic, "
          f"{n - basics - hand} extended, {hand} hand-install)")


if __name__ == "__main__":
    main()
