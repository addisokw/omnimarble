"""Generate the project-local footprint library (omnimarble.pretty).

Custom footprints for parts with no bundled-library equivalent. Pad
numbering matches the schematic symbols' pin numbers. DRAFT dimensions
from datasheet outline classes — verified against part datasheets during
the layout phase (tracked in hardware/README).

Run: uv run python hardware/scripts/gen_footprints.py
"""

from pathlib import Path

HW_DIR = Path(__file__).resolve().parent.parent
LIB = HW_DIR / "omnimarble-driver" / "omnimarble.pretty"

HDR = '(footprint "{name}"\n\t(version 20240108)\n\t(generator "omnimarble_gen")\n\t(layer "F.Cu")\n\t(attr {attr})\n'


def smd_pad(num, x, y, w, h, shape="roundrect"):
    rr = '\n\t\t(roundrect_rratio 0.15)' if shape == "roundrect" else ""
    return (f'\t(pad "{num}" smd {shape}\n\t\t(at {x} {y})\n\t\t(size {w} {h})'
            f'\n\t\t(layers "F.Cu" "F.Paste" "F.Mask"){rr}\n\t)\n')


def tht_pad(num, x, y, dia, drill, shape="circle"):
    return (f'\t(pad "{num}" thru_hole {shape}\n\t\t(at {x} {y})\n\t\t(size {dia} {dia})'
            f'\n\t\t(drill {drill})\n\t\t(layers "*.Cu" "*.Mask")\n\t)\n')


def text(kind, s, y):
    return (f'\t(fp_text {kind} "{s}"\n\t\t(at 0 {y})\n\t\t(layer "F.SilkS")\n'
            f'\t\t(effects (font (size 1 1) (thickness 0.15)))\n\t)\n')


def write(name, attr, body):
    content = HDR.format(name=name, attr=attr)
    content += text("reference", "REF**", -8)
    content += text("value", name, 8)
    content += body + ")\n"
    (LIB / f"{name}.kicad_mod").write_text(content, encoding="utf-8")
    print(f"  {name}.kicad_mod")


def main():
    LIB.mkdir(parents=True, exist_ok=True)

    # TOLL-8 (HSOF-8 class, SFT040N150C3 / NCEP15T14LL): drain tab pad "2",
    # gate pad "1", source leads merged as pad "3" (symbol Q_NMOS_GDS pins)
    body = ""
    body += smd_pad("2", 0, -3.1, 9.6, 8.4, shape="rect")       # drain tab
    body += smd_pad("1", -4.445, 4.4, 0.7, 1.6)                 # gate lead
    for i in range(7):                                          # source leads
        body += smd_pad("3", -3.175 + i * 1.27 + 1.27, 4.4, 0.7, 1.6)
    write("TOLL-8_DRAFT", "smd", body)

    # 9.5mm barrier terminal, 2 position (Kangnex HB9500 class)
    body = ""
    body += tht_pad("1", -4.75, 0, 4.4, 2.2)
    body += tht_pad("2", 4.75, 0, 4.4, 2.2)
    write("TerminalBarrier_1x02_P9.50mm_DRAFT", "through_hole", body)

    # 5930 pulse shunt (Yezhan ASR-M class): two big end pads; Kelvin sense
    # connections are made by track attach points in layout, same pad nets
    body = ""
    body += smd_pad("1", -6.6, 0, 3.4, 8.2, shape="rect")
    body += smd_pad("2", 6.6, 0, 3.4, 8.2, shape="rect")
    write("Shunt_5930_DRAFT", "smd", body)

    # HF3FF / T73 relay: pads named to match Relay_SPDT symbol pins
    # (A1/A2 coil, 11 common, 12 NC, 14 NO). T73 grid (draft).
    body = ""
    body += tht_pad("A1", -6.1, -6.0, 2.6, 1.4)
    body += tht_pad("A2", 6.1, -6.0, 2.6, 1.4)
    body += tht_pad("11", 6.1, 6.0, 2.6, 1.4)
    body += tht_pad("14", -6.1, 6.0, 2.6, 1.4)
    body += tht_pad("12", -2.1, 6.0, 2.6, 1.4)
    write("Relay_HF3FF_T73_DRAFT", "through_hole", body)

    # fp-lib-table so 'omnimarble:' resolves for ERC/DRC/netlist
    (LIB.parent / "fp-lib-table").write_text(
        '(fp_lib_table\n  (version 7)\n'
        '  (lib (name "omnimarble")(type "KiCad")'
        '(uri "${KIPRJMOD}/omnimarble.pretty")(options "")(descr "project local"))\n)\n',
        encoding="utf-8")
    print("  fp-lib-table")


if __name__ == "__main__":
    main()
