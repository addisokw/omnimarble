"""Generate the project-local footprint library (omnimarble.pretty).

Datasheet-exact land patterns (sources recorded per footprint below).
Pad numbering matches the schematic symbols' pin numbers.

Run: uv run python hardware/scripts/gen_footprints.py
"""

from pathlib import Path

HW_DIR = Path(__file__).resolve().parent.parent
LIB = HW_DIR / "omnimarble-driver" / "omnimarble.pretty"

HDR = '(footprint "{name}"\n\t(version 20240108)\n\t(generator "omnimarble_gen")\n\t(layer "F.Cu")\n\t(attr {attr})\n'


def smd_pad(num, x, y, w, h, shape="roundrect"):
    rr = "\n\t\t(roundrect_rratio 0.1)" if shape == "roundrect" else ""
    return (f'\t(pad "{num}" smd {shape}\n\t\t(at {x} {y})\n\t\t(size {w} {h})'
            f'\n\t\t(layers "F.Cu" "F.Paste" "F.Mask"){rr}\n\t)\n')


def tht_pad(num, x, y, pw, ph, dw, dh=None, shape="oval"):
    drill = f"(drill {dw})" if dh is None else f"(drill oval {dw} {dh})"
    return (f'\t(pad "{num}" thru_hole {shape}\n\t\t(at {x} {y})\n\t\t(size {pw} {ph})'
            f'\n\t\t{drill}\n\t\t(layers "*.Cu" "*.Mask")\n\t)\n')


def silk_line(x1, y1, x2, y2, w=0.15):
    return (f'\t(fp_line\n\t\t(start {x1} {y1})\n\t\t(end {x2} {y2})\n'
            f'\t\t(stroke (width {w}) (type solid))\n\t\t(layer "F.SilkS")\n\t)\n')


def silk_rect(x1, y1, x2, y2, w=0.15):
    return (silk_line(x1, y1, x2, y1, w) + silk_line(x2, y1, x2, y2, w)
            + silk_line(x2, y2, x1, y2, w) + silk_line(x1, y2, x1, y2 if False else y2, w)
            + silk_line(x1, y2, x1, y1, w))


def crtyd_rect(x1, y1, x2, y2):
    return (f'\t(fp_rect\n\t\t(start {x1} {y1})\n\t\t(end {x2} {y2})\n'
            f'\t\t(stroke (width 0.05) (type solid))\n\t\t(fill none)\n'
            f'\t\t(layer "F.CrtYd")\n\t)\n')


def silk_circle(cx, cy, r, w=0.3):
    return (f'\t(fp_circle\n\t\t(center {cx} {cy})\n\t\t(end {cx + r} {cy})\n'
            f'\t\t(stroke (width {w}) (type solid))\n\t\t(fill solid)\n'
            f'\t\t(layer "F.SilkS")\n\t)\n')


def text(kind, s, y, layer="F.SilkS", size=1.0):
    t = "fp_text" if kind in ("reference", "value") else "fp_text"
    kindtok = kind if kind in ("reference", "value") else "user"
    return (f'\t(fp_text {kindtok} "{s}"\n\t\t(at 0 {y})\n\t\t(layer "{layer}")\n'
            f'\t\t(effects (font (size {size} {size}) (thickness 0.15)))\n\t)\n')


def write(name, attr, body, ref_y=-9, val_y=9):
    content = HDR.format(name=name, attr=attr)
    content += text("reference", "REF**", ref_y)
    content += text("value", name, val_y)
    content += body + ")\n"
    (LIB / f"{name}.kicad_mod").write_text(content, encoding="utf-8")
    print(f"  {name}.kicad_mod")


def main():
    LIB.mkdir(parents=True, exist_ok=True)
    # drop superseded drafts
    for old in LIB.glob("*_DRAFT.kicad_mod"):
        old.unlink()

    # ------------------------------------------------------------------
    # TOLL-8 (JEDEC HSOF-8) — SCILICON SFT040N150C3 / NCEP15T14LL
    # Source: SCILICON pkg drawing + Leadpower TOLL-X8 suggested land:
    #   drain pad 10.20 x 8.10 at (0, -2.60); 8 lead pads 0.80 x 2.80 at
    #   y=+5.25, x = +/-0.60/1.80/3.00/4.20 (1.20 pitch).
    # Pad map for Q_NMOS_GDS: gate = corner lead (x=-4.20) -> "1";
    # remaining 7 leads = source -> "3"; drain tab -> "2".
    # ------------------------------------------------------------------
    body = smd_pad("2", 0, -2.60, 10.20, 8.10, shape="rect")
    xs = [-4.20, -3.00, -1.80, -0.60, 0.60, 1.80, 3.00, 4.20]
    body += smd_pad("1", xs[0], 5.25, 0.80, 2.80)
    for x in xs[1:]:
        body += smd_pad("3", x, 5.25, 0.80, 2.80)
    body += silk_circle(-5.6, 5.25, 0.25)           # pin-1 (gate) marker
    body += silk_rect(-5.2, -7.2, 5.2, 3.6)
    body += crtyd_rect(-5.85, -7.45, 5.85, 7.0)
    body += text("user", "G", 7.4, size=0.8)
    write("TOLL-8", "smd", body)

    # ------------------------------------------------------------------
    # D2PAK (TO-263) Schottky, cathode tab — YFW MBR60100DC (dual
    # common-cathode: both anode leads paralleled).
    # Geometry from KiCad TO-263-3_TabPin2 (leads 4.6x1.1 at x=-7.65,
    # y=+/-2.54; tab 9.4x10.8 at (1.5, 0)); numbering REMAPPED for
    # Device:D_Schottky (1 = K = tab, 2 = A = both leads). Using the
    # stock TO-263-2 would put the ANODE on the tab — inverted for every
    # cathode-tab D2PAK schottky.
    # ------------------------------------------------------------------
    body = smd_pad("1", 1.5, 0, 9.4, 10.8, shape="rect")     # cathode tab
    body += smd_pad("2", -7.65, -2.54, 4.6, 1.1)             # anode lead
    body += smd_pad("2", -7.65, 2.54, 4.6, 1.1)              # anode lead
    body += silk_rect(-5.0, -5.5, 6.5, 5.5)
    body += crtyd_rect(-10.2, -5.9, 6.6, 5.9)
    body += text("user", "K=tab", 6.8, size=0.8)
    write("D2PAK_MBR60100DC", "smd", body, ref_y=-7, val_y=8)

    # ------------------------------------------------------------------
    # 5930 pulse shunt — Yezhan ASR-M-7 (datasheet land pattern):
    # element 15.0 x 7.6; two pads 5.2 x 8.75, centers 10.8mm apart.
    # ------------------------------------------------------------------
    body = smd_pad("1", -5.4, 0, 5.2, 8.75, shape="rect")
    body += smd_pad("2", 5.4, 0, 5.2, 8.75, shape="rect")
    body += silk_line(-2.0, -4.0, 2.0, -4.0) + silk_line(-2.0, 4.0, 2.0, 4.0)
    body += crtyd_rect(-8.25, -4.65, 8.25, 4.65)
    write("Shunt_5930", "smd", body, ref_y=-6, val_y=6)

    # ------------------------------------------------------------------
    # 9.5mm barrier terminal 2P — Kangnex HB9500 (datasheet):
    # pitch 9.50; blade 1.5 x 0.7 -> oval slot 2.0 x 1.2, pad 4.0 x 3.0;
    # body 20.5 x 17.8, pin row 5.1 from wire-entry face.
    # ------------------------------------------------------------------
    body = tht_pad("1", 0, 0, 4.0, 3.0, 2.0, 1.2)
    body += tht_pad("2", 9.5, 0, 4.0, 3.0, 2.0, 1.2)
    body += silk_rect(-5.5, -5.1, 15.0, 12.7)
    body += crtyd_rect(-5.75, -5.35, 15.25, 12.95)
    body += text("user", "wire entry ->", -6.2, size=0.8)
    write("TerminalBarrier_1x02_P9.50mm", "through_hole", body,
          ref_y=-7.5, val_y=14.2)

    # ------------------------------------------------------------------
    # HF3FF relay (T73) — Hongfa datasheet PCB layout (bottom view =
    # top-side footprint chirality). Pads named for Relay:Relay_SPDT
    # (A1/A2 coil, 11 = COM, 12 = NC, 14 = NO).
    # COM (0,0) drill 1.5; coil (2.0, -/+6.0) drill 1.3; contacts
    # (14.2, -/+6.0) drill 1.3. NO/NC assignment needs a continuity
    # check on the physical part (drawing does not number them).
    # ------------------------------------------------------------------
    body = tht_pad("11", 0, 0, 3.0, 3.0, 1.5, shape="circle")
    body += tht_pad("A1", 2.0, -6.0, 2.5, 2.5, 1.3, shape="circle")
    body += tht_pad("A2", 2.0, 6.0, 2.5, 2.5, 1.3, shape="circle")
    body += tht_pad("14", 14.2, -6.0, 3.0, 3.0, 1.3, shape="circle")
    body += tht_pad("12", 14.2, 6.0, 3.0, 3.0, 1.3, shape="circle")
    body += silk_rect(-1.4, -7.6, 17.6, 7.6)
    body += crtyd_rect(-1.65, -7.85, 17.85, 7.85)
    body += text("user", "VERIFY NO/NC by continuity", 8.6, size=0.7)
    write("Relay_HF3FF_T73", "through_hole", body, ref_y=-8.6, val_y=10.0)

    # fp-lib-table so 'omnimarble:' resolves for ERC/DRC/netlist
    (LIB.parent / "fp-lib-table").write_text(
        '(fp_lib_table\n  (version 7)\n'
        '  (lib (name "omnimarble")(type "KiCad")'
        '(uri "${KIPRJMOD}/omnimarble.pretty")(options "")(descr "project local"))\n)\n',
        encoding="utf-8")
    print("  fp-lib-table")


if __name__ == "__main__":
    main()
