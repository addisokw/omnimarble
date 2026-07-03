"""Generate gate-rail.kicad_pcb: 224x15mm 2-layer IR break-beam rail.

Run under KiCad python:
  "C:\\Program Files\\KiCad\\10.0\\bin\\python.exe" hardware/scripts/gen_rail_pcb.py

Six sensor stations at z = -60/-40/-20/+5/+60/+120 mm from the coil-center
fiducial (fiducial at board x=70 -> stations at x=10..190); the IDC
connector J1 lives on a 34mm tail east of the last station (its shrouded
courtyard cannot fit between stations). EMIT parts (R + 5mm IR LED) and
the RECV phototransistor are alternate-populate; silkscreen checkboxes
mark the built variant.

Deterministic, planar, via-free signal wiring (2 vias total, both +5V):
  B.Cu  GND zone; SIG_SPARE runs J1.9 -> TP1 at y=0.8; +5V riser J1.1
  F.Cu  +5V bus y=13.4; LED feeds y=11.4; SIG lanes y=1.0..3.5 with depth
        nested by station (west shallower) so every drop meets its own
        lane first; odd-row J1 pins rise directly (pin-x order matches
        lane nesting); even-row pins loop around J1's east side through
        the empty north corridor (rows y=10.6..11.4) and drop from above.

Station x positions are asserted to 0.01mm against STATIONS_MM.
"""

import sys
from pathlib import Path

import pcbnew

HW_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HW_DIR / "scripts"))
from kicad_sch import Sym, find, find_all, parse

PROJ = HW_DIR / "gate-rail"
NETLIST = PROJ / "gate-rail.net"
PCB_OUT = PROJ / "gate-rail.kicad_pcb"
FP_SHARE = Path(r"C:\Program Files\KiCad\10.0\share\kicad\footprints")

MM = pcbnew.FromMM
BOARD_W, BOARD_H = 224.0, 15.0
FIDUCIAL_X = 70.0
STATIONS_MM = [-60, -40, -20, 5, 60, 120]
LANE_F = {0: 1.0, 2: 1.5, 4: 2.0}   # odd stations: F.Cu lanes
LANE_B = {1: 1.5, 3: 2.5, 5: 3.5}   # even stations: B.Cu lanes (THT ends)
J1_ANCHOR = (210.5, 6.0, 270)                # rows run -x; odd row y=6


def V(x, y):
    return pcbnew.VECTOR2I(MM(x), MM(y))


def load_netlist():
    tree = parse(NETLIST.read_text(encoding="utf-8"))[0]
    comps = {}
    for c in find_all(find(tree, Sym("components")), Sym("comp")):
        ref = str(find(c, Sym("ref"))[1])
        fp = find(c, Sym("footprint"))
        comps[ref] = str(fp[1]) if fp else ""
    padnet = {}
    for n in find_all(find(tree, Sym("nets")), Sym("net")):
        name = str(find(n, Sym("name"))[1])
        for node in find_all(n, Sym("node")):
            padnet[(str(find(node, Sym("ref"))[1]),
                    str(find(node, Sym("pin"))[1]))] = name
    return comps, padnet


def load_footprint(fpid):
    lib, name = fpid.split(":", 1)
    fp = pcbnew.FootprintLoad(str(FP_SHARE / f"{lib}.pretty"), name)
    if not fp:
        raise RuntimeError(f"footprint not found: {fpid}")
    return fp


def main():
    comps, padnet = load_netlist()
    if PCB_OUT.exists():
        PCB_OUT.unlink()
    board = pcbnew.NewBoard(str(PCB_OUT))
    board.SetCopperLayerCount(2)
    F, B = pcbnew.F_Cu, pcbnew.B_Cu

    pts = [(0, 0), (BOARD_W, 0), (BOARD_W, BOARD_H), (0, BOARD_H)]
    for i in range(4):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(V(*pts[i]))
        seg.SetEnd(V(*pts[(i + 1) % 4]))
        seg.SetLayer(pcbnew.Edge_Cuts)
        seg.SetWidth(MM(0.1))
        board.Add(seg)

    nets = {}

    def netinfo(name):
        if name not in nets:
            ni = pcbnew.NETINFO_ITEM(board, name)
            board.Add(ni)
            nets[name] = ni
        return nets[name]

    def track(net, layer, x0, y0, x1, y1, w=0.3):
        t = pcbnew.PCB_TRACK(board)
        t.SetStart(V(x0, y0))
        t.SetEnd(V(x1, y1))
        t.SetWidth(MM(w))
        t.SetLayer(layer)
        t.SetNet(net)
        board.Add(t)

    def via(net, x, y):
        v = pcbnew.PCB_VIA(board)
        v.SetPosition(V(x, y))
        v.SetDrill(MM(0.3))
        v.SetWidth(MM(0.6))
        v.SetViaType(pcbnew.VIATYPE_THROUGH)
        v.SetNet(net)
        board.Add(v)

    def text(s, x, y, size=1.0, layer=None):
        t = pcbnew.PCB_TEXT(board)
        t.SetText(s)
        t.SetPosition(V(x, y))
        t.SetLayer(layer if layer is not None else pcbnew.F_SilkS)
        t.SetTextSize(pcbnew.VECTOR2I(MM(size), MM(size)))
        t.SetTextThickness(MM(size * 0.15))
        board.Add(t)

    # placement -------------------------------------------------------------
    P = {"J1": J1_ANCHOR, "TP1": (218, 2.5, 0)}
    for i, z in enumerate(STATIONS_MM):
        sx = FIDUCIAL_X + z
        P[f"D{i + 1}"] = (sx, 10.9, 0)     # 5mm IR LED (EMIT)
        P[f"Q{i + 1}"] = (sx, 4.8, 0)      # 3mm phototransistor (RECV)
        # R6 sits west of its LED (J1 courtyard owns the east tail)
        P[f"R{i + 1}"] = (sx + (7 if i < 5 else -7), 12.6, 0)  # 150R

    for ref, fpid in sorted(comps.items()):
        fp = load_footprint(fpid)
        fp.SetReference(ref)
        for pad in fp.Pads():
            key = (ref, pad.GetNumber())
            if key in padnet:
                pad.SetNet(netinfo(padnet[key]))
        x, y, rot = P[ref]
        fp.SetPosition(V(x, y))
        fp.SetOrientationDegrees(rot)
        board.Add(fp)

    for i, (hx, hy) in enumerate([(4.5, 4), (220, 11)]):
        fp = load_footprint("MountingHole:MountingHole_3.2mm_M3_Pad")
        fp.SetReference(f"H{i + 1}")
        fp.SetPosition(V(hx, hy))
        board.Add(fp)

    fps = {fp.GetReference(): fp for fp in board.Footprints()}
    for i, z in enumerate(STATIONS_MM):
        want = FIDUCIAL_X + z
        for ref in (f"D{i + 1}", f"Q{i + 1}"):
            got = pcbnew.ToMM(fps[ref].GetPosition().x)
            assert abs(got - want) <= 0.01, f"{ref} x={got}, want {want}"
    print("station x assertions PASS (six stations within 0.01mm)")

    def pad_xy(ref, num):
        for pad in fps[ref].Pads():
            if pad.GetNumber() == num:
                pp = pad.GetPosition()
                return pcbnew.ToMM(pp.x), pcbnew.ToMM(pp.y)
        raise KeyError((ref, num))

    # B.Cu GND zone
    z = pcbnew.ZONE(board)
    z.SetLayer(B)
    z.SetNet(netinfo("GND"))
    outline = z.Outline()
    outline.NewOutline()
    for x, y in [(0.6, 0.6), (BOARD_W - 0.6, 0.6),
                 (BOARD_W - 0.6, BOARD_H - 0.6), (0.6, BOARD_H - 0.6)]:
        outline.Append(MM(x), MM(y))
    z.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
    z.SetMinThickness(MM(0.25))
    board.Add(z)

    # +5V: J1.1 rises on B.Cu, via at the bus, bus west along y=13.4
    five = netinfo("+5V")
    x5, y5 = pad_xy("J1", "1")
    track(five, B, x5, y5, x5 + 1.6, y5, 0.8)
    track(five, B, x5 + 1.6, y5, x5 + 1.6, 10.0, 0.8)
    via(five, x5 + 1.6, 10.0)
    track(five, F, x5 + 1.6, 10.0, x5 + 1.6, 13.7, 0.8)
    bus_xs = [x5 + 1.6]
    for i in range(6):
        px, py = pad_xy(f"R{i + 1}", "1")
        track(five, F, px, py, px, 13.7, 0.5)
        bus_xs.append(px)
    track(five, F, min(bus_xs), 13.7, max(bus_xs), 13.7, 0.8)

    # LED feed: R pad2 -> y=11.4 -> east of LED anode -> pad (never crosses
    # the GND cathode pad, which sits west of the anode)
    for i in range(6):
        n = netinfo(padnet[(f"R{i + 1}", "2")])
        rx2, ry2 = pad_xy(f"R{i + 1}", "2")
        ax, ay = pad_xy(f"D{i + 1}", "2")
        row = 11.4 if i < 5 else 12.9   # R6 feed passes over the LED body
        track(n, F, rx2, ry2, rx2, row, 0.4)
        track(n, F, rx2, row, ax, row, 0.4)
        track(n, F, ax, row, ax, ay, 0.4)

    # SIG lanes: odd stations on F.Cu, even stations on B.Cu (both ends
    # THT so the B lanes need no vias); depths nested per side
    for i in range(6):
        n = netinfo(f"SIG{i + 1}")
        cx, cy = pad_xy(f"Q{i + 1}", "1")
        jx, jy = pad_xy("J1", str(3 + i))
        if i in LANE_F:
            lane = LANE_F[i]
            track(n, F, cx, cy, cx, lane, 0.3)
            track(n, F, cx, lane, jx, lane, 0.3)
            track(n, F, jx, lane, jx, jy, 0.3)
        else:
            lane = LANE_B[i]
            ex = jx + 1.27
            track(n, B, cx, cy, cx, lane, 0.3)
            track(n, B, cx, lane, ex, lane, 0.3)
            track(n, B, ex, lane, ex, jy, 0.3)
            track(n, B, ex, jy, jx, jy, 0.3)

    # SIG_SPARE: J1.9 -> north on B -> east -> down to TP1 (clear of H2)
    n = netinfo("SIG_SPARE")
    jx, jy = pad_xy("J1", "9")
    tx, ty = pad_xy("TP1", "1")
    ex = jx + 1.27
    track(n, B, jx, jy, ex, jy, 0.3)
    track(n, B, ex, jy, ex, 13.0, 0.3)
    track(n, B, ex, 13.0, 215.4, 13.0, 0.3)
    track(n, B, 215.4, 13.0, 215.4, ty, 0.3)
    via(n, 215.4, ty)
    track(n, F, 215.4, ty, tx, ty, 0.3)

    # silkscreen -------------------------------------------------------------
    for (a, b) in [((FIDUCIAL_X - 2, 8.2), (FIDUCIAL_X + 2, 8.2)),
                   ((FIDUCIAL_X, 6.7), (FIDUCIAL_X, 9.7))]:
        sseg = pcbnew.PCB_SHAPE(board)
        sseg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        sseg.SetStart(V(*a))
        sseg.SetEnd(V(*b))
        sseg.SetLayer(pcbnew.F_SilkS)
        sseg.SetWidth(MM(0.3))
        board.Add(sseg)
    text("z=0", FIDUCIAL_X + 4.5, 8.2, 0.9)
    for i, zz in enumerate(STATIONS_MM):
        sx = FIDUCIAL_X + zz
        text(f"z={zz:+d}", sx + 4.5, 7.4, 0.8)
    text("EMIT [  ]  RECV [  ]", 46, 8.0, 1.1)
    text("OMNIMARBLE GATE RAIL", 110, 8.6, 1.3)

    filler = pcbnew.ZONE_FILLER(board)
    filler.Fill(board.Zones())
    pcbnew.SaveBoard(str(PCB_OUT), board)
    board.BuildConnectivity()
    unconn = board.GetConnectivity().GetUnconnectedCount(True)
    print(f"SAVED {PCB_OUT.name}: unconnected={unconn}")


if __name__ == "__main__":
    main()
