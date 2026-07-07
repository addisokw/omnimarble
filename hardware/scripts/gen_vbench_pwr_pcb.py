"""Generate vbench-pwr.kicad_pcb: 2-layer 2oz power board (freerouting flow).

Placement + B.Cu GND pour only; the signal/power copper is autorouted by
freerouting (this board is 2-D, unlike the planar gate-rail). Mirrors the
parity metadata sync of gen_rail_pcb.py.

Pipeline:
  gen_vbench_pwr_pcb.py            -> place-only board (footprints + GND pour)
  gen_vbench_pwr_pcb.py dsn        -> export vbench-pwr.dsn (tracks stripped)
  <java -jar freerouting.jar -de ...dsn -do ...ses -mp 30 -mt 1>
  gen_vbench_pwr_pcb.py import     -> import the .ses, refill pours, save

Run under KiCad python.
"""

import sys
from pathlib import Path

import pcbnew

HW_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HW_DIR / "scripts"))
from kicad_sch import Sym, find, find_all, parse

PROJ = HW_DIR / "vbench-pwr"
NETLIST = PROJ / "vbench-pwr.net"
PCB_OUT = PROJ / "vbench-pwr.kicad_pcb"
DSN = PROJ / "vbench-pwr.dsn"
SES = PROJ / "vbench-pwr.ses"
FP_SHARE = Path(r"C:\Program Files\KiCad\10.0\share\kicad\footprints")
FP_LOCAL = [HW_DIR / "omnimarble-driver", HW_DIR / "gate-rail"]

MM = pcbnew.FromMM
BOARD_W, BOARD_H = 150.0, 136.0
DNP_REFS = {"C2", "C3", "C4", "C5", "Q2", "Q3", "RG2", "RG3"}


def V(x, y):
    return pcbnew.VECTOR2I(MM(x), MM(y))


def load_netlist():
    tree = parse(NETLIST.read_text(encoding="utf-8"))[0]
    comps, meta = {}, {}
    for c in find_all(find(tree, Sym("components")), Sym("comp")):
        ref = str(find(c, Sym("ref"))[1])
        fp = find(c, Sym("footprint"))
        comps[ref] = str(fp[1]) if fp else ""
        val = find(c, Sym("value"))
        lcsc = ""
        fields = find(c, Sym("fields"))
        if fields:
            for f in find_all(fields, Sym("field")):
                nm = find(f, Sym("name"))
                if nm and len(f) > 2 and str(nm[1]) == "LCSC":
                    lcsc = str(f[2])
        meta[ref] = (str(val[1]) if val else "", lcsc)
    padnet = {}
    for n in find_all(find(tree, Sym("nets")), Sym("net")):
        name = str(find(n, Sym("name"))[1])
        for node in find_all(n, Sym("node")):
            padnet[(str(find(node, Sym("ref"))[1]),
                    str(find(node, Sym("pin"))[1]))] = name
    return comps, padnet, meta


def load_footprint(fpid):
    lib, name = fpid.split(":", 1)
    for base in (FP_SHARE, *FP_LOCAL):
        p = base / f"{lib}.pretty"
        if p.exists():
            fp = pcbnew.FootprintLoad(str(p), name)
            if fp:
                return fp
    raise RuntimeError(f"footprint not found: {fpid}")


# placement (ref -> x,y,rot). 30mm caps dominate: all mid/lower parts sit BELOW
# the cap courtyards (y>=44) so x-alignment with the cap columns is harmless.
P = {}
for i in range(5):
    P[f"C{i+1}"] = (24 + i * 20, 22, 270)      # 18mm bank, pad1 VBANK up, pad2 GND down
P["J1"] = (14, 52, 0)                          # COIL
for i in range(3):
    P[f"Q{i+1}"] = (38 + i * 24, 52, 0)        # FETs G(1) D(2)=SW S(3)=SRC
    P[f"RG{i+1}"] = (98 + i * 8, 70, 0)
P["RS1"] = (110, 52, 0)                        # shunt
P["D1"] = (134, 52, 180)                       # freewheel K(1)=VBANK A(2)=SW
P["U1"] = (50, 72, 0)
P["RGS"] = (34, 72, 0)
P["DZ1"] = (64, 72, 0)
P["RF"] = (78, 70, 0)
P["RFP"] = (86, 70, 0)
P["C10"] = (40, 66, 0)
P["C11"] = (46, 66, 0)
P["RD1"] = (124, 66, 90)
P["RD2"] = (124, 74, 90)
P["C12"] = (131, 70, 90)
P["RB1"] = (140, 70, 90)
P["J2"] = (18, 84, 0)                          # CHARGE
P["J3"] = (48, 84, 0)                          # AUX-BANK
P["J4"] = (78, 84, 0)                          # 12V-IN
P["J5"] = (104, 84, 0)                         # IFACE
P["J6"] = (130, 84, 0)                         # ISENSE
# logic zone (bottom, separated from the pulse loop by the GND plane + distance)
P["J7"] = (24, 104, 90)     # PICO-L  (pins along x)
P["J8"] = (24, 122, 90)     # PICO-R  (17.78mm below)
P["J9"] = (86, 104, 0)      # WAVESHARE-A
P["J10"] = (102, 104, 0)    # WAVESHARE-B
P["J11"] = (118, 104, 0)    # WAVESHARE-C
P["J12"] = (134, 104, 0)    # WAVESHARE-D


def build_place():
    comps, padnet, meta = load_netlist()
    if PCB_OUT.exists():
        PCB_OUT.unlink()
    board = pcbnew.NewBoard(str(PCB_OUT))
    board.SetCopperLayerCount(2)

    pts = [(0, 0), (BOARD_W, 0), (BOARD_W, BOARD_H), (0, BOARD_H)]
    for i in range(4):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(V(*pts[i])); seg.SetEnd(V(*pts[(i + 1) % 4]))
        seg.SetLayer(pcbnew.Edge_Cuts); seg.SetWidth(MM(0.15))
        board.Add(seg)

    nets = {}

    def netinfo(name):
        if name not in nets:
            ni = pcbnew.NETINFO_ITEM(board, name); board.Add(ni); nets[name] = ni
        return nets[name]

    for ref, fpid in sorted(comps.items()):
        fp = load_footprint(fpid)
        fp.SetReference(ref)
        if ":" in fpid:
            ln, fn = fpid.split(":", 1)
            try:
                fp.SetFPID(pcbnew.LIB_ID(ln, fn))
            except Exception:
                pass
        if ref in meta:
            val, lcsc = meta[ref]
            if val:
                fp.SetValue(val)
            if lcsc:
                try:
                    fp.SetField("LCSC", lcsc)
                    for fld in fp.GetFields():
                        if fld.GetName() == "LCSC":
                            fld.SetVisible(False)
                except Exception:
                    pass
        for pad in fp.Pads():
            key = (ref, pad.GetNumber())
            if key in padnet:
                pad.SetNet(netinfo(padnet[key]))
        if ref in DNP_REFS:
            try:
                fp.SetDNP(True)
            except Exception:
                pass
        try:
            fp.Reference().SetLayer(pcbnew.F_Fab)
            fp.Value().SetVisible(False)
        except Exception:
            pass
        x, y, rot = P[ref]
        fp.SetPosition(V(x, y)); fp.SetOrientationDegrees(rot)
        board.Add(fp)

    for i, (hx, hy) in enumerate([(6, 6), (144, 6), (6, 130), (144, 130)]):
        fp = load_footprint("MountingHole:MountingHole_3.2mm_M3_Pad")
        try:
            fp.SetExcludedFromBOM(True); fp.SetBoardOnly(True)
        except Exception:
            pass
        fp.SetReference(f"H{i+1}")
        fp.SetPosition(V(hx, hy))
        board.Add(fp)

    # B.Cu GND pour (return plane; freerouting routes signal + drops GND vias)
    z = pcbnew.ZONE(board)
    z.SetLayer(pcbnew.B_Cu); z.SetNet(netinfo("GND"))
    o = z.Outline(); o.NewOutline()
    for x, y in [(0.6, 0.6), (BOARD_W - 0.6, 0.6),
                 (BOARD_W - 0.6, BOARD_H - 0.6), (0.6, BOARD_H - 0.6)]:
        o.Append(MM(x), MM(y))
    z.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
    z.SetMinThickness(MM(0.25))
    board.Add(z)

    t = pcbnew.PCB_TEXT(board)
    t.SetText("OMNIMARBLE VBENCH-PWR  (charge <=55V; star GND at bank -)")
    t.SetPosition(V(40, 10)); t.SetLayer(pcbnew.F_SilkS)
    t.SetTextSize(pcbnew.VECTOR2I(MM(1.4), MM(1.4))); t.SetTextThickness(MM(0.25))
    board.Add(t)

    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    pcbnew.SaveBoard(str(PCB_OUT), board)
    print(f"SAVED place-only {PCB_OUT.name}")


def export_dsn():
    board = pcbnew.LoadBoard(str(PCB_OUT))
    for tr in list(board.Tracks()):
        board.Remove(tr)
    ok = pcbnew.ExportSpecctraDSN(board, str(DSN))
    print(f"DSN export: {ok} -> {DSN.name}")


def import_ses():
    board = pcbnew.LoadBoard(str(PCB_OUT))
    for tr in list(board.Tracks()):
        board.Remove(tr)
    ok = pcbnew.ImportSpecctraSES(board, str(SES))
    # freerouting's B.Cu routing traps a small GND pocket around the Pico/sensor
    # pins; stitch it to the main plane with an F.Cu track between a pocket GND pad
    # (J9.2) and an adjacent main-plane GND pad (J10.2). Both are THT, so the track
    # ties the two B.Cu pours through the pads. (freerouting is deterministic, so
    # the pocket is stable for this placement.)
    gnd = board.FindNet("GND")

    def padxy(ref, num):
        for fp in board.Footprints():
            if fp.GetReference() == ref:
                for p in fp.Pads():
                    if p.GetNumber() == num:
                        pp = p.GetPosition()
                        return pcbnew.ToMM(pp.x), pcbnew.ToMM(pp.y)
        raise KeyError((ref, num))

    (x0, y0), (x1, y1) = padxy("J9", "2"), padxy("J10", "2")
    t = pcbnew.PCB_TRACK(board)
    t.SetStart(V(x0, y0)); t.SetEnd(V(x1, y1))
    t.SetWidth(MM(0.6)); t.SetLayer(pcbnew.F_Cu); t.SetNet(gnd); board.Add(t)
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    pcbnew.SaveBoard(str(PCB_OUT), board)
    print(f"SES import: {ok} (run kicad-cli drc for the unconnected count)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "place"
    {"place": build_place, "dsn": export_dsn, "import": import_ses}[mode]()
