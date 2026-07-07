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

import re
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
# 4 sensor connectors (20 ch); GND is a solid inner plane on 4-layer, so signals
# route freely across F.Cu + B.Cu and the plane never fragments.
P["J9"] = (86, 104, 0)      # WAVESHARE-A
P["J10"] = (102, 104, 0)    # WAVESHARE-B
P["J11"] = (118, 104, 0)    # WAVESHARE-C
P["J12"] = (134, 104, 0)    # WAVESHARE-D


def build_place():
    comps, padnet, meta = load_netlist()
    if PCB_OUT.exists():
        PCB_OUT.unlink()
    board = pcbnew.NewBoard(str(PCB_OUT))
    board.SetCopperLayerCount(4)   # F.Cu | In1 GND plane | In2 GND plane | B.Cu
    for lyr in (pcbnew.In1_Cu, pcbnew.In2_Cu):
        try:
            board.SetLayerType(lyr, pcbnew.LT_POWER)
        except Exception:
            pass                    # DSN plane_lock enforces the plane type anyway

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

    # Solid GND planes on the two inner layers. Signals route on F.Cu + B.Cu
    # (2 layers = routes cleanly); the inner planes are never cut, so GND cannot
    # fragment -- freerouting just drops GND-pad vias into them. No F.Cu/B.Cu GND
    # pour (that would re-introduce the fragmentation this 4-layer stackup fixes).
    def rect_zone(layer, net, box, prio=0):
        z = pcbnew.ZONE(board)
        z.SetLayer(layer); z.SetNet(netinfo(net))
        x0, y0, x1, y1 = box
        o = z.Outline(); o.NewOutline()
        for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
            o.Append(MM(x), MM(y))
        z.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
        z.SetMinThickness(MM(0.25))
        z.SetAssignedPriority(prio)
        board.Add(z)

    full = (0.6, 0.6, BOARD_W - 0.6, BOARD_H - 0.6)
    rect_zone(pcbnew.In1_Cu, "GND", full)
    rect_zone(pcbnew.In2_Cu, "GND", full)
    # +3V3 pocket on In2 over the logic zone: all +3V3 pins are THT, so they connect
    # straight into this pocket (higher priority carves it out of the In2 GND plane)
    # -- no F.Cu/B.Cu +3V3 routing, hence no crossings. All SMD GND via-drops sit at
    # y<90, clear of this pocket, so GND through-vias never short to it.
    rect_zone(pcbnew.In2_Cu, "+3V3", (55, 100, 141, 126), prio=1)

    t = pcbnew.PCB_TEXT(board)
    t.SetText("OMNIMARBLE VBENCH-PWR  (charge <=55V; star GND at bank -)")
    t.SetPosition(V(40, 10)); t.SetLayer(pcbnew.F_SilkS)
    t.SetTextSize(pcbnew.VECTOR2I(MM(1.4), MM(1.4))); t.SetTextThickness(MM(0.25))
    board.Add(t)

    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    pcbnew.SaveBoard(str(PCB_OUT), board)
    print(f"SAVED place-only {PCB_OUT.name}")


def plane_lock_dsn():
    """Mark the two inner layers `(type power)` in the exported DSN so freerouting
    treats them as plane-only: signals route on F.Cu + B.Cu, GND-pad vias drop into
    the inner GND planes, and the planes are NEVER cut by signal tracks. This is the
    driver's dedicated-inner-plane pattern and makes re-routing reproducible (no
    fragile post-hoc GND stitching). KiCad usually already emits inner layers as
    power (we set LT_POWER), but flip any that come out signal to be safe."""
    text = DSN.read_text(encoding="utf-8")
    n = 0
    for layer in ("In1.Cu", "In2.Cu"):
        text, k = re.subn(r"(\(layer " + re.escape(layer) + r"\s*\n\s*)\(type signal\)",
                          r"\1(type power)", text, count=1)
        n += k
    DSN.write_text(text, encoding="utf-8")
    print(f"plane_lock: {n} inner layer(s) forced -> (type power) "
          "[solid inner GND planes]")


def export_dsn():
    board = pcbnew.LoadBoard(str(PCB_OUT))
    for tr in list(board.Tracks()):
        board.Remove(tr)
    ok = pcbnew.ExportSpecctraDSN(board, str(DSN))
    plane_lock_dsn()
    print(f"DSN export: {ok} -> {DSN.name}")


def import_ses():
    board = pcbnew.LoadBoard(str(PCB_OUT))
    for tr in list(board.Tracks()):
        board.Remove(tr)
    ok = pcbnew.ImportSpecctraSES(board, str(SES))
    # The inner GND planes are solid, but freerouting doesn't via SMD GND pads down
    # to them (THT GND pads already pass through). Drop a GND through-via at each SMD
    # GND pad -> ties it to both inner planes. Deterministic; no fragile stitching,
    # since the planes are never cut. Same-net so no clearance conflict.
    gnd = board.FindNet("GND")
    n_via = 0
    for fp in board.Footprints():
        for p in fp.Pads():
            if p.GetNetname() == "GND" and p.GetAttribute() != pcbnew.PAD_ATTRIB_PTH:
                v = pcbnew.PCB_VIA(board)
                v.SetViaType(pcbnew.VIATYPE_THROUGH)
                v.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
                v.SetPosition(p.GetPosition())
                v.SetDrill(MM(0.3)); v.SetWidth(MM(0.6))
                v.SetNet(gnd)
                board.Add(v)
                n_via += 1
    # +3V3 (all-THT: Pico J8.16 + 4 Waveshare pin-1s) connects through the In2 +3V3
    # pocket built in build_place -- no F.Cu/B.Cu routing, so no crossings.
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    pcbnew.SaveBoard(str(PCB_OUT), board)
    print(f"SES import: {ok}; {n_via} SMD-GND via-drops to inner planes")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "place"
    {"place": build_place, "dsn": export_dsn, "import": import_ses}[mode]()
