"""Generate omnimarble-driver.kicad_pcb (4-layer) from the netlist.

MUST run under KiCad's bundled python (pcbnew API):
  "C:\\Program Files\\KiCad\\10.0\\bin\\python.exe" hardware/scripts/gen_pcb.py

Pipeline: netlist -> board + footprints + nets -> placement (placement.py)
-> pulse zones (F.Cu + B.Cu mirror, via-stitched) + In1/In2 GND planes
-> pulse pad-in-zone assertions (+ explicit stub bridges for package
straddlers) -> GND via-drops -> grid-router for signal nets -> zone fill
-> save + connectivity report. DRC runs separately via kicad-cli.
"""

import json
import math
import random
import sys
from pathlib import Path

import pcbnew

HW_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = HW_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS))

from kicad_sch import Sym, find, find_all, parse  # stdlib-only parser
from placement import (AUTO_CLUSTERS, BOARD_H, BOARD_W, BRIDGE_TABLE,
                       CRITICAL_NETS, KELVIN_TRACKS, MANUAL_TRACKS, MOUNT_HOLES,
                       NET_WIDTHS, P, PLANE_NETS, PULSE_NETS, PULSE_ZONES,
                       VIA_CLUSTERS)

PROJ = HW_DIR / "omnimarble-driver"
NETLIST = PROJ / "omnimarble-driver.net"
PCB_OUT = PROJ / "omnimarble-driver.kicad_pcb"
FP_SHARE = Path(r"C:\Program Files\KiCad\10.0\share\kicad\footprints")

MM = pcbnew.FromMM


def V(x, y):
    return pcbnew.VECTOR2I(MM(x), MM(y))


# --------------------------------------------------------------------------
# Netlist
# --------------------------------------------------------------------------

def load_netlist():
    tree = parse(NETLIST.read_text(encoding="utf-8"))[0]
    comps = {}
    meta = {}
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
                if nm and str(nm[1]) == "LCSC" and len(f) > 2:
                    lcsc = str(f[2])
        meta[ref] = (str(val[1]) if val else "", lcsc)
    nets = {}
    padnet = {}
    for n in find_all(find(tree, Sym("nets")), Sym("net")):
        name = str(find(n, Sym("name"))[1])
        for node in find_all(n, Sym("node")):
            ref = str(find(node, Sym("ref"))[1])
            pin = str(find(node, Sym("pin"))[1])
            nets.setdefault(name, []).append((ref, pin))
            padnet[(ref, pin)] = name
    return comps, nets, padnet, meta


# --------------------------------------------------------------------------
# Board scaffolding
# --------------------------------------------------------------------------

def new_board():
    if PCB_OUT.exists():
        PCB_OUT.unlink()
    board = pcbnew.NewBoard(str(PCB_OUT))
    board.SetCopperLayerCount(4)
    ds = board.GetDesignSettings()
    ds.SetBoardThickness(MM(1.6))
    return board


def add_outline(board):
    pts = [(0, 0), (BOARD_W, 0), (BOARD_W, BOARD_H), (0, BOARD_H)]
    for i in range(4):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(V(*pts[i]))
        seg.SetEnd(V(*pts[(i + 1) % 4]))
        seg.SetLayer(pcbnew.Edge_Cuts)
        seg.SetWidth(MM(0.1))
        board.Add(seg)


def add_text(board, s, x, y, size=2.0, layer=None):
    t = pcbnew.PCB_TEXT(board)
    t.SetText(s)
    t.SetPosition(V(x, y))
    t.SetLayer(layer if layer is not None else pcbnew.F_SilkS)
    t.SetTextSize(pcbnew.VECTOR2I(MM(size), MM(size)))
    t.SetTextThickness(MM(size * 0.15))
    board.Add(t)


def load_footprint(fpid):
    lib, name = fpid.split(":", 1)
    for base in (FP_SHARE, PROJ, HW_DIR / "gate-rail"):
        p = base / f"{lib}.pretty"
        if p.exists():
            fp = pcbnew.FootprintLoad(str(p), name)
            if fp:
                return fp
    raise RuntimeError(f"footprint not found: {fpid}")


def place_all(board, comps, padnet, meta=None):
    nets = {}

    def netinfo(name):
        if name not in nets:
            ni = pcbnew.NETINFO_ITEM(board, name)
            board.Add(ni)
            nets[name] = ni
        return nets[name]

    auto_refs = {}
    for rect, refs in AUTO_CLUSTERS:
        for r in refs:
            auto_refs[r] = rect

    missing = []
    pending_auto = {}
    for ref, fpid in sorted(comps.items()):
        if not fpid:
            missing.append(ref)
            continue
        fp = load_footprint(fpid)
        fp.SetReference(ref)
        # schematic-parity metadata: correct LIB_ID, symbol value, LCSC field
        lib_nick, fp_name = fpid.split(":", 1)
        try:
            fp.SetFPID(pcbnew.LIB_ID(lib_nick, fp_name))
        except Exception:
            pass
        if meta and ref in meta:
            val, lcsc = meta[ref]
            if val:
                fp.SetValue(val)
            if lcsc:
                try:
                    fp.SetField("LCSC", lcsc)
                except Exception:
                    pass
        for pad in fp.Pads():
            key = (ref, pad.GetNumber())
            if key in padnet:
                pad.SetNet(netinfo(padnet[key]))
        if ref in P:
            x, y, rot = P[ref]
            fp.SetPosition(V(x, y))
            fp.SetOrientationDegrees(rot)
            board.Add(fp)
        elif ref in auto_refs:
            pending_auto[ref] = fp
        else:
            missing.append(ref)

    # shelf-pack the auto clusters using real bounding boxes
    GAP = 1.4
    for rect, refs in AUTO_CLUSTERS:
        x0, y0, x1, y1 = rect
        cx, cy, row_h = x0, y0, 0.0
        for ref in refs:
            fp = pending_auto.get(ref)
            if fp is None:
                continue
            fp.SetPosition(V(0, 0))
            bb = fp.GetBoundingBox(False)
            w = pcbnew.ToMM(bb.GetWidth()) + GAP
            h = pcbnew.ToMM(bb.GetHeight()) + GAP
            if cx + w > x1:
                cx = x0
                cy += row_h
                row_h = 0.0
            # anchor offset: bbox may not be centred on origin
            offx = pcbnew.ToMM(bb.GetLeft())
            offy = pcbnew.ToMM(bb.GetTop())
            fp.SetPosition(V(cx - offx + GAP / 2, cy - offy + GAP / 2))
            board.Add(fp)
            cx += w
            row_h = max(row_h, h)
            if cy + row_h > y1 + 3:
                print(f"  SHELF OVERFLOW in cluster {rect} at {ref}")
    # mounting holes
    for i, (x, y) in enumerate(MOUNT_HOLES):
        fp = load_footprint(
            "MountingHole:MountingHole_3.2mm_M3_Pad")
        try:
            fp.SetExcludedFromBOM(True)
            fp.SetBoardOnly(True)
        except Exception:
            pass
        fp.SetReference(f"H{i + 1}")
        fp.SetPosition(V(x, y))
        board.Add(fp)
    return nets, missing


# --------------------------------------------------------------------------
# Zones
# --------------------------------------------------------------------------

def add_zone(board, netinfo, layer, pts, clearance=0.3, min_thick=0.25,
             priority=1):
    z = pcbnew.ZONE(board)
    z.SetLayer(layer)
    if netinfo is not None:
        z.SetNet(netinfo)
    outline = z.Outline()
    outline.NewOutline()
    for x, y in pts:
        outline.Append(MM(x), MM(y))
    z.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
    z.SetLocalClearance(MM(clearance))
    z.SetMinThickness(MM(min_thick))
    z.SetAssignedPriority(priority)
    board.Add(z)
    return z


def add_edge_keepout(board, margin=1.5):
    """Perimeter rule-area ring that keeps tracks + vias `margin` mm off the
    board edge (so wide rails clear the 0.5mm copper-edge rule even with their
    half-width), while still allowing the GND pour to flood up to its own 2mm
    inset. Only J1.3 sits within 1.6mm of an edge (at 1.55mm) and stays
    outside a 1.5mm ring, so no pad is orphaned."""
    W, H, m = BOARD_W, BOARD_H, margin
    strips = [(0, 0, W, m), (0, H - m, W, H), (0, 0, m, H), (W - m, 0, W, H)]
    for (x0, y0, x1, y1) in strips:
        z = pcbnew.ZONE(board)
        z.SetIsRuleArea(True)
        z.SetDoNotAllowTracks(True)
        z.SetDoNotAllowVias(True)
        z.SetLayerSet(pcbnew.LSET.AllCuMask())
        o = z.Outline()
        o.NewOutline()
        for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
            o.Append(MM(x), MM(y))
        board.Add(z)


# Fine-pitch ICs the full-board F/B GND pour boxes in, blocking signal-pad
# escape (DeepPCB flagged U10/U11/U12/U4/U1 pins as physically unroutable).
POUR_KEEPOUT_ICS = ["U10", "U11", "U12", "U4", "U1", "U9", "U2"]


def add_pour_keepouts(board, refs, margin=0.8):
    """Rule areas that keep the F/B GND pour OFF the listed ICs (tracks + vias
    still allowed) so signal pads can escape. Only F.Cu/B.Cu -- the inner GND
    planes stay solid, so each IC's GND pad still connects via an inner-plane
    via."""
    ls = pcbnew.LSET()
    ls.AddLayer(pcbnew.F_Cu)
    ls.AddLayer(pcbnew.B_Cu)
    for ref in refs:
        fp = board.FindFootprintByReference(ref)
        if fp is None:
            continue
        bb = fp.GetBoundingBox(False)
        x0 = pcbnew.ToMM(bb.GetLeft()) - margin
        x1 = pcbnew.ToMM(bb.GetRight()) + margin
        y0 = pcbnew.ToMM(bb.GetTop()) - margin
        y1 = pcbnew.ToMM(bb.GetBottom()) + margin
        z = pcbnew.ZONE(board)
        z.SetIsRuleArea(True)
        z.SetDoNotAllowZoneFills(True)
        z.SetDoNotAllowTracks(False)
        z.SetDoNotAllowVias(False)
        z.SetLayerSet(ls)
        o = z.Outline()
        o.NewOutline()
        for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
            o.Append(MM(x), MM(y))
        board.Add(z)


def add_fb_gnd_pours(board, nets):
    """Full-board F/B GND flood, added AFTER routing so it pours around the
    finished traces (thermal-connecting GND pads) instead of boxing fine-pitch
    IC signal pads before they are routed. The inner In1/In2 GND planes carry
    the return; these outer pours add shielding + short GND-pad connections."""
    frame = [(2, 2), (BOARD_W - 2, 2), (BOARD_W - 2, BOARD_H - 2),
             (2, BOARD_H - 2)]
    for layer in (pcbnew.F_Cu, pcbnew.B_Cu):
        add_zone(board, nets["GND"], layer, frame, clearance=0.2,
                 min_thick=0.13, priority=0)


def build_zones(board, nets):
    add_edge_keepout(board)
    zones = []
    prio = 2
    for name, pts in PULSE_ZONES:
        prio += 1
        for layer in (pcbnew.F_Cu, pcbnew.B_Cu):
            zones.append((name, pts,
                          add_zone(board, nets[name], layer, pts,
                                   clearance=1.0, min_thick=0.5,
                                   priority=prio)))
    # Inner layers: solid GND planes (the return path). The OUTER F/B GND
    # flood is NOT poured here -- pouring it before routing boxes fine-pitch IC
    # signal pads. It is added after the route lands (add_fb_gnd_pours, called
    # from import-clean) so it flows around the finished traces instead.
    frame = [(2, 2), (BOARD_W - 2, 2), (BOARD_W - 2, BOARD_H - 2),
             (2, BOARD_H - 2)]
    for layer in (pcbnew.In1_Cu, pcbnew.In2_Cu):
        add_zone(board, nets["GND"], layer, frame, clearance=0.3,
                 priority=0)
    return zones


def stitch_zone(board, g, nets, name, pts, pitch=5.0):
    """Via-stitch a pulse zone polygon interior (F<->B<->planes),
    skipping lattice points that collide with foreign copper."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # THT pad holes to keep 0.6mm-via stitch points clear of (hole-to-hole)
    tht = []
    for fp in board.Footprints():
        for pad in fp.Pads():
            dr = pad.GetDrillSize()
            if dr.x > 0 and pad.GetAttribute() != pcbnew.PAD_ATTRIB_SMD:
                pp = pad.GetPosition()
                tht.append((pcbnew.ToMM(pp.x), pcbnew.ToMM(pp.y),
                            0.3 + pcbnew.ToMM(dr.x) / 2 + 0.25))
    added = 0
    y = min(ys) + 2.0
    while y < max(ys) - 1.0:
        x = min(xs) + 2.0
        while x < max(xs) - 1.0:
            if point_in_poly(x, y, pts):
                ix, iy = int(round(x / GRID)), int(round(y / GRID))
                ok = all(g.is_free(l, ix + dx, iy + dy, name)
                         for l in (0, 1)
                         for dx in (-1, 0, 1) for dy in (-1, 0, 1))
                if ok and not any((x - hx) ** 2 + (y - hy) ** 2 < hr * hr
                                  for hx, hy, hr in tht):
                    add_via(board, g, nets, name, x, y, 0.6, 1.2)
                    added += 1
            x += pitch
        y += pitch
    return added


def point_in_poly(x, y, pts):
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and \
                x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


# --------------------------------------------------------------------------
# Pulse pad-in-zone assertions + stub bridges
# --------------------------------------------------------------------------

# (ref, pad) -> (dx, dy, width): draw a wide F.Cu stub from the pad center
# by (dx,dy) into its zone for package straddlers.
BRIDGES = BRIDGE_TABLE


def check_pulse_pads(board, zones):
    zone_by_net = {}
    for name, pts, _ in zones:
        zone_by_net.setdefault(name, []).append(pts)
    misses = []
    for fp in board.Footprints():
        for pad in fp.Pads():
            net = pad.GetNetname()
            if net not in PULSE_NETS:
                continue
            pos = pad.GetPosition()
            x, y = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
            if not any(point_in_poly(x, y, pts)
                       for pts in zone_by_net.get(net, [])):
                misses.append((fp.GetReference(), pad.GetNumber(), net,
                               round(x, 2), round(y, 2)))
    return misses


def add_bridges(board, nets, padnet):
    for (ref, padnum), (dx, dy, w) in BRIDGES.items():
        fp = board.FindFootprintByReference(ref)
        if not fp:
            continue
        for pad in fp.Pads():
            if pad.GetNumber() == padnum:
                s = pad.GetPosition()
                t = pcbnew.PCB_TRACK(board)
                t.SetStart(s)
                t.SetEnd(pcbnew.VECTOR2I(s.x + MM(dx), s.y + MM(dy)))
                t.SetWidth(MM(w))
                t.SetLayer(pcbnew.F_Cu)
                t.SetNet(pad.GetNet())
                board.Add(t)


# --------------------------------------------------------------------------
# GND via drops for SMD pads
# --------------------------------------------------------------------------

def gnd_via_drops(board, nets, occupied):
    gnd = nets["GND"]
    added = 0
    for fp in board.Footprints():
        for pad in fp.Pads():
            if pad.GetNetname() != "GND":
                continue
            if pad.GetAttribute() != pcbnew.PAD_ATTRIB_SMD:
                continue  # THT reaches the planes directly
            pos = pad.GetPosition()
            px, py = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
            # find a free spot near the pad for the via
            for ddx, ddy in [(0, 1.4), (0, -1.4), (1.4, 0), (-1.4, 0),
                             (1.2, 1.2), (-1.2, 1.2), (1.2, -1.2),
                             (-1.2, -1.2), (0, 2.2), (0, -2.2)]:
                vx, vy = px + ddx, py + ddy
                key = (round(vx * 2) / 2, round(vy * 2) / 2)
                if key in occupied:
                    continue
                if not (2 < vx < BOARD_W - 2 and 2 < vy < BOARD_H - 2):
                    continue
                occupied.add(key)
                via = pcbnew.PCB_VIA(board)
                via.SetPosition(V(vx, vy))
                via.SetDrill(MM(0.3))
                via.SetWidth(MM(0.6))
                via.SetViaType(pcbnew.VIATYPE_THROUGH)
                via.SetNet(gnd)
                board.Add(via)
                t = pcbnew.PCB_TRACK(board)
                t.SetStart(pos)
                t.SetEnd(V(vx, vy))
                t.SetWidth(MM(0.4))
                t.SetLayer(pad.GetLayer() if pad.IsOnLayer(pcbnew.F_Cu)
                           else pcbnew.B_Cu)
                t.SetNet(gnd)
                board.Add(t)
                added += 1
                break
    return added


# --------------------------------------------------------------------------
# Grid router (rectilinear A*, F.Cu + B.Cu)
# --------------------------------------------------------------------------

GRID = 0.5  # mm

# owner sentinel for cells claimed by two different-net pads: blocked for
# every net and never whitelisted by filter_extra
CONTESTED = "__x__"


class Grid:
    def __init__(self, w, h):
        self.nx = int(w / GRID) + 1
        self.ny = int(h / GRID) + 1
        self.blocked = [bytearray(self.nx * self.ny) for _ in range(2)]
        self.owner = [{} for _ in range(2)]

    def idx(self, ix, iy):
        return ix * self.ny + iy

    def block_cell(self, layer, ix, iy, net=None):
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            i = self.idx(ix, iy)
            self.blocked[layer][i] = 1
            if net is not None:
                self.owner[layer][i] = net
            elif i in self.owner[layer]:
                del self.owner[layer][i]

    def block(self, layer, x, y, net=None):
        self.block_cell(layer, int(round(x / GRID)), int(round(y / GRID)), net)

    def block_rect(self, layer, x0, y0, x1, y1, net=None):
        ix0 = max(0, int(x0 / GRID))
        ix1 = min(self.nx - 1, int(math.ceil(x1 / GRID)))
        iy0 = max(0, int(y0 / GRID))
        iy1 = min(self.ny - 1, int(math.ceil(y1 / GRID)))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                self.block_cell(layer, ix, iy, net)

    def is_free(self, layer, ix, iy, net, extra=frozenset()):
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return False
        i = self.idx(ix, iy)
        if not self.blocked[layer][i]:
            return True
        if (layer, ix, iy) in extra:
            return True
        return self.owner[layer].get(i) == net


# Zones that hard-block routing (per-cell polygon test); SHUNT_HI stays
# routable (signal tracks legally perforate it), GND zones stay routable.
ROUTING_BLOCK_ZONES = {"VBANK", "COIL_HI", "SW_DRAIN"}

# via keepout: (ix, iy) -> zone net; vias of other nets may not land here
VIA_FORBID = {}


def build_via_forbid():
    VIA_FORBID.clear()
    for name, pts in PULSE_ZONES:
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        ix0 = max(0, int((min(xs) - 1) / GRID))
        ix1 = int(math.ceil((max(xs) + 1) / GRID))
        iy0 = max(0, int((min(ys) - 1) / GRID))
        iy1 = int(math.ceil((max(ys) + 1) / GRID))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                if point_in_poly_margin(ix * GRID, iy * GRID, pts, 1.0):
                    VIA_FORBID[(ix, iy)] = name


def via_ok(ix, iy, net):
    z = VIA_FORBID.get((ix, iy))
    return z is None or z == net


def build_grid(board):
    g = Grid(BOARD_W, BOARD_H)
    for layer in (0, 1):
        g.block_rect(layer, 0, 0, BOARD_W, 1.4)
        g.block_rect(layer, 0, BOARD_H - 1.4, BOARD_W, BOARD_H)
        g.block_rect(layer, 0, 0, 1.4, BOARD_H)
        g.block_rect(layer, BOARD_W - 1.4, 0, BOARD_W, BOARD_H)
    # zone interiors (with 1.0 margin)
    for name, pts in PULSE_ZONES:
        if name not in ROUTING_BLOCK_ZONES:
            continue
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        ix0, ix1 = int((min(xs) - 1) / GRID), int(math.ceil((max(xs) + 1) / GRID))
        iy0, iy1 = int((min(ys) - 1) / GRID), int(math.ceil((max(ys) + 1) / GRID))
        for ix in range(max(0, ix0), min(g.nx, ix1 + 1)):
            for iy in range(max(0, iy0), min(g.ny, iy1 + 1)):
                x, y = ix * GRID, iy * GRID
                if point_in_poly_margin(x, y, pts, 1.0):
                    g.block_cell(0, ix, iy, name)
                    g.block_cell(1, ix, iy, name)
    # pads: world-space bounding boxes + clearance
    clr = 0.45
    for fp in board.Footprints():
        for pad in fp.Pads():
            bb = pad.GetBoundingBox()
            x0 = pcbnew.ToMM(bb.GetLeft()) - clr
            x1 = pcbnew.ToMM(bb.GetRight()) + clr
            y0 = pcbnew.ToMM(bb.GetTop()) - clr
            y1 = pcbnew.ToMM(bb.GetBottom()) + clr
            net = pad.GetNetname() or "__nc__"
            if pad.GetAttribute() == pcbnew.PAD_ATTRIB_SMD:
                layers = [0] if pad.IsOnLayer(pcbnew.F_Cu) else [1]
            else:
                layers = [0, 1]
            for layer in layers:
                ix0 = max(0, int(x0 / GRID))
                ix1 = min(g.nx - 1, int(math.ceil(x1 / GRID)))
                iy0 = max(0, int(y0 / GRID))
                iy1 = min(g.ny - 1, int(math.ceil(y1 / GRID)))
                for ix in range(ix0, ix1 + 1):
                    for iy in range(iy0, iy1 + 1):
                        i = g.idx(ix, iy)
                        prev = (g.owner[layer].get(i)
                                if g.blocked[layer][i] else None)
                        if prev is not None and prev != net:
                            # cell inside the clearance ring of TWO
                            # different-net pads: hard for everyone --
                            # a single-owner cell would let the later
                            # pad's net route over the earlier pad's
                            # copper (top DRC short mechanism)
                            g.block_cell(layer, ix, iy, CONTESTED)
                        else:
                            g.block_cell(layer, ix, iy, net)
    return g


def point_in_poly_margin(x, y, pts, margin):
    if point_in_poly(x, y, pts):
        return True
    for dx, dy in ((margin, 0), (-margin, 0), (0, margin), (0, -margin)):
        if point_in_poly(x + dx, y + dy, pts):
            return True
    return False


def pad_cells(pad):
    """Grid cells covered by a pad's bounding box (for endpoint whitelisting)."""
    bb = pad.GetBoundingBox()
    x0 = pcbnew.ToMM(bb.GetLeft()) - 0.4
    x1 = pcbnew.ToMM(bb.GetRight()) + 0.4
    y0 = pcbnew.ToMM(bb.GetTop()) - 0.4
    y1 = pcbnew.ToMM(bb.GetBottom()) + 0.4
    cells = set()
    layers = ([0] if pad.GetAttribute() == pcbnew.PAD_ATTRIB_SMD
              and pad.IsOnLayer(pcbnew.F_Cu) else [0, 1])
    ix0, ix1 = int(x0 / GRID), int(math.ceil(x1 / GRID))
    iy0, iy1 = int(y0 / GRID), int(math.ceil(y1 / GRID))
    for layer in layers:
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                cells.add((layer, ix, iy))
    return cells


def filter_extra(g, net, cells):
    """Trim an endpoint whitelist: reaching your own pad through its own
    clearance ring is fine, but the inflated pad bbox must not open a
    corridor through a NEIGHBORING pad's copper (same-package sibling
    pads and fine-pitch/THT neighbors were the top DRC short source)."""
    keep = set()
    for (l, ix, iy) in cells:
        if 0 <= ix < g.nx and 0 <= iy < g.ny:
            own = g.owner[l].get(ix * g.ny + iy)
            if own is None or own == net:
                keep.add((l, ix, iy))
    return keep


def mark_swath(g, layer, x0, y0, x1, y1, net, width):
    # 0.3mm tracks land exactly on the 0.5mm grid: neighbours at 0.5 pitch
    # leave a legal 0.2mm gap. Wider tracks must reserve clearance for a
    # worst-case 0.3mm neighbour.  Distances are measured from the TRUE
    # (possibly off-grid) centerline so a straddling stub/tie marks every
    # cell its copper touches, not just the rounded-center cell.
    half = width / 2 + (0.15 if width <= 0.31 else 0.36)
    steps = max(1, int(max(abs(x1 - x0), abs(y1 - y0)) / GRID))
    r = int(math.ceil(half / GRID)) + 1
    for i in range(steps + 1):
        x = x0 + (x1 - x0) * i / steps
        y = y0 + (y1 - y0) * i / steps
        cix, ciy = int(round(x / GRID)), int(round(y / GRID))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                ex = (cix + dx) * GRID - x
                ey = (ciy + dy) * GRID - y
                if ex * ex + ey * ey <= half * half:
                    g.block_cell(layer, cix + dx, ciy + dy, net)


def add_track(board, g, nets, name, layer_i, x0, y0, x1, y1, width):
    lm = {0: pcbnew.F_Cu, 1: pcbnew.B_Cu}
    t = pcbnew.PCB_TRACK(board)
    t.SetStart(V(x0, y0))
    t.SetEnd(V(x1, y1))
    t.SetWidth(MM(width))
    t.SetLayer(lm[layer_i])
    t.SetNet(nets[name])
    board.Add(t)
    mark_swath(g, layer_i, x0, y0, x1, y1, name, width)


def add_via(board, g, nets, name, x, y, drill=0.3, size=0.6):
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(V(x, y))
    via.SetDrill(MM(drill))
    via.SetWidth(MM(size))
    via.SetViaType(pcbnew.VIATYPE_THROUGH)
    via.SetNet(nets[name])
    board.Add(via)
    half = size / 2 + 0.36
    r = int(math.ceil(half / GRID))
    for layer in (0, 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if (dx * GRID) ** 2 + (dy * GRID) ** 2 <= half * half:
                    g.block(layer,
                            x + dx * GRID, y + dy * GRID, name)


def fanout_escapes(board, g, nets):
    """Give every fine-pitch SMD pad a stub to an escape point clear of
    its neighbors (outward from the footprint center, axis-snapped)."""
    escapes = {}  # (ref, padnum) -> (x, y)
    # THT connector walls: give each signal pin a sideways stub so routes
    # start clear of the pin column (J9/J10 east, J11 col1 west/col2 east)
    # direction per connector; None = split by pad column. Stub lengths are
    # staggered so a single hugging lane cannot seal every stub tip.
    THT_STUB = {"J9": (1, 0), "J10": (1, 0), "J11": None,
                "J6": (0, -1), "J7": (0, -1), "J8": (-1, 0)}
    for fp in board.Footprints():
        ref = fp.GetReference()
        if ref not in THT_STUB:
            continue
        fx = pcbnew.ToMM(fp.GetPosition().x)
        idx = 0
        for pad in fp.Pads():
            net = pad.GetNetname()
            if not net or net.startswith("unconnected") or net == "GND":
                continue
            pos = pad.GetPosition()
            px, py = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
            d = THT_STUB[ref]
            if d is None:
                d = (-1, 0) if px <= fx + 1.0 else (1, 0)
            dist = 2.0 + (idx % 6) * 0.6
            idx += 1
            fy = pcbnew.ToMM(fp.GetPosition().y)
            segs = None
            if ref in ("J6", "J7") and py > fy + 1.0:
                # outer (edge-side) IDC row: thread diagonally between the
                # inner-row pads, then run north beside the inner stubs
                mx = px + 1.27
                my = py - 2.54
                segs = [((px, py), (mx, my)),
                        ((mx, my), (mx, my - dist))]
                tip = (mx, my - dist)
            else:
                tip = (px + d[0] * dist, py + d[1] * dist)
                segs = [((px, py), tip)]

            def path_ok(a, b):
                steps = max(1, int(max(abs(b[0] - a[0]),
                                       abs(b[1] - a[1])) / GRID))
                for k in range(steps + 1):
                    x = a[0] + (b[0] - a[0]) * k / steps
                    y = a[1] + (b[1] - a[1]) * k / steps
                    ix = int(round(x / GRID))
                    iy = int(round(y / GRID))
                    if abs(x - px) < 1.0 and abs(y - py) < 1.0:
                        # own pad vicinity: pad geometry is legal by
                        # construction, but a neighbour's already-emitted
                        # STUB here is real copper -- never cross it
                        own2 = g.owner[0].get(ix * g.ny + iy)
                        if own2 in (None, net, CONTESTED):
                            continue
                        return False
                    if not g.is_free(0, ix, iy, net):
                        return False
                return True

            # the diagonal first segment of a two-segment IDC escape is
            # legal by construction (0.42mm pad clearance) even though the
            # 0.5mm grid cannot represent it -- check only later segments
            if len(segs) > 1:
                # skip the stretch still inside the inner pad's inflated
                # bbox; the geometry there is legal by construction
                check = [((segs[1][0][0], segs[1][0][1] - 1.8), tip)]
            else:
                check = segs
            if all(path_ok(a, b) for a, b in check):
                for a, b in segs:
                    add_track(board, g, nets, net, 0, a[0], a[1],
                              b[0], b[1], 0.3)
                escapes[(ref, pad.GetNumber())] = tip
    for fp in board.Footprints():
        c = fp.GetPosition()
        cx, cy = pcbnew.ToMM(c.x), pcbnew.ToMM(c.y)
        for pad in fp.Pads():
            if pad.GetAttribute() != pcbnew.PAD_ATTRIB_SMD:
                continue
            net = pad.GetNetname()
            if not net or net == "GND" or net in PULSE_NETS:
                continue
            sz = pad.GetSize()
            small = min(pcbnew.ToMM(sz.x), pcbnew.ToMM(sz.y)) < 0.9
            if not small:
                continue
            pos = pad.GetPosition()
            px, py = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
            dx, dy = px - cx, py - cy
            xdir = ((1 if dx >= 0 else -1), 0)
            ydir = (0, (1 if dy >= 0 else -1))
            # dominant axis first, other axis as fallback: a corner pin's
            # dominant axis can run ALONG the pin column, and an unchecked
            # stub there plows through the neighbouring pads
            dirs = (xdir, ydir) if abs(dx) >= abs(dy) else (ydir, xdir)

            def stub_ok(ex, ey):
                # whole stub body must be clear, not just the tip; near
                # the own pad allow only own/contested/unowned cells -- a
                # neighbour's already-emitted stub is real copper
                steps = max(1, int(max(abs(ex - px), abs(ey - py)) / GRID))
                for k in range(steps + 1):
                    x = px + (ex - px) * k / steps
                    y = py + (ey - py) * k / steps
                    ix = int(round(x / GRID))
                    iy = int(round(y / GRID))
                    if abs(x - px) <= 0.6 and abs(y - py) <= 0.6:
                        own2 = g.owner[0].get(ix * g.ny + iy)
                        if own2 in (None, net, CONTESTED):
                            continue
                        return False
                    if not g.is_free(0, ix, iy, net):
                        return False
                return True

            base = (1.2, 1.6, 2.0, 2.5, 3.0)
            smd_idx = sum(ord(ch) for ch in net) % 3
            done = False
            for ux, uy in dirs:
                for dist in base[smd_idx:] + base[:smd_idx]:
                    ex, ey = px + ux * dist, py + uy * dist
                    eix, eiy = int(round(ex / GRID)), int(round(ey / GRID))
                    if g.is_free(0, eix, eiy, net) and stub_ok(ex, ey):
                        add_track(board, g, nets, net, 0, px, py, ex, ey,
                                  0.25)
                        escapes[(fp.GetReference(), pad.GetNumber())] = (ex,
                                                                         ey)
                        done = True
                        break
                if done:
                    break
    return escapes


_HARD_DISC = {}


def hard_disc(width):
    """Off-center swath offsets a track of `width` needs free around its
    centerline (same footprint mark_swath will block on emit).  Empty for
    0.3mm tracks, which fit the 0.5mm grid exactly."""
    offs = _HARD_DISC.get(width)
    if offs is None:
        half = width / 2 + (0.15 if width <= 0.31 else 0.36)
        r = int(math.ceil(half / GRID))
        offs = [(dx, dy) for dx in range(-r, r + 1)
                for dy in range(-r, r + 1)
                if (dx * GRID) ** 2 + (dy * GRID) ** 2 <= half * half
                and (dx, dy) != (0, 0)]
        _HARD_DISC[width] = offs
    return offs


def seg_free(g, layer, x0, y0, x1, y1, net, width, extra=frozenset()):
    """True if a straight track's whole swath is free (or own-net).
    Distances are measured from the TRUE (possibly off-grid) centerline
    so a straddling stub/tie is checked against every cell its copper
    touches, not just the rounded-center cell."""
    half = width / 2 + (0.15 if width <= 0.31 else 0.36)
    r = int(math.ceil(half / GRID)) + 1
    steps = max(1, int(max(abs(x1 - x0), abs(y1 - y0)) / GRID))
    for k in range(steps + 1):
        x = x0 + (x1 - x0) * k / steps
        y = y0 + (y1 - y0) * k / steps
        ix, iy = int(round(x / GRID)), int(round(y / GRID))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                ex = (ix + dx) * GRID - x
                ey = (iy + dy) * GRID - y
                if ex * ex + ey * ey <= half * half and \
                        not g.is_free(layer, ix + dx, iy + dy, net, extra):
                    return False
    return True


def astar(g, net, start, goal, extra, width=0.3):
    import heapq
    hoffs = hard_disc(width)
    sx, sy = int(round(start[0] / GRID)), int(round(start[1] / GRID))
    gx, gy = int(round(goal[0] / GRID)), int(round(goal[1] / GRID))
    startn = (0, sx, sy)
    goals = {(0, gx, gy), (1, gx, gy)}
    openq = [(0, startn)]
    came = {startn: None}
    cost = {startn: 0}
    VIA_COST = 8
    pops = 0
    while openq and pops < 1500000:
        _, node = heapq.heappop(openq)
        pops += 1
        if node in goals:
            path = []
            while node:
                path.append(node)
                node = came[node]
            return path[::-1]
        layer, ix, iy = node
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nix, niy = ix + dx, iy + dy
            if not g.is_free(layer, nix, niy, net, extra):
                continue
            # wide tracks: the whole swath must be free, not just the
            # centerline (a 2mm track physically overlaps copper up to
            # ~1.4mm from its centerline)
            if hoffs and not all(g.is_free(layer, nix + hx, niy + hy,
                                           net, extra)
                                 for hx, hy in hoffs):
                continue
            nn = (layer, nix, niy)
            nc = cost[node] + 1
            if nc < cost.get(nn, 1 << 30):
                cost[nn] = nc
                came[nn] = node
                heapq.heappush(openq, (nc + abs(nix - gx)
                                       + abs(niy - gy), nn))
        ol = 1 - layer
        if g.is_free(ol, ix, iy, net, extra) and via_ok(ix, iy, net) \
                and all(g.is_free(ll, ix + vdx, iy + vdy, net, extra)
                        for ll in (layer, ol)
                        for vdx, vdy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            if not hoffs or all(g.is_free(ol, ix + hx, iy + hy, net, extra)
                                for hx, hy in hoffs):
                nn = (ol, ix, iy)
                nc = cost[node] + VIA_COST
                if nc < cost.get(nn, 1 << 30):
                    cost[nn] = nc
                    came[nn] = node
                    heapq.heappush(openq, (nc + abs(ix - gx)
                                           + abs(iy - gy), nn))
    return None


def emit_tie(board, g, nets, name, exact, node, width):
    ex, ey = exact
    cx, cy = node[1] * GRID, node[2] * GRID
    if abs(ex - cx) < 0.01 and abs(ey - cy) < 0.01:
        return
    lm = {0: pcbnew.F_Cu, 1: pcbnew.B_Cu}
    import math as _m
    if _m.hypot(ex - cx, ey - cy) > 1.6:
        return False  # blind tie too long - crossing risk; caller treats edge as failed
    # the tie is off-grid and was never seen by the router: refuse it if
    # its swath crosses foreign copper (caller treats the edge as failed)
    if not seg_free(g, node[0], ex, ey, cx, cy, name, min(width, 0.3)):
        return False
    t = pcbnew.PCB_TRACK(board)
    t.SetStart(V(ex, ey))
    t.SetEnd(V(cx, cy))
    t.SetWidth(MM(min(width, 0.3)))
    t.SetLayer(lm[node[0]])
    t.SetNet(nets[name])
    board.Add(t)
    # mark it: ties were invisible to everything emitted later, which
    # produced tie-vs-track overlaps
    mark_swath(g, node[0], ex, ey, cx, cy, name, min(width, 0.3))
    return True


def emit_path(board, g, nets, name, path, width):
    k = 0
    while k < len(path) - 1:
        l0, x0, y0 = path[k]
        m = k + 1
        if path[m][0] != l0:
            add_via(board, g, nets, name, path[m][1] * GRID, path[m][2] * GRID)
            k = m
            continue
        dx = path[m][1] - x0
        dy = path[m][2] - y0
        while (m + 1 < len(path) and path[m + 1][0] == l0
               and path[m + 1][1] - path[m][1] == dx
               and path[m + 1][2] - path[m][2] == dy):
            m += 1
        add_track(board, g, nets, name, l0, x0 * GRID, y0 * GRID,
                  path[m][1] * GRID, path[m][2] * GRID, width)
        k = m


def route_stragglers(board, g, nets, misses, zone_by_net):
    """Fat-track pulse pads (outside zones) to the nearest in-zone point."""
    failed = []
    for ref, padnum, net, px, py in misses:
        width = NET_WIDTHS.get(net, 2.0)
        # nearest zone interior point on a 2mm lattice
        best = None
        for pts in zone_by_net[net]:
            xs = [q[0] for q in pts]
            ys = [q[1] for q in pts]
            y = min(ys) + 2.0
            while y < max(ys) - 1.0:
                x = min(xs) + 2.0
                while x < max(xs) - 1.0:
                    if point_in_poly(x, y, pts):
                        d = abs(x - px) + abs(y - py)
                        if best is None or d < best[0]:
                            best = (d, x, y)
                    x += 2.0
                y += 2.0
        if best is None:
            failed.append((net, ref, padnum))
            continue
        fp = board.FindFootprintByReference(ref)
        extra = set()
        for pad in fp.Pads():
            if pad.GetNumber() == padnum:
                extra |= pad_cells(pad)
        fextra = frozenset(filter_extra(g, net, extra))
        # width ladder: these are sense taps / zone feeds -- a narrower
        # connection beats a missing one when no fat corridor exists
        path = None
        for w_try in (width, 1.0, 0.4):
            if w_try > width:
                continue
            path = astar(g, net, (px, py), (best[1], best[2]), fextra,
                         width=w_try)
            if path is not None:
                if w_try < width:
                    print(f"   straggler {net} {ref}.{padnum}: "
                          f"narrowed {width} -> {w_try}")
                width = w_try
                break
        if path is None:
            failed.append((net, ref, padnum))
            continue
        emit_path(board, g, nets, net, path, width)
        emit_tie(board, g, nets, net, (px, py), path[0], width)
    return failed


def _seg_pt_dist2(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 <= 1e-12:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    cx, cy = ax + t * vx, ay + t * vy
    return (px - cx) ** 2 + (py - cy) ** 2


def via_clear_of_foreign(board, px, py, net, radius):
    """True iff no foreign-net track/via/THT-pad copper lies within `radius`
    mm of (px,py). Real geometry (not the coarse routing grid), so it is safe
    for via-in-pad on fine-pitch parts where the grid aliases."""
    r2 = radius * radius
    for t in board.GetTracks():
        if t.GetNetname() == net:
            continue
        if t.GetClass() == "PCB_VIA":
            vp = t.GetPosition()
            try:
                hw = pcbnew.ToMM(t.GetWidth(pcbnew.F_Cu)) / 2
            except Exception:
                hw = 0.3
            if (pcbnew.ToMM(vp.x) - px) ** 2 + (pcbnew.ToMM(vp.y) - py) ** 2 \
                    < (radius + hw) ** 2:
                return False
        else:
            a, b = t.GetStart(), t.GetEnd()
            hw = pcbnew.ToMM(t.GetWidth()) / 2
            if _seg_pt_dist2(px, py, pcbnew.ToMM(a.x), pcbnew.ToMM(a.y),
                             pcbnew.ToMM(b.x), pcbnew.ToMM(b.y)) \
                    < (radius + hw) ** 2:
                return False
    for fp in board.Footprints():
        for pad in fp.Pads():
            if pad.GetNetname() == net:
                continue
            pp = pad.GetPosition()
            sz = pad.GetSize()
            reach = radius + max(pcbnew.ToMM(sz.x), pcbnew.ToMM(sz.y)) / 2
            if (pcbnew.ToMM(pp.x) - px) ** 2 + (pcbnew.ToMM(pp.y) - py) ** 2 \
                    < reach * reach:
                # cheap bbox refine
                bb = pad.GetBoundingBox()
                if (pcbnew.ToMM(bb.GetLeft()) - radius <= px
                        <= pcbnew.ToMM(bb.GetRight()) + radius
                        and pcbnew.ToMM(bb.GetTop()) - radius <= py
                        <= pcbnew.ToMM(bb.GetBottom()) + radius):
                    return False
    return True


def gnd_via_drops(board, g, nets):
    gnd = nets["GND"]
    added = 0
    manual = {(r, pn) for r, pn, _, _, _ in MANUAL_TRACKS}
    for fp in board.Footprints():
        for pad in fp.Pads():
            if pad.GetNetname() != "GND":
                continue
            if pad.GetAttribute() != pcbnew.PAD_ATTRIB_SMD:
                continue
            if (fp.GetReference(), pad.GetNumber()) in manual:
                continue
            pos = pad.GetPosition()
            px, py = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
            # via-in-pad first: a small GND via (0.45/0.25) at the pad centre
            # is inside GND copper (cannot short, same net) and always lands in
            # the full In1/In2 GND plane. The grid free-check is skipped here
            # on purpose -- at the 0.5mm grid a fine-pitch pad aliases onto a
            # neighbour's cell and falsely reads blocked. We instead trust the
            # pad and only reject a real hole-to-hole conflict with an existing
            # THT pad / via within 0.5mm. This closes the congested IC/cap GND
            # pads (U9/U11/C5/C11...) the >=1.4mm offsets can't reach.
            done = False
            vix = int(round(px / GRID))
            viy = int(round(py / GRID))
            # via 0.45mm + 0.2mm clearance -> foreign copper must clear 0.425mm;
            # and my 0.25 drill needs >=0.25mm edge clearance to any other hole
            # (0.125 + other_r + 0.25 centre distance).
            hole_ok = True
            for t in board.GetTracks():
                if t.GetClass() != "PCB_VIA":
                    continue
                vp = t.GetPosition()
                if (pcbnew.ToMM(vp.x) - px) ** 2 + (pcbnew.ToMM(vp.y) - py) ** 2 \
                        < (0.125 + 0.15 + 0.25) ** 2:
                    hole_ok = False
                    break
            if hole_ok:
                for fp2 in board.Footprints():
                    for pad2 in fp2.Pads():
                        if pad2.GetAttribute() == pcbnew.PAD_ATTRIB_SMD:
                            continue
                        dr = pad2.GetDrillSize()
                        if dr.x <= 0:
                            continue
                        pp = pad2.GetPosition()
                        need = 0.125 + pcbnew.ToMM(dr.x) / 2 + 0.25
                        if (pcbnew.ToMM(pp.x) - px) ** 2 \
                                + (pcbnew.ToMM(pp.y) - py) ** 2 < need * need:
                            hole_ok = False
                            break
                    if not hole_ok:
                        break
            # size the via to fit INSIDE the pad so a fine-pitch pad (U10 GND
            # 0.35mm) gets a small via that does not overrun onto -- or box in
            # the escape of -- its neighbour pads. 0.3mm floor (JLC min).
            psz = pad.GetSize()
            pad_min = min(pcbnew.ToMM(psz.x), pcbnew.ToMM(psz.y))
            via_sz = min(0.45, max(0.3, pad_min - 0.02))
            via_dr = max(0.15, round(via_sz - 0.2, 2))
            clr = via_sz / 2 + 0.2
            if hole_ok and via_ok(vix, viy, "GND") \
                    and via_clear_of_foreign(board, px, py, "GND", clr):
                add_via(board, g, nets, "GND", px, py,
                        drill=via_dr, size=via_sz)
                added += 1
                done = True
            if done:
                continue
            for ddx, ddy in [(0, 1.4), (0, -1.4), (1.4, 0), (-1.4, 0),
                             (1.2, 1.2), (-1.2, 1.2), (1.2, -1.2),
                             (-1.2, -1.2), (0, 2.2), (0, -2.2),
                             (2.2, 0), (-2.2, 0), (0, 3.0), (0, -3.0),
                             (3.0, 0), (-3.0, 0), (2.2, 2.2),
                             (-2.2, 2.2), (2.2, -2.2), (-2.2, -2.2),
                             # far ring: safe now that the stub swath is
                             # checked against foreign copper (seg_free)
                             (0, 3.8), (0, -3.8), (3.8, 0), (-3.8, 0),
                             (3.0, 3.0), (-3.0, 3.0), (3.0, -3.0),
                             (-3.0, -3.0), (0, 4.6), (0, -4.6),
                             (4.6, 0), (-4.6, 0)]:
                vix = int(round((px + ddx) / GRID))
                viy = int(round((py + ddy) / GRID))
                vx, vy = vix * GRID, viy * GRID   # snapped exactly on-grid
                # center + orthogonal cells (<=0.5mm) can short a 0.6mm
                # via; diagonal cells (0.707mm) geometrically cannot --
                # requiring the full 3x3 starves dense areas of drops
                free = all(g.is_free(l, vix + dx, viy + dy, "GND")
                           for l in (0, 1)
                           for dx, dy in ((0, 0), (1, 0), (-1, 0),
                                          (0, 1), (0, -1)))
                if not (free and via_ok(vix, viy, "GND")):
                    continue
                # the pad-to-via stub must not cross foreign copper either
                # (0.3mm stub is grid-legal: centerline check suffices)
                if not seg_free(g, 0, px, py, vx, vy, "GND", 0.3):
                    continue
                add_via(board, g, nets, "GND", vx, vy)
                add_track(board, g, nets, "GND", 0, px, py, vx, vy, 0.3)
                added += 1
                done = True
                break
            if not done:
                print(f"   GND drop FAILED: {fp.GetReference()}"
                      f".{pad.GetNumber()} at ({px:.1f},{py:.1f})")
    return added


def purge_track_shorts(board):
    """Geometric safety net: delete the shorter of any two touching
    different-net tracks on the same outer layer.  Every upstream check
    works on the 0.5mm grid; off-grid ties/stubs can still produce
    boundary touches at the 0.05mm scale.  A deleted segment shows up as
    honest unconnected copper instead of a short."""
    from collections import defaultdict

    def seg_pt_d2(px, py, ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy
        if L2 <= 1e-12:
            dx, dy = px - ax, py - ay
            return dx * dx + dy * dy
        t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
        dx, dy = px - (ax + t * vx), py - (ay + t * vy)
        return dx * dx + dy * dy

    def ccw(x1, y1, x2, y2, x3, y3):
        return (y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1)

    def seg_seg_dist(a, b):
        ax, ay, bx, by = a
        cx, cy, dx, dy = b
        d1 = ccw(ax, ay, bx, by, cx, cy)
        d2 = ccw(ax, ay, bx, by, dx, dy)
        d3 = ccw(cx, cy, dx, dy, ax, ay)
        d4 = ccw(cx, cy, dx, dy, bx, by)
        if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
            return 0.0
        return math.sqrt(min(
            seg_pt_d2(cx, cy, ax, ay, bx, by),
            seg_pt_d2(dx, dy, ax, ay, bx, by),
            seg_pt_d2(ax, ay, cx, cy, dx, dy),
            seg_pt_d2(bx, by, cx, cy, dx, dy)))

    # net-tie fences: tracks of two different nets legally meet there
    nt_zones = []
    for fp in board.Footprints():
        if fp.GetReference().startswith("NT"):
            c = fp.GetPosition()
            nt_zones.append((pcbnew.ToMM(c.x), pcbnew.ToMM(c.y)))

    def near_nt(x, y):
        return any(abs(x - zx) < 2.5 and abs(y - zy) < 2.5
                   for zx, zy in nt_zones)

    segs = []
    for t in board.GetTracks():
        if t.GetClass() != "PCB_TRACK":
            continue
        li = {pcbnew.F_Cu: 0, pcbnew.B_Cu: 1}.get(t.GetLayer())
        if li is None:
            continue
        a, b = t.GetStart(), t.GetEnd()
        segs.append([t, li, pcbnew.ToMM(a.x), pcbnew.ToMM(a.y),
                     pcbnew.ToMM(b.x), pcbnew.ToMM(b.y),
                     pcbnew.ToMM(t.GetWidth()), t.GetNetname()])

    bins = defaultdict(list)
    B = 4.0
    for i, s in enumerate(segs):
        for bx in range(int(min(s[2], s[4]) // B),
                        int(max(s[2], s[4]) // B) + 1):
            for by in range(int(min(s[3], s[5]) // B),
                            int(max(s[3], s[5]) // B) + 1):
                bins[(s[1], bx, by)].append(i)

    doomed = set()
    seen = set()
    for idxs in bins.values():
        for ii in range(len(idxs)):
            for jj in range(ii + 1, len(idxs)):
                i1, i2 = idxs[ii], idxs[jj]
                if i1 in doomed or i2 in doomed:
                    continue
                pair = (i1, i2)
                if pair in seen:
                    continue
                seen.add(pair)
                s1, s2 = segs[i1], segs[i2]
                if s1[7] == s2[7] or not s1[7] or not s2[7]:
                    continue
                lim = s1[6] / 2 + s2[6] / 2 + 1e-3
                d = seg_seg_dist((s1[2], s1[3], s1[4], s1[5]),
                                 (s2[2], s2[3], s2[4], s2[5]))
                if d >= lim:
                    continue
                if near_nt(s1[2], s1[3]) or near_nt(s1[4], s1[5]):
                    continue
                l1 = math.hypot(s1[4] - s1[2], s1[5] - s1[3])
                l2 = math.hypot(s2[4] - s2[2], s2[5] - s2[3])
                victim, other = (i1, s2) if l1 <= l2 else (i2, s1)
                doomed.add(victim)
                v = segs[victim]
                print(f"   purge short: {v[7]} seg "
                      f"({v[2]:.2f},{v[3]:.2f})-({v[4]:.2f},{v[5]:.2f}) "
                      f"vs {other[7]}")
    for i in doomed:
        board.Delete(segs[i][0])
    return len(doomed)


def route_signals(board, g, nets, escapes):
    """Negotiated congestion routing: signal tracks are soft obstacles that
    other nets may cross at a per-iteration penalty; overused cells build
    history cost until every edge has a private path (PathFinder-lite).
    Hard obstacles (pads, pulse zones, stubs, GND vias) stay in `g`.

    The negotiation loop is parallelized (parallel PathFinder): every
    iteration's batch of edges is routed by a process pool against a
    snapshot of the usage/history penalty state (stale penalties within an
    iteration are acceptable), then results are applied sequentially so
    conflict detection stays identical to the serial code."""
    import os
    import time
    from multiprocessing import Pool, shared_memory

    import numpy as np

    import route_worker

    netpads = {}
    padobjs = {}
    for fp in board.Footprints():
        for pad in fp.Pads():
            n = pad.GetNetname()
            if not n:
                continue
            key = (fp.GetReference(), pad.GetNumber())
            pos = escapes.get(key)
            if pos is None:
                pp = pad.GetPosition()
                pos = (pcbnew.ToMM(pp.x), pcbnew.ToMM(pp.y))
            netpads.setdefault(n, []).append((key, pos))
            padobjs[key] = pad

    todo = [(n, pads) for n, pads in netpads.items()
            if n not in PULSE_NETS and n != "GND" and len(pads) >= 2]
    todo.sort(key=lambda item: len(item[1]))

    edges_all = []  # (eid, name, keyA, keyB, ptA, ptB, width, extra)
    for name, pads in todo:
        width = NET_WIDTHS.get(name, 0.3)
        pts = [p for _, p in pads]
        keys = [k for k, _ in pads]
        in_tree = [0]
        mst = []
        while len(in_tree) < len(pts):
            best = None
            for i in in_tree:
                for j in range(len(pts)):
                    if j in in_tree:
                        continue
                    d = (abs(pts[i][0] - pts[j][0])
                         + abs(pts[i][1] - pts[j][1]))
                    if best is None or d < best[0]:
                        best = (d, i, j)
            mst.append((best[1], best[2]))
            in_tree.append(best[2])
        for i, j in mst:
            extra = filter_extra(g, name, pad_cells(padobjs[keys[i]])
                                 | pad_cells(padobjs[keys[j]]))
            edges_all.append([len(edges_all), name, keys[i], keys[j],
                              pts[i], pts[j], width, frozenset(extra)])

    usage = [{}, {}]   # cellidx -> {netname: refcount}
    epaths = {}
    eswaths = {}

    def swath_of(path, width):
        cells = set()
        half = width / 2 + (0.15 if width <= 0.31 else 0.36)
        r = int(math.ceil(half / GRID))
        prev_l = None
        for (l, ix, iy) in path:
            if prev_l is not None and prev_l != l:
                for ll in (0, 1):
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            cells.add((ll, ix + dx, iy + dy))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if (dx * GRID) ** 2 + (dy * GRID) ** 2 <= half * half:
                        cells.add((l, ix + dx, iy + dy))
            prev_l = l
        return cells

    # `usage` stays the source of truth for has_conflict/history; the
    # shared-memory `use_arr` mirror encodes each cell for the workers as
    # -1 = free, -2 = multiple nets, else the id of the single net present.
    def occupy(name, cells):
        nid = nid_of[name]
        for (l, ix, iy) in cells:
            i = ix * g.ny + iy
            d = usage[l].setdefault(i, {})
            d[name] = d.get(name, 0) + 1
            if 0 <= i < ncell:
                use_arr[l][i] = nid if len(d) == 1 else -2

    def vacate(name, cells):
        for (l, ix, iy) in cells:
            i = ix * g.ny + iy
            d = usage[l].get(i)
            if d and name in d:
                d[name] -= 1
                if d[name] <= 0:
                    del d[name]
                    if 0 <= i < ncell:
                        if not d:
                            use_arr[l][i] = -1
                        elif len(d) == 1:
                            use_arr[l][i] = nid_of[next(iter(d))]
                        else:
                            use_arr[l][i] = -2

    def has_conflict(eid, name):
        for (l, ix, iy) in eswaths.get(eid, ()):
            d = usage[l].get(ix * g.ny + iy)
            if d and any(k != name for k in d):
                return True
        return False

    # ---- multiprocessing PathFinder setup --------------------------------
    # Static A*-relevant state (grid occupancy, via keepouts, per-edge
    # metadata) ships once per worker via the Pool initializer; the mutable
    # usage/history penalty state lives in shared-memory int32 arrays that
    # only the parent writes, and only between batches.
    names = set(e[1] for e in edges_all)
    for l in (0, 1):
        names.update(g.owner[l].values())
    names.update(VIA_FORBID.values())
    nid_of = {n: i for i, n in enumerate(sorted(names))}

    ncell = g.nx * g.ny
    shms = []

    def _shm_arr(fill):
        s = shared_memory.SharedMemory(create=True, size=ncell * 4)
        shms.append(s)
        a = np.frombuffer(s.buf, dtype=np.int32, count=ncell)
        a[:] = fill
        return s.name, a

    use_arr, hist_arr, use_shm, hist_shm = [], [], [], []
    for l in (0, 1):
        nm_, a_ = _shm_arr(-1)
        use_shm.append(nm_)
        use_arr.append(a_)
    for l in (0, 1):
        nm_, a_ = _shm_arr(0)
        hist_shm.append(nm_)
        hist_arr.append(a_)
    nm_ = a_ = None

    static = {
        "grid": GRID, "nx": g.nx, "ny": g.ny,
        "blocked": tuple(bytes(b) for b in g.blocked),
        "owner": tuple({i: nid_of[v] for i, v in g.owner[l].items()}
                       for l in (0, 1)),
        "via_forbid": {k: nid_of[v] for k, v in VIA_FORBID.items()},
        "edges": {
            e[0]: (nid_of[e[1]],
                   int(round(e[4][0] / GRID)), int(round(e[4][1] / GRID)),
                   int(round(e[5][0] / GRID)), int(round(e[5][1] / GRID)),
                   e[6], e[7])
            for e in edges_all},
        "use_shm": use_shm, "hist_shm": hist_shm,
    }

    nproc = int(os.environ.get("OMNI_NEG_PROCS", "0")) \
        or max(1, min(24, os.cpu_count() or 8))
    max_iters = int(os.environ.get("OMNI_NEG_ITERS", "300"))
    p_cap = int(os.environ.get("OMNI_NEG_PCAP", "0"))  # 0 = uncapped
    print(f"   negotiate: {len(edges_all)} edges, {nproc} workers",
          flush=True)
    pool = Pool(nproc, initializer=route_worker.init_worker,
                initargs=(static,))
    best = None   # (score, it, epaths copy, eswaths copy, over, missing)
    score = None
    try:
        for it in range(max_iters):
            # present-sharing ramp; the best-state snapshot below banks
            # the minimum, so late-run thrash from the growing penalty is
            # harmless -- OMNI_NEG_PCAP>0 caps the ramp if desired
            p_now = 4 + it * 3
            if p_cap:
                p_now = min(p_now, p_cap)
            if it == 0:
                work = list(edges_all)
            else:
                work = [e for e in edges_all
                        if e[0] not in epaths or has_conflict(e[0], e[1])]
                if not work:
                    break
                rng = random.Random(1000 + it)
                rng.shuffle(work)
            # route in sub-batches of ~nproc edges: within a sub-batch,
            # edges route in parallel against a stale usage/hist snapshot
            # (parallel PathFinder semantics -- an edge's own old path
            # never penalizes itself, so no pre-vacate is needed); the
            # applied results become visible through shared memory before
            # the next sub-batch.  Bounding staleness to nproc edges keeps
            # convergence near-serial (a single whole-iteration snapshot
            # livelocks: conflicting pairs keep moving simultaneously).
            t_map = t_apply = t_astar = t_snap = t_slow = 0.0
            for s0 in range(0, len(work), nproc):
                sub = work[s0:s0 + nproc]
                token = it * 100000 + s0
                tasks = [(token, p_now, [e[0]]) for e in sub]
                rmap = {}
                t0 = time.perf_counter()
                slow = 0.0
                for batch in pool.map(route_worker.route_batch, tasks):
                    for eid, path, dsnap, dt in batch:
                        rmap[eid] = path
                        t_astar += dt
                        t_snap += dsnap
                        slow = max(slow, dt)
                t1 = time.perf_counter()
                t_map += t1 - t0
                t_slow += slow
                # apply sequentially, in `work` order, exactly as the
                # serial loop did (vacate old path, occupy new swath)
                for e in sub:
                    eid, name, ka, kb, pa, pb, width, extra = e
                    if eid in epaths:
                        vacate(name, eswaths[eid])
                        del epaths[eid], eswaths[eid]
                    path = rmap[eid]
                    if path is None:
                        continue
                    sw = swath_of(path, width)
                    epaths[eid] = path
                    eswaths[eid] = sw
                    occupy(name, sw)
                t_apply += time.perf_counter() - t1
            t_hist0 = time.perf_counter()
            # build history on overused cells
            over = 0
            for e in edges_all:
                eid, name = e[0], e[1]
                if eid not in epaths:
                    continue
                if has_conflict(eid, name):
                    over += 1
                    for (l, ix, iy) in eswaths[eid]:
                        i = ix * g.ny + iy
                        d = usage[l].get(i)
                        if d and any(k != name for k in d) \
                                and 0 <= i < ncell:
                            hist_arr[l][i] += 1
            missing = [e for e in edges_all if e[0] not in epaths]
            print(f"   negotiate iter {it}: rerouted {len(work)}, "
                  f"conflicted {over}, unrouted {len(missing)}", flush=True)
            if os.environ.get("OMNI_NEG_PROF"):
                t_hist = time.perf_counter() - t_hist0
                nsb = (len(work) + nproc - 1) // nproc
                util = (t_astar / (t_map * nproc) * 100) if t_map else 0.0
                print(f"      prof: map {t_map:.2f}s (astar_sum "
                      f"{t_astar:.2f}s, snap_sum {t_snap:.2f}s, "
                      f"slowest-per-sb {t_slow:.2f}s, {nsb} sub-batches) "
                      f"apply {t_apply:.2f}s hist {t_hist:.2f}s "
                      f"util {util:.0f}%", flush=True)
            # keep the best state seen so far; the answer is monotone
            score = over + len(missing)
            if best is None or score < best[0]:
                best = (score, it, dict(epaths), dict(eswaths), over,
                        len(missing))
            if over == 0 and not missing:
                break
            # stagnation break: past the warm-up, stop burning iterations
            # if no new minimum in 30 iters (best state is banked above)
            if it > 26 and it - best[1] >= 30:
                print(f"   negotiate: stagnated (best iter {best[1]}, "
                      f"score {best[0]}), stopping early", flush=True)
                break
        # restore the best state if the final iteration was worse
        if best is not None and score is not None and score > best[0]:
            _, bit, bep, besw, bover, bmiss = best
            print(f"   negotiate: restoring best state from iter {bit} "
                  f"(conflicted {bover}, unrouted {bmiss})", flush=True)
            epaths.clear()
            epaths.update(bep)
            eswaths.clear()
            eswaths.update(besw)
            ename = {e[0]: e[1] for e in edges_all}
            for l in (0, 1):
                usage[l].clear()
                use_arr[l][:] = -1
            for eid, sw in eswaths.items():
                occupy(ename[eid], sw)
    finally:
        pool.close()
        pool.join()
        use_arr.clear()
        hist_arr.clear()
        for s in shms:
            try:
                s.close()
                s.unlink()
            except Exception:
                pass

    failed = []
    routed = 0
    for e in edges_all:
        eid, name, ka, kb, pa, pb, width, extra = e
        if eid not in epaths or has_conflict(eid, name):
            failed.append((name, ka, kb))
            continue
        path = epaths[eid]
        # re-verify against copper emitted SO FAR in this phase: off-grid
        # ties are invisible to the negotiation's usage map, so a path
        # fixed during negotiation can overlap a tie emitted just before
        # it; bounce such edges to pass2, which sees the live grid.
        # The build-time `extra` whitelist is deliberately NOT honoured
        # for foreign-owned cells: a whitelisted cell that acquired
        # foreign copper during the emit phase must count as dirty.
        hoffs = hard_disc(width)
        dirty = False
        for (pl, pix, piy) in path:
            cells = [(pix, piy)]
            if hoffs:
                cells += [(pix + hx, piy + hy) for hx, hy in hoffs]
            for cx2, cy2 in cells:
                if g.is_free(pl, cx2, cy2, name):
                    continue
                own2 = (g.owner[pl].get(cx2 * g.ny + cy2)
                        if 0 <= cx2 < g.nx and 0 <= cy2 < g.ny else None)
                if own2 in (None, CONTESTED) and (pl, cx2, cy2) in extra:
                    continue  # static blockage the whitelist already vetted
                dirty = True
                break
            if dirty:
                break
        if dirty:
            failed.append((name, ka, kb))
            continue
        ta = emit_tie(board, g, nets, name, pa, path[0], width)
        tb = emit_tie(board, g, nets, name, pb, path[-1], width)
        if ta is False or tb is False:
            failed.append((name, ka, kb))
            continue
        emit_path(board, g, nets, name, path, width)
        routed += 1
    return routed, failed, [e[1] for e in edges_all]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def author_shunt_hi_bus(board, g, nets):
    """Explicit B.Cu bus tying the three pulse-FET sources (Q10/Q11/Q12 pin 3)
    to the shunt RS1, so the high-current SHUNT_HI path does NOT depend on the
    F.Cu pour -- which the autorouter fragments when it runs gate/snubber nets
    through the FET strip, islanding sources. B.Cu keeps F.Cu clear for gate
    routing; the bus is locked (SHUNT_HI in PLANE_NETS) so the router routes
    around it. A via at each source pad and at the shunt ties them to the bus.
    """
    if "SHUNT_HI" not in nets:
        return
    y = 72.0                          # inside the source pads (y 70.3-73.2)
    # two vias per FET source (its source is several pads that a fragmented
    # pour can split) + the shunt RS1 pad, all on the bus line
    via_x = [76.2, 79.8, 94.2, 96.6, 112.2, 115.8, 128.6]  # clear of stitch grid
    add_track(board, g, nets, "SHUNT_HI", 1, min(via_x), y, max(via_x), y, 1.5)
    for x in via_x:
        add_via(board, g, nets, "SHUNT_HI", x, y, drill=0.4, size=0.7)


def author_u10_escape(board, g, nets):
    """U10 (VSSOP-8, 0.35mm pads) fine-pitch power/GND escape.

    Both freerouting and DeepPCB fail to fan U10's four corner power pins
    (1,7 = 3V3; 5,8 = GND) out past its SPI signal pins -- the 3V3/GND copper
    is 3-6mm away with no channel through the fine pitch. We author it, like the
    Kelvin ties: the GND pins drop a pad-fitted via straight to the inner GND
    plane; the two 3V3 pins drop pad-fitted vias to B.Cu and link there to the
    U10 bypass cap C28. DeepPCB is then left only the four easy F.Cu signal pins
    (2,3,4,6). Locked in the DSN via 3V3 in CRITICAL_NETS + GND in PLANE_NETS.
    """
    u10 = board.FindFootprintByReference("U10")
    if u10 is None:
        return
    pad = {p.GetNumber(): p for p in u10.Pads()}

    def xy(p):
        pos = p.GetPosition()
        return pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)

    # Dogbone fan-out with 0.2mm stubs (0.3mm > 0.5mm pitch would grip the
    # neighbour pads). Each pin exits straight out past the pad row first, then
    # the two outer right pins (8,5) stagger up/down to vias at distinct radii
    # so no via sits at pad pitch. GND vias tie straight to the inner GND plane;
    # the 3V3 vias are open-space anchors the router finishes to the net (C28 is
    # just right of pin7). Paths are (dx,dy)-from-pad waypoints, last = via.
    paths = [
        ("1", "3V3", [(-1.9, 0.0)]),               # left, clear
        ("7", "3V3", [(1.25, 0.0)]),               # straight right
        ("8", "GND", [(1.05, 0.0), (1.65, -0.9)]),  # out, then up
        ("5", "GND", [(1.05, 0.0), (1.65, 0.9)]),   # out, then down
    ]
    for num, net, waypts in paths:
        px, py = xy(pad[num])
        cx, cy = px, py
        for dx, dy in waypts:
            nx, ny = px + dx, py + dy
            add_track(board, g, nets, net, 0, cx, cy, nx, ny, 0.2)
            cx, cy = nx, ny
        # drill 0.25 on the 0.45 via -> 0.10mm annular (min); 0.3 drill was 0.075
        add_via(board, g, nets, net, cx, cy, drill=0.25, size=0.45)


def build_preroute(gnd_drops=True):
    """Board with placement, pulse copper, Kelvin, GND drops - no signals.

    gnd_drops=False (DeepPCB flow): skip the GND via-in-pad pass. Those
    0.45mm vias land in 0.35mm fine-pitch IC GND pads (U10 etc.), which
    physically blocks the router; DeepPCB connects GND to the inner planes
    itself, so the drops are unneeded and harmful there.
    """
    comps, netlist_nets, padnet, meta = load_netlist()
    board = new_board()
    add_outline(board)
    nets, missing = place_all(board, comps, padnet, meta)
    if missing:
        print(f"UNPLACED refs: {missing}")

    # schematic-parity: DNP on the unpopulated bank positions, BOM-exclude on
    # the test point + Kelvin net-ties (match the schematic; else kicad-cli
    # --schematic-parity flags 6 attribute mismatches on every regen).
    dnp_refs = {"C92", "C93", "C94"}
    exbom_refs = {"TP1", "NT1", "NT2"}
    for fp in board.Footprints():
        ref = fp.GetReference()
        if ref in dnp_refs:
            fp.SetDNP(True)
        if ref in exbom_refs:
            fp.SetExcludedFromBOM(True)

    boxes = []
    for fp in board.Footprints():
        bb = fp.GetBoundingBox(False)
        boxes.append((fp.GetReference(),
                      pcbnew.ToMM(bb.GetLeft()), pcbnew.ToMM(bb.GetTop()),
                      pcbnew.ToMM(bb.GetRight()), pcbnew.ToMM(bb.GetBottom())))
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]:
                overlaps.append((a[0], b[0]))
    if overlaps:
        print(f"FOOTPRINT OVERLAPS ({len(overlaps)}): {overlaps[:25]}")

    zones = build_zones(board, nets)
    add_bridges(board, nets, padnet)

    zone_by_net = {}
    for name, pts, _ in zones:
        zone_by_net.setdefault(name, []).append(pts)

    misses = check_pulse_pads(board, zones)
    print(f"pulse pads outside zones: {len(misses)} (straggler-routed)")

    g = build_grid(board)
    build_via_forbid()

    for ra, pa, rb, pb, w in KELVIN_TRACKS:
        fa = board.FindFootprintByReference(ra)
        fb = board.FindFootprintByReference(rb)
        pada = padb = None
        for pad in fa.Pads():
            if pad.GetNumber() == pa:
                pada = pad
        for pad in fb.Pads():
            if pad.GetNumber() == pb:
                padb = pad
        a, b = pada.GetPosition(), padb.GetPosition()
        net = pada.GetNetname()
        add_track(board, g, nets, net, 0, pcbnew.ToMM(a.x), pcbnew.ToMM(a.y),
                  pcbnew.ToMM(b.x), pcbnew.ToMM(b.y), w)

    for ref, padnum, (tx, ty), w, layer_i in MANUAL_TRACKS:
        fp = board.FindFootprintByReference(ref)
        for pad in fp.Pads():
            if pad.GetNumber() == padnum:
                pp = pad.GetPosition()
                add_track(board, g, nets, pad.GetNetname(), layer_i,
                          pcbnew.ToMM(pp.x), pcbnew.ToMM(pp.y), tx, ty, w)

    for net, cx, cy, nx, ny, pitch, drill, size in VIA_CLUSTERS:
        for ix in range(nx):
            for iy in range(ny):
                add_via(board, g, nets, net,
                        cx + (ix - (nx - 1) / 2) * pitch,
                        cy + (iy - (ny - 1) / 2) * pitch, drill, size)

    author_u10_escape(board, g, nets)
    author_shunt_hi_bus(board, g, nets)

    # NOTE: route_critical() (deterministic A* authoring of the critical nets)
    # is intentionally NOT called here. Emitting astar-generated critical
    # tracks as DSN `(type fix)` wires crashes freerouting v2.2.4 in
    # insert_forced_trace_polyline ('from_corner is null' on the short diagonal
    # pad ties). Critical nets are instead routed by freerouting and reviewed
    # via highlighted screenshots; any net whose routing is inadequate gets a
    # clean explicit trace in placement.MANUAL_TRACKS (which freerouting
    # digests fine), not a blanket astar pass.

    # stitch AFTER kelvin/manual/cluster copper is marked: the lattice
    # test (is_free) then skips sites colliding with it, instead of the
    # blind-placed clusters landing on an already-stitched lattice
    for name, pts, z in zones:
        if z.GetLayer() == pcbnew.F_Cu:
            stitch_zone(board, g, nets, name, pts)

    strag_failed = route_stragglers(board, g, nets, misses, zone_by_net)
    print(f"straggler failures: {strag_failed}")

    if gnd_drops:
        nvias = gnd_via_drops(board, g, nets)
        print(f"GND via drops: {nvias}")

    add_text(board, "DANGER - STORED ENERGY", 100, 6, 2.5)
    add_text(board, "CAPACITORS MAY BE CHARGED - CHECK LIVE LED", 100, 10, 1.5)
    add_text(board, "COIL: L>=1uH R_total>=90mOhm", 30, 37, 1.2)

    return board, g, nets, misses, strag_failed


def route_critical(board, g, nets):
    """Deterministically A*-route the critical signal nets BEFORE DSN export so
    dsn_fixup locks them (type fix) and freerouting only fills the noncritical
    interconnect around them. Reviewer requirement: pulse-adjacent gate /
    boost-sense / Kelvin / safety-interlock nets are authored + reviewed, not
    autorouted. Pours/planes and VBOOST (freerouting handles it well at 1.5mm)
    are excluded. Nets are ordered so the tightest loops route first into open
    copper; each net's pads are joined by a nearest-neighbour MST."""
    route_nets = sorted(CRITICAL_NETS - PLANE_NETS - {"VBOOST"})
    by_net = {}
    for fp in board.Footprints():
        for pad in fp.Pads():
            n = pad.GetNetname()
            if n in route_nets:
                by_net.setdefault(n, []).append(pad)

    # Build the full edge list (nearest-neighbour MST per net) up front, then
    # route ALL edges shortest-first so tight local loops lock into open copper
    # before the long cross-board runs consume it.
    edges = []
    for net in route_nets:
        pads = by_net.get(net, [])
        if len(pads) < 2:
            continue
        pts = [(pcbnew.ToMM(p.GetPosition().x), pcbnew.ToMM(p.GetPosition().y),
                p) for p in pads]
        connected = [0]
        remaining = list(range(1, len(pts)))
        while remaining:
            best = None
            for ci in connected:
                for ri in remaining:
                    d = (pts[ci][0] - pts[ri][0]) ** 2 \
                        + (pts[ci][1] - pts[ri][1]) ** 2
                    if best is None or d < best[0]:
                        best = (d, ci, ri)
            d, ci, ri = best
            edges.append((d, net, pts[ci][2], pts[ri][2]))
            connected.append(ri)
            remaining.remove(ri)
    edges.sort(key=lambda e: e[0])

    def try_edge(net, pa, pb):
        w = NET_WIDTHS.get(net, 0.3)
        sa = (pcbnew.ToMM(pa.GetPosition().x), pcbnew.ToMM(pa.GetPosition().y))
        sb = (pcbnew.ToMM(pb.GetPosition().x), pcbnew.ToMM(pb.GetPosition().y))
        extra = filter_extra(g, net, pad_cells(pa) | pad_cells(pb))
        path = astar(g, net, sa, sb, frozenset(extra), width=w)
        if path is None:
            return False
        ta = emit_tie(board, g, nets, net, sa, path[0], w)
        tb = emit_tie(board, g, nets, net, sb, path[-1], w)
        if ta is False or tb is False:
            return False
        emit_path(board, g, nets, net, path, w)
        return True

    authored = 0
    failed = []
    for _, net, pa, pb in edges:
        if try_edge(net, pa, pb):
            authored += 1
        else:
            failed.append((net, pa, pb))
    # one retry pass (copper freed/added since first attempt can open a path)
    still = []
    for net, pa, pb in failed:
        if try_edge(net, pa, pb):
            authored += 1
        else:
            still.append(net)
    print(f"route_critical: authored {authored} edges, {len(still)} failed")
    if still:
        from collections import Counter
        print("   unlocked (-> freerouting):",
              dict(Counter(still)))
    return authored, still


def dedup_vias(board):
    """Delete vias co-located with another same-net via (stitch lattices from
    overlapping zone polygons can drop two vias on the same point -> KiCad
    'holes co-located')."""
    seen = {}
    removed = 0
    for t in list(board.GetTracks()):
        if t.GetClass() != "PCB_VIA":
            continue
        p = t.GetPosition()
        key = (round(pcbnew.ToMM(p.x), 2), round(pcbnew.ToMM(p.y), 2),
               t.GetNetname())
        if key in seen:
            board.Delete(t)
            removed += 1
        else:
            seen[key] = t
    if removed:
        print(f"deduped {removed} co-located vias")
    return removed


def tent_vias(board):
    """Tent all vias top+bottom. Prevents solder wicking on the GND stitch/
    via-in-pad vias (a review flag). NOTE: a via-in-pad sitting inside an SMD
    pad's mask opening is still exposed on that face -> those specifically
    require filled/capped (JLC POFV) treatment at fab; see hardware/README."""
    for t in board.GetTracks():
        if t.GetClass() == "PCB_VIA":
            t.SetFrontTentingMode(pcbnew.TENTING_MODE_TENTED)
            t.SetBackTentingMode(pcbnew.TENTING_MODE_TENTED)


def remove_degenerate_tracks(board):
    """Delete zero-length track segments (start == end). The critical-net
    authoring can emit a coincident pad->path tie; exported to DSN these become
    1-point polylines that crash freerouting's insert_forced_trace_polyline
    (NullPointerException 'from_corner is null')."""
    removed = 0
    for t in list(board.GetTracks()):
        if t.GetClass() == "PCB_TRACK" and t.GetStart() == t.GetEnd():
            board.Delete(t)
            removed += 1
    if removed:
        print(f"removed {removed} zero-length tracks")
    return removed


def finish(board, label):
    remove_degenerate_tracks(board)
    dedup_vias(board)
    tent_vias(board)
    filler = pcbnew.ZONE_FILLER(board)
    filler.Fill(board.Zones())
    pcbnew.SaveBoard(str(PCB_OUT), board)
    board.BuildConnectivity()
    unconn = board.GetConnectivity().GetUnconnectedCount(True)
    print(f"SAVED ({label}): unconnected={unconn}")
    return unconn


def main_scripted():
    board, g, nets, misses, strag_failed = build_preroute()
    escapes = fanout_escapes(board, g, nets)
    print(f"fanout escapes: {len(escapes)}")
    routed, failed, route_order = route_signals(board, g, nets, escapes)
    print(f"pass1: routed {routed} edges, {len(failed)} failed; retrying")
    still = []
    for name, ka, kb in failed:
        pa = board.FindFootprintByReference(ka[0])
        pb = board.FindFootprintByReference(kb[0])
        pada = padb = None
        for pad in pa.Pads():
            if pad.GetNumber() == ka[1]:
                pada = pad
        for pad in pb.Pads():
            if pad.GetNumber() == kb[1]:
                padb = pad
        extra = filter_extra(g, name, pad_cells(pada) | pad_cells(padb))
        sa = escapes.get(ka) or (pcbnew.ToMM(pada.GetPosition().x),
                                 pcbnew.ToMM(pada.GetPosition().y))
        sb = escapes.get(kb) or (pcbnew.ToMM(padb.GetPosition().x),
                                 pcbnew.ToMM(padb.GetPosition().y))
        w2 = NET_WIDTHS.get(name, 0.3)
        path = astar(g, name, sa, sb, frozenset(extra), width=w2)
        if path is None:
            still.append((name, ka, kb))
            continue
        emit_path(board, g, nets, name, path, w2)
        emit_tie(board, g, nets, name, sa, path[0], w2)
        emit_tie(board, g, nets, name, sb, path[-1], w2)
    print(f"pass2: {len(failed) - len(still)} recovered, "
          f"{len(still)} still FAILED")
    hf = SCRIPTS / ".route_hints.json"
    fail_names = []
    for n, _, _ in still:
        if n not in fail_names:
            fail_names.append(n)
    rest = [n for n in route_order if n not in fail_names]
    hf.write_text(json.dumps({"prio": fail_names, "order": rest}))
    purged = purge_track_shorts(board)
    print(f"purged {purged} shorting segments")
    unconn = finish(board, "scripted")
    print(f"route_failures={len(still)}, straggler_failures={len(strag_failed)}")


def main_preroute():
    # DeepPCB flow: no GND via-in-pad (it blocks fine-pitch IC pads; DeepPCB
    # connects GND to the inner planes itself)
    board, g, nets, misses, strag_failed = build_preroute(gnd_drops=False)
    board.SetLayerType(pcbnew.In1_Cu, pcbnew.LT_POWER)
    board.SetLayerType(pcbnew.In2_Cu, pcbnew.LT_POWER)
    unconn = finish(board, "preroute")
    dsn = PROJ / "omnimarble-driver.dsn"
    if dsn.exists():
        dsn.unlink()
    ok = pcbnew.ExportSpecctraDSN(board, str(dsn))
    print(f"DSN export: {ok} -> {dsn}")


def main_import_ses():
    board = pcbnew.LoadBoard(str(PCB_OUT))
    ses = PROJ / "omnimarble-driver.ses"
    ok = pcbnew.ImportSpecctraSES(board, str(ses))
    print(f"SES import: {ok}")

    # strip ALL pulse-net tracks/vias (mine and freerouting's), then rebuild
    # the scripted pulse copper deterministically on the imported board
    removed = inner = 0
    for t in list(board.GetTracks()):
        if t.GetNetname() in PULSE_NETS:
            board.Delete(t)
            removed += 1
        elif (t.GetClass() == "PCB_TRACK"
              and t.GetLayer() in (pcbnew.In1_Cu, pcbnew.In2_Cu)):
            board.Delete(t)
            inner += 1
    print(f"stripped {removed} pulse-net items, {inner} inner-layer tracks")

    nets = {}
    for fp in board.Footprints():
        for pad in fp.Pads():
            n = pad.GetNetname()
            if n and n not in nets:
                nets[n] = pad.GetNet()

    g = build_grid(board)
    build_via_forbid()

    # mark every surviving/imported track and via into the grid so the
    # rebuilt pulse copper and repairs route around them
    for t in board.GetTracks():
        n = t.GetNetname() or "__nc__"
        if t.GetClass() == "PCB_VIA":
            pos = t.GetPosition()
            # KiCad 10: PCB_VIA.GetWidth() with no layer arg fires a
            # blocking wx assert. Read the diameter with a layer arg,
            # falling back to the script's standard 0.6mm via.
            try:
                via_dia = pcbnew.ToMM(t.GetWidth(pcbnew.F_Cu))
            except Exception:
                via_dia = 0.6
            half = via_dia / 2 + 0.36
            r = int(math.ceil(half / GRID))
            for layer in (0, 1):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if (dx * GRID) ** 2 + (dy * GRID) ** 2 <= half * half:
                            g.block(layer,
                                    pcbnew.ToMM(pos.x) + dx * GRID,
                                    pcbnew.ToMM(pos.y) + dy * GRID, n)
        else:
            li = {pcbnew.F_Cu: 0, pcbnew.B_Cu: 1}.get(t.GetLayer())
            if li is None:
                continue
            a, b = t.GetStart(), t.GetEnd()
            mark_swath(g, li, pcbnew.ToMM(a.x), pcbnew.ToMM(a.y),
                       pcbnew.ToMM(b.x), pcbnew.ToMM(b.y), n,
                       pcbnew.ToMM(t.GetWidth()))

    # re-stitch pulse zones (GND stitch vias survived the strip)
    for name, pts in PULSE_ZONES:
        if name == "GND":
            continue
        stitch_zone(board, g, nets, name, pts)

    # kelvin ties: re-add only the stripped (pulse-net) ones
    for ra, pa, rb, pb, w in KELVIN_TRACKS:
        fa = board.FindFootprintByReference(ra)
        fb = board.FindFootprintByReference(rb)
        pada = padb = None
        for pad in fa.Pads():
            if pad.GetNumber() == pa:
                pada = pad
        for pad in fb.Pads():
            if pad.GetNumber() == pb:
                padb = pad
        net = pada.GetNetname()
        if net not in PULSE_NETS:
            continue
        a, b = pada.GetPosition(), padb.GetPosition()
        add_track(board, g, nets, net, 0, pcbnew.ToMM(a.x), pcbnew.ToMM(a.y),
                  pcbnew.ToMM(b.x), pcbnew.ToMM(b.y), w)
    # (manual GND tracks and via clusters survived the strip - not re-added)

    # straggler feeds
    zone_by_net = {}
    for name, pts in PULSE_ZONES:
        zone_by_net.setdefault(name, []).append(pts)
    zlist = [(name, pts, None) for name, pts in PULSE_ZONES]
    misses = check_pulse_pads(board, zlist)
    strag_failed = route_stragglers(board, g, nets, misses, zone_by_net)
    print(f"straggler failures: {strag_failed}")

    # drop GND vias for SMD ground pads freerouting left to the plane, then
    # purge any boundary-touch shorts before saving
    nvias = gnd_via_drops(board, g, nets)
    print(f"GND via drops: {nvias}")
    purged = purge_track_shorts(board)
    print(f"purged {purged} shorting segments")

    finish(board, "freerouted")


def add_shunt_hi_stitch(board, nets):
    """Post-import: drop a via on each pulse-FET source pad down to the solid
    B.Cu SHUNT_HI pour. A router's SHUNT_HI reconnect can leave a source pad
    islanded in the fragmented F.Cu pour (Q10/Q11); this guarantees the tie
    regardless of how the SES routed it. Positions clear of the 5mm stitch grid
    (see VIA_CLUSTERS)."""
    if "SHUNT_HI" not in nets:
        return 0
    n = 0
    for x, y in [(81.0, 71.75), (95.4, 71.75), (115.8, 71.75)]:
        via = pcbnew.PCB_VIA(board)
        via.SetPosition(V(x, y))
        via.SetViaType(pcbnew.VIATYPE_THROUGH)
        via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
        via.SetDrill(MM(0.3))
        via.SetWidth(MM(0.6))
        via.SetNet(nets["SHUNT_HI"])
        board.Add(via)
        n += 1
    return n


def fix_thin_annular(board, min_ann=0.10, min_hole=0.2):
    """Shrink the drill of any via whose annular ring is below min_ann so it
    meets the rule (keeps the via diameter and position). Patches vias an
    imported SES echoed at a thin-ring geometry (e.g. the U10 escape GND via
    that came back 0.45/0.30 = 0.075mm ring). DeepPCB can't edit via padstacks,
    so we do it here at import."""
    n = 0
    for t in board.GetTracks():
        if t.GetClass() != "PCB_VIA":
            continue
        try:                                    # KiCad-10 needs a layer arg on
            w = pcbnew.ToMM(t.GetWidth(pcbnew.F_Cu))  # a via or GetWidth() asserts
        except Exception:
            w = pcbnew.ToMM(t.GetWidth())
        dr = pcbnew.ToMM(t.GetDrillValue())
        if (w - dr) / 2 < min_ann - 1e-6:
            new_dr = round(w - 2 * min_ann, 3)
            if new_dr >= min_hole:
                t.SetDrill(MM(new_dr))
                n += 1
    return n


def local_finish(board, nets):
    """Deterministic post-route touch-ups that finished the board locally
    (rev-43 base) instead of paying for more cloud-router revisions:
      - narrow ISNS_P/N to 0.1mm: the router coupled the Kelvin pair at ~0.02mm
        (unmanufacturable, and the diff-pair mode always did this). Narrowing
        both traces grows the edge gap by 0.2mm everywhere -> clears the 0.2mm
        clearance rule while KEEPING the matched routing. 0.1mm is a fine sense
        trace (uA) and within JLC's 0.089mm capability.
      - drop dead SHUNT_HI stitch vias stranded in isolated F.Cu pour fragments
        (the wide B.Cu bus already carries the current path).
      - bridge the one CIN4 gap the router left open (R59.2 <-> C33.1).
    All are position-guarded no-ops if the feature is absent."""
    for t in board.GetTracks():
        if t.GetClass() == "PCB_TRACK" and t.GetNetname() in ("ISNS_P",
                                                               "ISNS_N"):
            t.SetWidth(MM(0.1))
    dead = [(113.0, 76.5), (123.0, 76.5), (128.0, 76.5)]
    for t in list(board.GetTracks()):
        if t.GetClass() == "PCB_VIA" and t.GetNetname() == "SHUNT_HI":
            x = pcbnew.ToMM(t.GetPosition().x)
            y = pcbnew.ToMM(t.GetPosition().y)
            if any(abs(x - vx) < 0.4 and abs(y - vy) < 0.4 for vx, vy in dead):
                board.Remove(t)
    if "CIN4" in nets:
        tr = pcbnew.PCB_TRACK(board)
        tr.SetStart(V(98.72, 127.255))
        tr.SetEnd(V(101.53, 127.255))
        tr.SetWidth(MM(0.2))
        tr.SetLayer(pcbnew.F_Cu)
        tr.SetNet(nets["CIN4"])
        board.Add(tr)


def main_import_clean():
    """Import a SES from a router that KEPT our fixed copper (DeepPCB) onto a
    fresh placement+pours board, WITHOUT the freerouting-era strip/rebuild/
    gnd-drop passes (those mangle a route that already carries the pulse/GND
    connections). The SES supplies all tracks+vias; the pours come from the
    board zones. Use for DeepPCB SES; use import-ses for freerouting."""
    board = new_board()
    add_outline(board)
    comps, netlist_nets, padnet, meta = load_netlist()
    nets, missing = place_all(board, comps, padnet, meta)
    if missing:
        print(f"UNPLACED refs: {missing}")
    dnp_refs = {"C92", "C93", "C94"}
    exbom_refs = {"TP1", "NT1", "NT2"}
    for fp in board.Footprints():
        if fp.GetReference() in dnp_refs:
            fp.SetDNP(True)
        if fp.GetReference() in exbom_refs:
            fp.SetExcludedFromBOM(True)
    build_zones(board, nets)
    board.SetLayerType(pcbnew.In1_Cu, pcbnew.LT_POWER)
    board.SetLayerType(pcbnew.In2_Cu, pcbnew.LT_POWER)
    pcbnew.SaveBoard(str(PCB_OUT), board)

    board = pcbnew.LoadBoard(str(PCB_OUT))
    ses = PROJ / "omnimarble-driver.ses"
    ok = pcbnew.ImportSpecctraSES(board, str(ses))
    print(f"SES import: {ok}")
    # outer F/B GND flood added AFTER the route (pours around finished traces)
    nets = {}
    for fp in board.Footprints():
        for pad in fp.Pads():
            n = pad.GetNetname()
            if n and n not in nets:
                nets[n] = pad.GetNet()
    add_fb_gnd_pours(board, nets)
    local_finish(board, nets)
    nfix = fix_thin_annular(board)
    if nfix:
        print(f"annular fix: shrank drill on {nfix} thin-ring via(s)")
    tent_vias(board)
    dedup_vias(board)
    remove_degenerate_tracks(board)
    filler = pcbnew.ZONE_FILLER(board)
    filler.Fill(board.Zones())
    pcbnew.SaveBoard(str(PCB_OUT), board)
    board.BuildConnectivity()
    print(f"SAVED (import-clean): unconnected="
          f"{board.GetConnectivity().GetUnconnectedCount(True)}")


def main_repair():
    """Route leftover unconnected pairs from the last DRC report."""
    import re
    board = pcbnew.LoadBoard(str(PCB_OUT))
    drc = json.loads((HW_DIR / "fab" / "drc_driver.json").read_text(
        encoding="utf-8"))

    nets = {}
    for fp in board.Footprints():
        for pad in fp.Pads():
            n = pad.GetNetname()
            if n and n not in nets:
                nets[n] = pad.GetNet()

    g = build_grid(board)
    build_via_forbid()
    for t in board.GetTracks():
        n = t.GetNetname() or "__nc__"
        if t.GetClass() == "PCB_VIA":
            pos = t.GetPosition()
            # KiCad 10: PCB_VIA.GetWidth() with no layer arg fires a
            # blocking wx assert. Read the diameter with a layer arg,
            # falling back to the script's standard 0.6mm via.
            try:
                via_dia = pcbnew.ToMM(t.GetWidth(pcbnew.F_Cu))
            except Exception:
                via_dia = 0.6
            half = via_dia / 2 + 0.36
            r = int(math.ceil(half / GRID))
            for layer in (0, 1):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if (dx * GRID) ** 2 + (dy * GRID) ** 2 <= half * half:
                            g.block(layer,
                                    pcbnew.ToMM(pos.x) + dx * GRID,
                                    pcbnew.ToMM(pos.y) + dy * GRID, n)
        else:
            li = {pcbnew.F_Cu: 0, pcbnew.B_Cu: 1}.get(t.GetLayer())
            if li is None:
                continue
            a, b = t.GetStart(), t.GetEnd()
            mark_swath(g, li, pcbnew.ToMM(a.x), pcbnew.ToMM(a.y),
                       pcbnew.ToMM(b.x), pcbnew.ToMM(b.y), n,
                       pcbnew.ToMM(t.GetWidth()))

    ok = fail = skipped = 0
    for u in drc.get("unconnected_items", []):
        items = u["items"]
        if len(items) != 2 or any("Zone" in it["description"]
                                  for it in items):
            skipped += 1
            continue
        m = re.search(r"\[(.+?)\]", items[0]["description"])
        if not m:
            skipped += 1
            continue
        net = m.group(1)
        if net in PULSE_NETS or net == "GND":
            skipped += 1
            continue
        pa = (items[0]["pos"]["x"], items[0]["pos"]["y"])
        pb = (items[1]["pos"]["x"], items[1]["pos"]["y"])
        w = NET_WIDTHS.get(net, 0.3)
        path = astar(g, net, pa, pb, frozenset(), width=w)
        if path is None:
            fail += 1
            print(f"   REPAIR FAIL {net} {pa} {pb}")
            continue
        emit_path(board, g, nets, net, path, w)
        emit_tie(board, g, nets, net, pa, path[0], w)
        emit_tie(board, g, nets, net, pb, path[-1], w)
        ok += 1
    print(f"repair: {ok} routed, {fail} failed, {skipped} skipped")
    purged = purge_track_shorts(board)
    print(f"repair purge: {purged} shorting segments removed")
    finish(board, "repaired")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scripted"
    if mode == "preroute":
        main_preroute()
    elif mode == "import-ses":
        main_import_ses()
    elif mode == "import-clean":
        main_import_clean()
    elif mode == "repair":
        main_repair()
    else:
        main_scripted()
