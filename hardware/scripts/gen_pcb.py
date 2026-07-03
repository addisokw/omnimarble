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
import sys
from pathlib import Path

import pcbnew

HW_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = HW_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS))

from kicad_sch import Sym, find, find_all, parse  # stdlib-only parser
from placement import (AUTO_CLUSTERS, BOARD_H, BOARD_W, BRIDGE_TABLE,
                       KELVIN_TRACKS, MANUAL_TRACKS, MOUNT_HOLES,
                       NET_WIDTHS, P, PULSE_NETS, PULSE_ZONES,
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
    for c in find_all(find(tree, Sym("components")), Sym("comp")):
        ref = str(find(c, Sym("ref"))[1])
        fp = find(c, Sym("footprint"))
        comps[ref] = str(fp[1]) if fp else ""
    nets = {}
    padnet = {}
    for n in find_all(find(tree, Sym("nets")), Sym("net")):
        name = str(find(n, Sym("name"))[1])
        for node in find_all(n, Sym("node")):
            ref = str(find(node, Sym("ref"))[1])
            pin = str(find(node, Sym("pin"))[1])
            nets.setdefault(name, []).append((ref, pin))
            padnet[(ref, pin)] = name
    return comps, nets, padnet


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


def place_all(board, comps, padnet):
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
            "MountingHole:MountingHole_3.2mm_M3")
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


def build_zones(board, nets):
    zones = []
    prio = 2
    for name, pts in PULSE_ZONES:
        prio += 1
        for layer in (pcbnew.F_Cu, pcbnew.B_Cu):
            zones.append((name, pts,
                          add_zone(board, nets[name], layer, pts,
                                   clearance=1.0, min_thick=0.5,
                                   priority=prio)))
    # Full-board GND planes on inner layers
    frame = [(2, 2), (BOARD_W - 2, 2), (BOARD_W - 2, BOARD_H - 2),
             (2, BOARD_H - 2)]
    for layer in (pcbnew.In1_Cu, pcbnew.In2_Cu):
        add_zone(board, nets["GND"], layer, frame, clearance=0.3,
                 priority=0)
    return zones


def stitch_zone(board, netinfo, pts, pitch=5.0):
    """Via-stitch a pulse zone polygon interior (F<->B<->planes)."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    added = 0
    y = min(ys) + 2.0
    while y < max(ys) - 1.0:
        x = min(xs) + 2.0
        while x < max(xs) - 1.0:
            if point_in_poly(x, y, pts):
                via = pcbnew.PCB_VIA(board)
                via.SetPosition(V(x, y))
                via.SetDrill(MM(0.6))
                via.SetWidth(MM(1.2))
                via.SetViaType(pcbnew.VIATYPE_THROUGH)
                via.SetNet(netinfo)
                board.Add(via)
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
    clr = 0.30
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
                g.block_rect(layer, x0, y0, x1, y1, net)
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


def mark_swath(g, layer, x0, y0, x1, y1, net, width):
    # 0.3mm tracks land exactly on the 0.5mm grid: neighbours at 0.5 pitch
    # leave a legal 0.2mm gap. Wider tracks must reserve clearance for a
    # worst-case 0.3mm neighbour.
    half = width / 2 + (0.15 if width <= 0.31 else 0.36)
    steps = max(1, int(max(abs(x1 - x0), abs(y1 - y0)) / GRID))
    for i in range(steps + 1):
        x = x0 + (x1 - x0) * i / steps
        y = y0 + (y1 - y0) * i / steps
        r = int(math.ceil(half / GRID))
        cix, ciy = int(round(x / GRID)), int(round(y / GRID))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if (dx * GRID) ** 2 + (dy * GRID) ** 2 <= half * half:
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
                    if abs(x - px) < 1.0 and abs(y - py) < 1.0:
                        continue  # own pad vicinity
                    if not g.is_free(0, int(round(x / GRID)),
                                     int(round(y / GRID)), net):
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
            if abs(dx) >= abs(dy):
                ux, uy = (1 if dx >= 0 else -1), 0
            else:
                ux, uy = 0, (1 if dy >= 0 else -1)
            base = (1.2, 1.6, 2.0, 2.5, 3.0)
            smd_idx = sum(ord(ch) for ch in net) % 3
            for dist in base[smd_idx:] + base[:smd_idx]:
                ex, ey = px + ux * dist, py + uy * dist
                eix, eiy = int(round(ex / GRID)), int(round(ey / GRID))
                if g.is_free(0, eix, eiy, net):
                    add_track(board, g, nets, net, 0, px, py, ex, ey, 0.25)
                    escapes[(fp.GetReference(), pad.GetNumber())] = (ex, ey)
                    break
    return escapes


def astar(g, net, start, goal, extra):
    import heapq
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
            nn = (layer, ix + dx, iy + dy)
            if not g.is_free(layer, ix + dx, iy + dy, net, extra):
                continue
            nc = cost[node] + 1
            if nc < cost.get(nn, 1 << 30):
                cost[nn] = nc
                came[nn] = node
                heapq.heappush(openq, (nc + abs(ix + dx - gx)
                                       + abs(iy + dy - gy), nn))
        ol = 1 - layer
        if g.is_free(ol, ix, iy, net, extra) and via_ok(ix, iy, net):
            nn = (ol, ix, iy)
            nc = cost[node] + VIA_COST
            if nc < cost.get(nn, 1 << 30):
                cost[nn] = nc
                came[nn] = node
                heapq.heappush(openq, (nc + abs(ix - gx) + abs(iy - gy), nn))
    return None


def emit_tie(board, g, nets, name, exact, node, width):
    ex, ey = exact
    cx, cy = node[1] * GRID, node[2] * GRID
    if abs(ex - cx) < 0.01 and abs(ey - cy) < 0.01:
        return
    lm = {0: pcbnew.F_Cu, 1: pcbnew.B_Cu}
    t = pcbnew.PCB_TRACK(board)
    t.SetStart(V(ex, ey))
    t.SetEnd(V(cx, cy))
    t.SetWidth(MM(min(width, 0.3)))
    t.SetLayer(lm[node[0]])
    t.SetNet(nets[name])
    board.Add(t)


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
        path = astar(g, net, (px, py), (best[1], best[2]), frozenset(extra))
        if path is None:
            failed.append((net, ref, padnum))
            continue
        emit_path(board, g, nets, net, path, width)
        emit_tie(board, g, nets, net, (px, py), path[0], width)
    return failed


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
            done = False
            for ddx, ddy in [(0, 1.4), (0, -1.4), (1.4, 0), (-1.4, 0),
                             (1.2, 1.2), (-1.2, 1.2), (1.2, -1.2),
                             (-1.2, -1.2), (0, 2.2), (0, -2.2),
                             (2.2, 0), (-2.2, 0), (0, 3.0), (0, -3.0),
                             (3.0, 0), (-3.0, 0), (2.2, 2.2),
                             (-2.2, 2.2), (2.2, -2.2), (-2.2, -2.2)]:
                vx, vy = px + ddx, py + ddy
                vix, viy = int(round(vx / GRID)), int(round(vy / GRID))
                if not (g.is_free(0, vix, viy, "GND")
                        and g.is_free(1, vix, viy, "GND")
                        and via_ok(vix, viy, "GND")):
                    continue
                add_via(board, g, nets, "GND", vx, vy)
                add_track(board, g, nets, "GND", 0, px, py, vx, vy, 0.4)
                added += 1
                done = True
                break
            if not done:
                print(f"   GND drop FAILED: {fp.GetReference()}"
                      f".{pad.GetNumber()} at ({px:.1f},{py:.1f})")
    return added


def route_signals(board, g, nets, escapes):
    """Negotiated congestion routing: signal tracks are soft obstacles that
    other nets may cross at a per-iteration penalty; overused cells build
    history cost until every edge has a private path (PathFinder-lite).
    Hard obstacles (pads, pulse zones, stubs, GND vias) stay in `g`."""
    import heapq

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
            extra = pad_cells(padobjs[keys[i]]) | pad_cells(padobjs[keys[j]])
            edges_all.append([len(edges_all), name, keys[i], keys[j],
                              pts[i], pts[j], width, frozenset(extra)])

    usage = [{}, {}]   # cellidx -> {netname: refcount}
    hist = [{}, {}]
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

    def occupy(name, cells):
        for (l, ix, iy) in cells:
            i = ix * g.ny + iy
            d = usage[l].setdefault(i, {})
            d[name] = d.get(name, 0) + 1

    def vacate(name, cells):
        for (l, ix, iy) in cells:
            i = ix * g.ny + iy
            d = usage[l].get(i)
            if d and name in d:
                d[name] -= 1
                if d[name] <= 0:
                    del d[name]

    def has_conflict(eid, name):
        for (l, ix, iy) in eswaths.get(eid, ()):
            d = usage[l].get(ix * g.ny + iy)
            if d and any(k != name for k in d):
                return True
        return False

    def disc_offsets(width):
        half = width / 2 + (0.15 if width <= 0.31 else 0.36)
        r = int(math.ceil(half / GRID))
        offs = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if (dx * GRID) ** 2 + (dy * GRID) ** 2 <= half * half:
                    offs.append((dx, dy))
        return offs

    _disc_cache = {}

    def pen(l, ix, iy, name, p_now, offs):
        c = 0
        clash = False
        for dx, dy in offs:
            i = (ix + dx) * g.ny + iy + dy
            c += hist[l].get(i, 0)
            if not clash:
                d = usage[l].get(i)
                if d and any(k != name for k in d):
                    clash = True
        if clash:
            c += p_now
        return c

    def astar_neg(name, start, goal, extra, p_now, width):
        offs = _disc_cache.get(width)
        if offs is None:
            offs = disc_offsets(width)
            _disc_cache[width] = offs
        sx, sy = int(round(start[0] / GRID)), int(round(start[1] / GRID))
        gx, gy = int(round(goal[0] / GRID)), int(round(goal[1] / GRID))
        startn = (0, sx, sy)
        goals = {(0, gx, gy), (1, gx, gy)}
        openq = [(0, startn)]
        came = {startn: None}
        cost = {startn: 0}
        pops = 0
        while openq and pops < 600000:
            _, node = heapq.heappop(openq)
            pops += 1
            if node in goals:
                path = []
                while node:
                    path.append(node)
                    node = came[node]
                return path[::-1]
            l, ix, iy = node
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nix, niy = ix + dx, iy + dy
                if not g.is_free(l, nix, niy, name, extra):
                    continue
                nn = (l, nix, niy)
                nc = cost[node] + 1 + pen(l, nix, niy, name, p_now, offs)
                if nc < cost.get(nn, 1 << 30):
                    cost[nn] = nc
                    came[nn] = node
                    heapq.heappush(openq, (nc + abs(nix - gx)
                                           + abs(niy - gy), nn))
            ol = 1 - l
            if g.is_free(ol, ix, iy, name, extra) and via_ok(ix, iy, name):
                nn = (ol, ix, iy)
                nc = cost[node] + 10 + pen(ol, ix, iy, name, p_now, offs)
                if nc < cost.get(nn, 1 << 30):
                    cost[nn] = nc
                    came[nn] = node
                    heapq.heappush(openq, (nc + abs(ix - gx)
                                           + abs(iy - gy), nn))
        return None

    hard_failed = []
    for it in range(40):
        p_now = 4 + it * 3
        if it == 0:
            work = list(edges_all)
        else:
            work = [e for e in edges_all
                    if e[0] not in epaths or has_conflict(e[0], e[1])]
            if not work:
                break
        for e in work:
            eid, name, ka, kb, pa, pb, width, extra = e
            if eid in epaths:
                vacate(name, eswaths[eid])
                del epaths[eid], eswaths[eid]
            path = astar_neg(name, pa, pb, extra, p_now, width)
            if path is None:
                continue
            sw = swath_of(path, width)
            epaths[eid] = path
            eswaths[eid] = sw
            occupy(name, sw)
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
                    if d and any(k != name for k in d):
                        hist[l][i] = hist[l].get(i, 0) + 1
        missing = [e for e in edges_all if e[0] not in epaths]
        print(f"   negotiate iter {it}: rerouted {len(work)}, "
              f"conflicted {over}, unrouted {len(missing)}")
        if over == 0 and not missing:
            break

    failed = []
    routed = 0
    for e in edges_all:
        eid, name, ka, kb, pa, pb, width, extra = e
        if eid not in epaths or has_conflict(eid, name):
            failed.append((name, ka, kb))
            continue
        path = epaths[eid]
        emit_path(board, g, nets, name, path, width)
        emit_tie(board, g, nets, name, pa, path[0], width)
        emit_tie(board, g, nets, name, pb, path[-1], width)
        routed += 1
    return routed, failed, [e[1] for e in edges_all]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    comps, netlist_nets, padnet = load_netlist()
    board = new_board()
    add_outline(board)
    nets, missing = place_all(board, comps, padnet)
    if missing:
        print(f"UNPLACED refs: {missing}")

    # courtyard/bbox overlap check (catches placement collisions)
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

    for name, pts, z in zones:
        if z.GetLayer() == pcbnew.F_Cu:
            stitch_zone(board, nets[name], pts)

    g = build_grid(board)
    build_via_forbid()

    # explicit Kelvin sense tracks (shunt -> net-ties -> INA240)
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

    # manual fat tracks (e.g. clamp anodes to their via cluster)
    for ref, padnum, (tx, ty), w, layer_i in MANUAL_TRACKS:
        fp = board.FindFootprintByReference(ref)
        for pad in fp.Pads():
            if pad.GetNumber() == padnum:
                pp = pad.GetPosition()
                add_track(board, g, nets, pad.GetNetname(), layer_i,
                          pcbnew.ToMM(pp.x), pcbnew.ToMM(pp.y), tx, ty, w)

    # surge via clusters
    for net, cx, cy, nx, ny, pitch, drill, size in VIA_CLUSTERS:
        for ix in range(nx):
            for iy in range(ny):
                add_via(board, g, nets, net,
                        cx + (ix - (nx - 1) / 2) * pitch,
                        cy + (iy - (ny - 1) / 2) * pitch, drill, size)

    strag_failed = route_stragglers(board, g, nets, misses, zone_by_net)
    print(f"straggler failures: {strag_failed}")

    escapes = fanout_escapes(board, g, nets)
    print(f"fanout escapes: {len(escapes)}")
    for r in ("J6", "J7", "J9", "J10", "J11"):
        got = sorted(pn for (rr, pn) in escapes if rr == r)
        print(f"   {r} escapes: {got}")

    nvias = gnd_via_drops(board, g, nets)
    print(f"GND via drops: {nvias}")

    routed, failed, route_order = route_signals(board, g, nets, escapes)
    print(f"pass1: routed {routed} edges, {len(failed)} failed; retrying")
    # phase 2: retry failures (board state has changed; some now succeed)
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
        extra = pad_cells(pada) | pad_cells(padb)
        sa = escapes.get(ka) or (pcbnew.ToMM(pada.GetPosition().x),
                                 pcbnew.ToMM(pada.GetPosition().y))
        sb = escapes.get(kb) or (pcbnew.ToMM(padb.GetPosition().x),
                                 pcbnew.ToMM(padb.GetPosition().y))
        path = astar(g, name, sa, sb, frozenset(extra))
        if path is None:
            still.append((name, ka, kb))
            continue
        w2 = NET_WIDTHS.get(name, 0.3)
        emit_path(board, g, nets, name, path, w2)
        emit_tie(board, g, nets, name, sa, path[0], w2)
        emit_tie(board, g, nets, name, sb, path[-1], w2)
    print(f"pass2: {len(failed) - len(still)} recovered, "
          f"{len(still)} still FAILED")
    # failures first, then the order just used (stable fixed-point)
    hf = SCRIPTS / ".route_hints.json"
    prio = []
    for n, _, _ in still:
        if n not in prio:
            prio.append(n)
    rest = [n for n in route_order if n not in prio]
    hf.write_text(json.dumps({"prio": prio, "order": rest}))
    for f in still[:30]:
        print("   FAIL", f)
    failed = still

    add_text(board, "DANGER - STORED ENERGY", 100, 6, 2.5)
    add_text(board, "CAPACITORS MAY BE CHARGED - CHECK LIVE LED", 100, 10, 1.5)
    add_text(board, "COIL: L>=1uH R_total>=90mOhm", 30, 36, 1.2)

    filler = pcbnew.ZONE_FILLER(board)
    filler.Fill(board.Zones())

    pcbnew.SaveBoard(str(PCB_OUT), board)

    board.BuildConnectivity()
    unconn = board.GetConnectivity().GetUnconnectedCount(True)
    print(f"SAVED: unconnected={unconn}, route_failures={len(failed)}, "
          f"straggler_failures={len(strag_failed)}")


if __name__ == "__main__":
    main()
