"""Debug: why does SNB1 (R36.2 -> C23.1) fail to route?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pcbnew
import gen_pcb as G

board = pcbnew.LoadBoard(str(G.PCB_OUT))
g = G.build_grid(board)

fps = {fp.GetReference(): fp for fp in board.Footprints()}
r36 = fps["J11"]
c23 = fps["J9"]
for fp in (r36, c23):
    print(fp.GetReference(), "at",
          pcbnew.ToMM(fp.GetPosition().x), pcbnew.ToMM(fp.GetPosition().y))
    for pad in fp.Pads():
        pp = pad.GetPosition()
        sz = pad.GetSize()
        print("  pad", pad.GetNumber(), pad.GetNetname(),
              (pcbnew.ToMM(pp.x), pcbnew.ToMM(pp.y)),
              "size", (round(pcbnew.ToMM(sz.x), 2), round(pcbnew.ToMM(sz.y), 2)))

# occupancy map around the snubber column
pads = {}
for fp in (r36, c23):
    for pad in fp.Pads():
        pads[(fp.GetReference(), pad.GetNumber())] = pad
pa = pads[("J11", "1")]
pb = pads[("J9", "4")]
sa = (pcbnew.ToMM(pa.GetPosition().x), pcbnew.ToMM(pa.GetPosition().y))
sb = (pcbnew.ToMM(pb.GetPosition().x), pcbnew.ToMM(pb.GetPosition().y))
print("route", sa, "->", sb)

x0 = int((sa[0] - 4) / G.GRID)
x1 = int((sa[0] + 4) / G.GRID)
ylo = int((min(sa[1], sb[1]) - 4) / G.GRID)
yhi = int((max(sa[1], sb[1]) + 4) / G.GRID)
net = "GATE1"
for iy in range(ylo, yhi + 1):
    row = ""
    for ix in range(x0, x1 + 1):
        i = g.idx(ix, iy)
        if not g.blocked[0][i]:
            row += "."
        else:
            own = g.owner[0].get(i)
            if own == net:
                row += "s"
            elif own == "SW_DRAIN":
                row += "W"
            elif own == "SHUNT_HI":
                row += "H"
            elif own is None:
                row += "#"
            else:
                row += own[0].lower()
    print(f"y={iy * G.GRID:6.1f} {row}")

extra = G.pad_cells(pa) | G.pad_cells(pb)
path = G.astar(g, net, sa, sb, frozenset(extra))
print("astar:", "OK len=%d" % len(path) if path else "FAIL")
