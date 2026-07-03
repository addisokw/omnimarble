"""Occupancy map of the Pico/J11 region."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcbnew
import gen_pcb as G

board = pcbnew.LoadBoard(str(G.PCB_OUT))
g = G.build_grid(board)
x0, x1 = int(160 / G.GRID), int(214 / G.GRID)
for iy in range(int(56 / G.GRID), int(134 / G.GRID) + 1, 2):
    row = ""
    for ix in range(x0, x1 + 1):
        i = g.idx(ix, iy)
        if not g.blocked[0][i]:
            row += "."
        else:
            own = g.owner[0].get(i)
            row += (own[0] if own else "#")
    print(f"y={iy * G.GRID:5.1f} {row}")
