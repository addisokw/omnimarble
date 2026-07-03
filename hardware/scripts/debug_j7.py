import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcbnew
import gen_pcb as G
board = pcbnew.LoadBoard(str(G.PCB_OUT))
for fp in board.Footprints():
    if fp.GetReference() == "J7":
        print("J7 anchor", pcbnew.ToMM(fp.GetPosition().x),
              pcbnew.ToMM(fp.GetPosition().y),
              "rot", fp.GetOrientationDegrees())
        for pad in fp.Pads():
            pp = pad.GetPosition()
            print(" pad", pad.GetNumber(), pad.GetNetname(),
                  round(pcbnew.ToMM(pp.x), 2), round(pcbnew.ToMM(pp.y), 2))
