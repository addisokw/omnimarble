"""Validator: prove the routed board honours the intended per-net track widths.

Because `new_board()` builds on a fresh board that bypasses `.kicad_pro` (whose
netclasses stay flat 0.2mm), the real routing intent lives in
`placement.NET_WIDTHS` + the DSN net classes, not in the KiCad project. This
script closes that gap: it loads the actual `.kicad_pcb` and asserts every net
in NET_WIDTHS is routed at (approximately) its intended width, so a reviewer can
trust the imported copper matches intent without reading the DSN.

A net passes if the MEDIAN width of its routed segments is within tolerance of
the intended width (freerouting necks down at pad entries, so we allow thinner
spurs but flag a net whose bulk copper is wrong). Reports thinner-than-intended
nets; exits non-zero if any intended net is >20% under spec on its median.

Run: "C:\\Program Files\\KiCad\\10.0\\bin\\python.exe" hardware/scripts/validate_widths.py
"""

import statistics
import sys
from pathlib import Path

import pcbnew

HW_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HW_DIR / "scripts"))
from placement import NET_WIDTHS, PLANE_NETS  # noqa: E402

PCB = HW_DIR / "omnimarble-driver" / "omnimarble-driver.kicad_pcb"
TOL = 0.20  # median may be up to 20% under intended before we fail


def main():
    board = pcbnew.LoadBoard(str(PCB))
    widths = {}
    for t in board.GetTracks():
        if t.GetClass() != "PCB_TRACK":
            continue
        n = t.GetNetname()
        widths.setdefault(n, []).append(pcbnew.ToMM(t.GetWidth()))

    fails = []
    print(f"{'net':14s} {'intended':>9s} {'median':>8s} {'min':>6s} {'segs':>5s}")
    for net, want in sorted(NET_WIDTHS.items()):
        segs = widths.get(net, [])
        if not segs:
            # plane nets are pours, not tracks -- absence is expected
            note = "pour/plane" if net in PLANE_NETS else "NO TRACKS"
            print(f"{net:14s} {want:9.2f} {'-':>8s} {'-':>6s}  {note}")
            if net not in PLANE_NETS:
                fails.append((net, "no routed tracks"))
            continue
        med = statistics.median(segs)
        flag = "" if med >= want * (1 - TOL) else "  << UNDER"
        if flag:
            fails.append((net, f"median {med:.2f} < {want:.2f}"))
        print(f"{net:14s} {want:9.2f} {med:8.2f} {min(segs):6.2f} "
              f"{len(segs):5d}{flag}")

    print()
    if fails:
        print(f"FAIL: {len(fails)} net(s) under intended width:")
        for n, why in fails:
            print(f"  {n}: {why}")
        sys.exit(1)
    print("PASS: all intended nets routed at spec width (median within "
          f"{int(TOL*100)}%).")


if __name__ == "__main__":
    main()
