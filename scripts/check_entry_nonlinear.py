"""Can ANY local M(B) law, volume-integrated over the real 12.7 mm ball,
broaden the entry side of the fire-position curve at high current?

Static check: force(x) = integral over the ball of chi(B_local) * (B.grad)B
at the peak current, for several chi laws, at 298 A (49 V) and 187 A (30 V).
Ratios relative to +9 are what the bench measures.
"""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/Users/boxga/Documents/git/omnimarble")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "source" / "extensions" / "omni.marble.coaster" / "omni" / "marble" / "coaster"))
from rig_profile import load_profile  # noqa: E402
from warp_bfield_solver import WarpBFieldSolver  # noqa: E402

profile = load_profile(ROOT, "vbench_v0")
params = json.loads((ROOT / "config" / "coil_params.json").read_text())
params["num_turns"] = profile.coil["num_turns"]
params["length_mm"] = profile.coil["length_mm"]
solver = WarpBFieldSolver(params, chi_eff=3.0)
R = 6.35

# Gauss-Legendre nodes over the sphere (rho, cos theta), azimuth exact.
rho_n, rho_w = np.polynomial.legendre.leggauss(8)
cos_n, cos_w = np.polynomial.legendre.leggauss(8)
nodes = []
for node, wr in zip(rho_n, rho_w):
    rr = 0.5 * R * (node + 1.0)
    for ct, wt in zip(cos_n, cos_w):
        st = math.sqrt(max(0.0, 1.0 - ct * ct))
        nodes.append((rr * st, rr * ct, rr * rr * 0.5 * R * wr * wt))

xs = [-28.78, -25.78, -22.78, -19.78, -16.78, -13.78, -10.78, -7.78]
# Field per amp at every node for every centre position (linear in I).
cache = {}
for x in xs:
    rows = []
    for r, dz, w in nodes:
        Br, Bz, dBr_dr, dBr_dz, dBz_dr, dBz_dz = solver.field_with_grad(r, x + dz, 1.0)
        rows.append((w, Br, Bz, dBr_dr, dBr_dz, dBz_dr, dBz_dz))
    cache[x] = rows

def force(x, I, chi_fn):
    tot = 0.0
    for w, Br, Bz, dBr_dr, dBr_dz, dBz_dr, dBz_dz in cache[x]:
        Br_, Bz_ = Br * I, Bz * I
        Bmag = math.hypot(Br_, Bz_)
        dens = Br_ * dBz_dr * I + Bz_ * dBz_dz * I   # axial (B.grad)B
        tot += w * chi_fn(Bmag) * dens
    return tot

laws = {
    "linear chi=3":            lambda B: 3.0,
    "hard cap Bsat 0.5 T":     lambda B: min(3.0, 0.5 / max(3 * B, 1e-9) * 3),
    "hard cap Bsat 0.3 T":     lambda B: min(3.0, 0.3 / max(3 * B, 1e-9) * 3),
    "soft sat tanh 0.4 T":     lambda B: 3.0 * math.tanh(3 * B / 0.4) / max(3 * B / 0.4, 1e-9),
    "rising chi 2->3, B0 0.2": lambda B: 2.0 + 1.0 * (1 - math.exp(-3 * B / 0.2)),
    "rising chi 1->3, B0 0.3": lambda B: 1.0 + 2.0 * (1 - math.exp(-3 * B / 0.3)),
}
meas49 = [0.15, 0.31, 0.70, 0.99, 1.06, 1.00, 0.88, 0.68]
print("offsets      : " + " ".join("%6.0f" % (x + 22.78) for x in xs))
print("measured 49V : " + " ".join("%6.2f" % v for v in meas49))
for name, fn in laws.items():
    for I in (298.0, 187.0):
        f = [force(x, I, fn) for x in xs]
        ref = f[5]
        print("%-24s %3.0f A: " % (name, I) + " ".join("%6.2f" % (v / ref) for v in f)
              + "   F(+9) rel linear %.2f" % (ref / force(-13.78, I, laws["linear chi=3"])))
