"""How wrong is treating the ball as a point dipole?

    uv run python scripts/check_finite_size.py

Both force paths evaluate (B.grad)B at the ball's CENTRE and multiply by the
full volume. That is a point-dipole approximation, and this ball is not small
against the field: radius 6.35 mm against a field varying on the coil's ~15 mm
scale, so (a/lambda)^2 ~ 0.18. The audit put the resulting error at "5-20%,
sign unknown" -- the largest un-quantified approximation left in the chain, and
comparable to the effects we have been carefully correcting.

Rather than build a quadrature into the hot loop on the strength of an
estimate, this measures it: integrate the force density over the sphere with
Gauss-Legendre product rules and compare against the point value, at the fire
position that actually matters and across the coil.

The magnetisation is taken as uniform (the ball is small enough that the
demagnetising correction is a global factor which cancels in the ratio), so
this bounds the FIELD-VARIATION part of the finite-size error, which is the
part that scales as (a/lambda)^2. It does not address magnetisation lag or
mutual demagnetisation.
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "source" / "extensions" / "omni.marble.coaster"
                       / "omni" / "marble" / "coaster"))

from rig_profile import load_profile  # noqa: E402
from simulate_rig_shot import finite_size_factor  # noqa: E402

MU_0_MM = 4 * math.pi * 1e-4


def force_density_axial(solver, r, z, current, chi_eff):
    """(B.grad)B_z at one point, per unit volume (the integrand)."""
    Br, Bz, dBr_dr, dBr_dz, dBz_dr, dBz_dz = solver.field_with_grad(r, z, current)
    return (chi_eff / MU_0_MM) * (Br * dBz_dr + Bz * dBz_dz)


def main():
    profile = load_profile(ROOT, "vbench_v0")
    radius = profile.marble["radius_mm"]
    chi_eff = 3.0
    current = 211.0                      # the rig's 1-can peak

    params = json.loads((ROOT / "config" / "coil_params.json").read_text())
    params["num_turns"] = profile.coil["num_turns"]
    params["length_mm"] = profile.coil["length_mm"]
    from warp_bfield_solver import WarpBFieldSolver
    solver = WarpBFieldSolver(params, chi_eff=chi_eff)

    face_in = profile.coil["face_in_x_mm"]
    half_winding = profile.coil["length_mm"] / 2.0
    lam = profile.coil["loop_center_radius_mm"]
    print(f"ball radius {radius} mm, field scale ~{lam} mm, "
          f"(a/lambda)^2 = {(radius/lam)**2:.3f}")
    print(f"fire position {face_in:.2f} mm (coil entry face), "
          f"winding end {-half_winding:.1f} mm\n")

    print(f"{'z (mm)':>8}{'point':>14}{'volume-avg':>14}{'ratio':>9}{'error':>9}")
    print("-" * 54)
    positions = [-40.0, -30.0, face_in, -20.0, -15.0, -10.0, -5.0, 0.0]
    worst = 0.0
    for z in positions:
        point = force_density_axial(solver, 0.0, z, current, chi_eff)
        avg = point * finite_size_factor(solver, z, radius, current)
        if abs(point) < 1e-9:
            print(f"{z:>8.2f}{point:>14.4g}{avg:>14.4g}{'--':>9}{'--':>9}")
            continue
        ratio = avg / point
        err = ratio - 1.0
        marker = "  <-- fire point" if abs(z - face_in) < 1e-6 else ""
        print(f"{z:>8.2f}{point:>14.4g}{avg:>14.4g}{ratio:>9.4f}"
              f"{err:>8.2%}{marker}")
        if abs(z - face_in) < 1e-6 or z in (-15.0, -20.0):
            worst = max(worst, abs(err))

    print(f"\nAt and around the fire position the finite-size correction is "
          f"{worst:.2%}.")
    print("Convergence check (samples per dimension):")
    for n in (4, 6, 8, 12):
        avg = finite_size_factor(solver, face_in, radius, current,
                                 n_r=n, n_theta=n)
        print(f"  n={n:>3}: {avg:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
