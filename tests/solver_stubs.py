"""Field-solver stubs for tests that need simulate_shot without the PINN.

GradientSolver is the smooth bump from tests/test_rig_shot.py, duplicated
here so the impulse-map and sustain suites can import it without importing
that module's tests -- with one change: it is scaled PER AMP against the
same 211 A reference ConstantForceSolver uses. The original is 211x too
strong (Bz ~ 10 T at 200 A, dv ~ 50 m/s), which its only test never sees
because it checks a ratio. Here the impulse must be of the rig's order
(tens of mm/s) or the map's v_in axis and the saturation cap behave
unphysically.
"""

import math

REF_CURRENT_A = 211.0


class GradientSolver:
    """Field with a curved axial profile, peaked at peak_z (mm); force ~ I^2."""

    def __init__(self, peak_z=-15.0, width=12.0, bz=0.05, dbz_dz=7.2e-3):
        self.peak_z, self.width = peak_z, width
        self.bz_per_A = bz / REF_CURRENT_A
        self.dbz_dz_per_A = dbz_dz / REF_CURRENT_A

    def field_with_grad(self, r, z, current_A):
        u = (z - self.peak_z) / self.width
        shape = math.exp(-u * u)
        radial = 1.0 - 0.02 * (r / 6.35) ** 2
        return (0.0, self.bz_per_A * current_A * shape * radial, 0.0, 0.0, 0.0,
                self.dbz_dz_per_A * current_A * shape * radial)
