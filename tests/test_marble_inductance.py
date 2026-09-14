"""The marble's inductance perturbation is the force law in disguise.

TWIN_AUDIT.md S-6: the coupled ODE's L_effective() used a trapezoid overlap
scaled by 0.01 and disagreed with the force model by 8-17x. These tests pin
the replacement to the force model and to the Kit-side copy.
"""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "source" / "extensions" / "omni.marble.coaster"
                    / "omni" / "marble" / "coaster"))

from rlc_circuit import (  # noqa: E402
    MU_0_MM,
    L_effective,
    axial_field_per_amp,
    compute_rlc_params,
    dL_dx,
    marble_dL_H,
    saturated_force,
)

# The rig's coil as wound: 30 loops at R = 15 mm over 30 mm; 12.7 mm ball.
RIG = {
    "num_turns": 30,
    "inner_radius_mm": 12.0,
    "outer_radius_mm": 18.0,
    "length_mm": 30.0,
    "marble_radius_mm": 6.35,
    "chi_eff": 3.0,
}


def test_on_axis_field_matches_the_field_module():
    from analytical_bfield import solenoid_field
    params = dict(RIG, current_A=1.0)
    for z in (-30.0, -13.78, 0.0, 9.0):
        _, bz = solenoid_field(0.0, z, params)
        assert axial_field_per_amp(z, 30, 15.0, 30.0) == pytest.approx(bz, rel=1e-9)


def test_dL_at_centre_is_the_dipole_value():
    # COIL_AS_SENSOR.md section 1 quotes 2.02 uH at centre from a slightly
    # different loop placement; with the field module's placement (30 loops,
    # endpoints included) it is 1.95 uH. The dipole form is +/-30% anyway;
    # what is pinned here is that the value is the dipole one, not 0.12 uH.
    assert marble_dL_H(0.0, RIG) * 1e6 == pytest.approx(1.95, abs=0.06)
    # ~0.92 uH at the fire point, slope peaking near -12 mm
    assert marble_dL_H(-13.78, RIG) * 1e6 == pytest.approx(0.92, abs=0.06)
    slopes = {z: dL_dx(z, RIG, None) for z in (-16.0, -14.0, -12.0, -10.0, -8.0)}
    assert max(slopes, key=slopes.get) == -12.0


def test_force_is_half_I_squared_dL_dx():
    """F = 1/2 I^2 dL/dx must hold to numerical precision below saturation."""
    I = 100.0
    marble = {"chi_eff": 3.0, "volume_mm3": (4 / 3) * math.pi * 6.35 ** 3,
              "saturation_T": 1.8}
    for z in (-25.0, -13.78, -5.0, 4.0):
        b = axial_field_per_amp(z, 30, 15.0, 30.0) * I
        dbdz = (axial_field_per_amp(z + 0.05, 30, 15.0, 30.0)
                - axial_field_per_amp(z - 0.05, 30, 15.0, 30.0)) / 0.1 * I
        f_mN = saturated_force(b, dbdz, marble)
        # dL/dx in H/mm -> H/m is x1e3; F[N] = 1/2 I^2 dL/dx[H/m]; mN = x1e3
        f_from_dL_mN = 0.5 * I * I * dL_dx(z, RIG, None, dx=0.05) * 1e3 * 1e3
        assert f_mN == pytest.approx(f_from_dL_mN, rel=2e-3)


def test_far_away_the_marble_adds_nothing():
    rlc = compute_rlc_params({"capacitance_uF": 5350.0, "charge_voltage_V": 49.0,
                              "inductance_uH": 17.9, "total_resistance_ohm": 0.141})
    assert L_effective(-200.0, RIG, rlc) == pytest.approx(rlc["inductance_H"], rel=1e-4)
    assert L_effective(0.0, RIG, rlc) > rlc["inductance_H"] * 1.05


def test_kit_side_copy_agrees():
    from coil_physics import CoilPhysics
    cp = CoilPhysics(inner_radius=12.0, outer_radius=18.0, length=30.0,
                     num_turns=30, marble_radius=6.35, chi_eff=3.0)
    for z in (-20.0, -13.78, 0.0, 7.0):
        assert cp.marble_inductance_H(z) == pytest.approx(marble_dL_H(z, RIG), rel=1e-9)
        assert cp.marble_dL_dx_H_per_mm(z) == pytest.approx(dL_dx(z, RIG, None), rel=1e-6)
