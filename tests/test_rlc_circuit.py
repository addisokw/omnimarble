"""Unit tests for scripts/rlc_circuit.py helpers and the mm/mN/T unit system."""

import math

import pytest

from rlc_circuit import (
    MU_0_MM,
    MU_0_SI,
    compute_rlc_params,
    eddy_braking_force,
    resistance_at_temperature,
    rlc_current,
    rlc_current_with_cutoff,
    saturated_force,
)

MARBLE = {
    "chi_eff": 3.0,
    "volume_mm3": (4 / 3) * math.pi * 5.0 ** 3,
    "saturation_T": 1.8,
    "conductivity_S_per_m": 6e6,
    "radius_mm": 5.0,
}


@pytest.fixture
def rlc():
    return compute_rlc_params({
        "capacitance_uF": 470.0,
        "charge_voltage_V": 50.0,
        "inductance_uH": 12.4,
        "total_resistance_ohm": 0.11,
    })


def test_saturated_force_linear_regime():
    # chi_eff=3: B_internal = 2*B, so B=0.5 -> internal 1.0 < 1.8 (linear)
    F1 = saturated_force(0.4, 0.01, MARBLE)
    F2 = saturated_force(0.8, 0.01, MARBLE)
    assert F2 == pytest.approx(2 * F1, rel=1e-9)  # linear in B below saturation
    expected = (MARBLE["chi_eff"] * MARBLE["volume_mm3"] / MU_0_MM) * 0.4 * 0.01
    assert F1 == pytest.approx(expected, rel=1e-12)


def test_saturated_force_saturated_regime():
    # B=2.0 -> internal 4.0 > 1.8: force capped at M_sat * V * dBdz
    F_sat = saturated_force(2.0, 0.01, MARBLE)
    expected = (MARBLE["saturation_T"] / MU_0_MM) * MARBLE["volume_mm3"] * 0.01 * MU_0_MM
    assert F_sat == pytest.approx(expected, rel=1e-12)
    # Doubling B in saturation changes nothing
    assert saturated_force(4.0, 0.01, MARBLE) == pytest.approx(F_sat, rel=1e-12)
    # Saturated force is below the (unphysical) linear extrapolation
    linear = (MARBLE["chi_eff"] * MARBLE["volume_mm3"] / MU_0_MM) * 2.0 * 0.01
    assert F_sat < linear


def test_eddy_braking_opposes_motion():
    F_fwd = eddy_braking_force(100.0, +1000.0, MARBLE)
    F_bwd = eddy_braking_force(100.0, -1000.0, MARBLE)
    assert F_fwd < 0  # opposes positive velocity
    assert F_bwd > 0  # opposes negative velocity
    assert F_fwd == pytest.approx(-F_bwd, rel=1e-12)
    assert eddy_braking_force(0.0, 1000.0, MARBLE) == 0.0
    assert eddy_braking_force(100.0, 0.0, MARBLE) == 0.0
    # Quadratic in dB/dt
    assert eddy_braking_force(200.0, 1000.0, MARBLE) == pytest.approx(
        4 * F_fwd, rel=1e-9,
    )


def test_resistance_temperature_slope():
    R20 = 0.08
    assert resistance_at_temperature(R20, 20.0) == pytest.approx(R20)
    R120 = resistance_at_temperature(R20, 120.0)
    assert R120 == pytest.approx(R20 * (1 + 0.00393 * 100), rel=1e-9)
    assert R120 > R20


def test_cutoff_exponential_decay(rlc):
    t_cut = rlc["time_to_peak_s"]
    I_cut = rlc_current(t_cut, rlc)
    R = rlc["total_resistance_ohm"]
    L = rlc["inductance_H"]
    for dt_after in (1e-5, 1e-4, 1e-3):
        I = rlc_current_with_cutoff(t_cut + dt_after, t_cut, rlc)
        assert I == pytest.approx(I_cut * math.exp(-(R / L) * dt_after), rel=1e-9)
    # Before cutoff: identical to closed form
    assert rlc_current_with_cutoff(t_cut / 2, t_cut, rlc) == pytest.approx(
        rlc_current(t_cut / 2, rlc), rel=1e-12,
    )


def test_unit_system_force_prefactor():
    """The mm-scaled force must equal the SI computation converted to mN.

    mm system: F[mN] = chi * V[mm^3] / MU_0_MM * B[T] * dBdz[T/mm]
    SI system: F[N]  = chi * V[m^3]  / MU_0_SI * B[T] * dBdz[T/m]
    """
    chi = 3.0
    V_mm3 = (4 / 3) * math.pi * 5.0 ** 3
    B = 0.05          # T
    dBdz_mm = 0.002   # T/mm

    F_mN = chi * V_mm3 / MU_0_MM * B * dBdz_mm

    V_m3 = V_mm3 * 1e-9
    dBdz_m = dBdz_mm * 1e3  # T/m
    F_N = chi * V_m3 / MU_0_SI * B * dBdz_m

    assert F_mN == pytest.approx(F_N * 1e3, rel=1e-12)


def test_rlc_stored_energy_and_regime(rlc):
    assert rlc["regime"] == "underdamped"
    assert rlc["stored_energy_J"] == pytest.approx(0.5 * 470e-6 * 50 ** 2)
    # I(t) never negative (flyback clamp in rlc_current)
    for i in range(300):
        assert rlc_current(i * 1e-4, rlc) >= 0.0
