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


# The sphere's magnetisation is M = min(chi_eff*H, M_sat), so it saturates when
# chi_eff*|B| reaches B_sat -- i.e. at |B| = 1.8/3 = 0.6 T for chi_eff = 3. The
# old code switched at a different criterion (an internal field of (1+chi/3)|B|)
# than the one where the two branches meet, which is what made the force jump.
B_SAT_ONSET_T = MARBLE["saturation_T"] / MARBLE["chi_eff"]


def test_saturated_force_linear_regime():
    """Below onset the force is linear in B."""
    F1 = saturated_force(0.2, 0.01, MARBLE)
    F2 = saturated_force(0.4, 0.01, MARBLE)
    assert 0.4 < B_SAT_ONSET_T, "both samples must be below saturation onset"
    assert F2 == pytest.approx(2 * F1, rel=1e-9)
    expected = (MARBLE["chi_eff"] * MARBLE["volume_mm3"] / MU_0_MM) * 0.2 * 0.01
    assert F1 == pytest.approx(expected, rel=1e-12)


def test_saturated_force_saturated_regime():
    """Above onset the force is capped at M_sat * V * dB/dz.

    Regression: this used to compute M_sat = B_sat/MU_0_MM and then multiply by
    MU_0_MM again, cancelling to B_sat*V*dB/dz -- low by 1/mu_0, a factor ~796.
    """
    F_sat = saturated_force(2.0, 0.01, MARBLE)
    expected = (MARBLE["saturation_T"] / MU_0_MM) * MARBLE["volume_mm3"] * 0.01
    assert F_sat == pytest.approx(expected, rel=1e-12)
    # Doubling B in saturation changes nothing
    assert saturated_force(4.0, 0.01, MARBLE) == pytest.approx(F_sat, rel=1e-12)
    # Saturated force is below the (unphysical) linear extrapolation
    linear = (MARBLE["chi_eff"] * MARBLE["volume_mm3"] / MU_0_MM) * 2.0 * 0.01
    assert F_sat < linear


def test_saturated_force_is_continuous_at_onset():
    """No jump at the threshold -- the branches must meet.

    The previous implementation dropped ~800x here, which is the kind of
    discontinuity that produces a plausible-looking trajectory with a
    physically impossible kink in it.
    """
    below = saturated_force(B_SAT_ONSET_T * 0.999, 0.01, MARBLE)
    above = saturated_force(B_SAT_ONSET_T * 1.001, 0.01, MARBLE)
    assert above == pytest.approx(below, rel=2e-3)


def test_saturated_force_sign_follows_the_field():
    for dBdz in (0.01, -0.01):
        assert saturated_force(-2.0, dBdz, MARBLE) == pytest.approx(
            -saturated_force(2.0, dBdz, MARBLE), rel=1e-12)


def test_eddy_braking_opposes_motion():
    """Now takes dB/dz (T/mm) rather than dB/dt (T/s) -- see below for why."""
    F_fwd = eddy_braking_force(0.01, +1000.0, MARBLE)
    F_bwd = eddy_braking_force(0.01, -1000.0, MARBLE)
    assert F_fwd < 0  # opposes positive velocity
    assert F_bwd > 0  # opposes negative velocity
    assert F_fwd == pytest.approx(-F_bwd, rel=1e-12)
    assert eddy_braking_force(0.0, 1000.0, MARBLE) == 0.0
    assert eddy_braking_force(0.01, 0.0, MARBLE) == 0.0
    # Quadratic in the gradient
    assert eddy_braking_force(0.02, 1000.0, MARBLE) == pytest.approx(
        4 * F_fwd, rel=1e-9,
    )


def test_eddy_braking_is_proportional_to_velocity():
    """Any drag law must scale with speed.

    The previous form depended only on the SIGN of velocity -- it took
    (dB/dt)^2 and used sign(v) for direction -- which is not a drag law at all.
    """
    base = eddy_braking_force(0.01, 1000.0, MARBLE)
    assert eddy_braking_force(0.01, 2000.0, MARBLE) == pytest.approx(
        2 * base, rel=1e-9)
    assert eddy_braking_force(0.01, 500.0, MARBLE) == pytest.approx(
        0.5 * base, rel=1e-9)


def test_eddy_braking_is_step_size_independent():
    """The whole point of taking dB/dz instead of a differenced dB/dt.

    Callers used to form dB/dt as a backward difference over the physics step,
    so the term scaled as 1/dt^2 -- a step-size artefact. It looked harmless
    only because the 3-D mirror steps at 2ms and smeared the difference down;
    at the 1-D model's 2us it came to ~20 N against a ~1.5 N drive and would
    have annihilated the shot.

    Here the same physical situation is expressed at three step sizes; the
    force must not move.
    """
    v_mm_s, dBdz = 1000.0, 0.01
    reference = eddy_braking_force(dBdz, v_mm_s, MARBLE)
    for dt in (2e-6, 2e-4, 2e-3):
        # A caller differencing B over dt would see dB/dt = v*dB/dz; the
        # gradient it passes is unchanged by dt.
        assert eddy_braking_force(dBdz, v_mm_s, MARBLE) == reference, (
            f"eddy drag changed with a {dt}s step")


def test_eddy_braking_is_negligible_at_the_rig_operating_point():
    """~0.5 mN against a ~500 mN drive: 0.1%, which is the honest answer.

    Worth pinning as a number rather than a comment, because the previous
    version of this term was ~13x the entire drive.
    """
    rig_marble = {
        "conductivity_S_per_m": 6e6,
        "radius_mm": 6.35,
        "volume_mm3": (4 / 3) * math.pi * 6.35 ** 3,
    }
    # 1 m/s through the rig's ~6.3 T/m axial gradient.
    force_mN = abs(eddy_braking_force(6.3e-3, 1000.0, rig_marble))
    assert 0.1 < force_mN < 2.0, f"expected ~0.5 mN, got {force_mN:.3f}"


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


def test_overdamped_peak_current():
    """Overdamped analytic peak matches the numeric waveform maximum.

    Regression for the ln(s1/s2) sign bug that zeroed I_peak for every
    overdamped circuit (e.g. the 4700uF/60V SELV driver-board bank).
    """
    rlc = compute_rlc_params({
        "capacitance_uF": 4700.0,
        "charge_voltage_V": 60.0,
        "inductance_uH": 12.4,
        "total_resistance_ohm": 0.11,
    })
    assert rlc["regime"] == "overdamped"
    assert rlc["time_to_peak_s"] > 0

    # Numeric maximum over the pulse
    n = 20000
    t_end = rlc["effective_pulse_duration_s"]
    I_num_max, t_num_max = max(
        (rlc_current(i * t_end / n, rlc), i * t_end / n) for i in range(n)
    )
    assert I_num_max > 100.0  # sanity: hundreds of amps, not zero
    assert rlc["peak_current_A"] == pytest.approx(I_num_max, rel=0.01)
    assert rlc["time_to_peak_s"] == pytest.approx(t_num_max, rel=0.02)
    # I(t_peak) from the closed form equals the reported peak
    assert rlc_current(rlc["time_to_peak_s"], rlc) == pytest.approx(
        rlc["peak_current_A"], rel=1e-9,
    )


def test_rlc_stored_energy_and_regime(rlc):
    assert rlc["regime"] == "underdamped"
    assert rlc["stored_energy_J"] == pytest.approx(0.5 * 470e-6 * 50 ** 2)
    # I(t) never negative (flyback clamp in rlc_current)
    for i in range(300):
        assert rlc_current(i * 1e-4, rlc) >= 0.0
