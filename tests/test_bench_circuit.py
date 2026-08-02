"""The circuit model against values MEASURED on the as-built vbench rig.

Reference table: omnimarble-vbench/docs/NEXT_SESSION.md section 7 ("Known-good
reference values"), measured 2026-07-26. Its own framing is the reason these are
tests rather than notes:

    "If a fresh measurement disagrees, something changed -- don't just accept
     the new number."

The same applies in this direction. These pin the twin's circuit model against
hardware, so a refactor that quietly changes the RLC solver is caught here
rather than in a sweep six steps later.

Note the C-sweep crosses zeta = 1 between one and two cans, so it exercises all
three damping regimes. The overdamped branch used to report a hardcoded
peak_current = 0, which would have made four of the five sweep points read
0 A -- that regression is covered by test_every_sweep_point_has_a_real_peak.
"""

import math

import pytest

from rlc_circuit import compute_rlc_params

# --- measured on the rig -----------------------------------------------------
COIL_L_UH = 17.9          # 4-wire LCR @10 kHz (Q ~ 8)
BANK_UNIT_UF = 1909.0     # @100 Hz; 13% under the 2200 label, inside tolerance
BANK_UNIT_ESR_MOHM = 48.0 # @100 Hz, parallels as 1/n
LOOP_R_1CAN_OHM = 0.164   # on-time sweep fit; DC injection gave 0.161

# Loop resistance excluding the bank's own ESR contribution.
LOOP_R_FIXED_OHM = LOOP_R_1CAN_OHM - BANK_UNIT_ESR_MOHM / 1000.0


def loop_resistance_ohm(cans):
    """Bank ESR parallels as 1/n, so loop R FALLS as cans are added.

    vbench, NEXT_SESSION.md section 5: "A model treating R as fixed will
    attribute that to capacitance and appear to validate for the wrong reason."
    """
    return LOOP_R_FIXED_OHM + (BANK_UNIT_ESR_MOHM / 1000.0) / cans


def bank(cans, voltage):
    return compute_rlc_params({
        "capacitance_uF": BANK_UNIT_UF * cans,
        "charge_voltage_V": voltage,
        "inductance_uH": COIL_L_UH,
        "total_resistance_ohm": loop_resistance_ohm(cans),
    })


def test_reproduces_measured_single_can_pulse():
    """The headline check: 1 can at 9.5 V gives 40 A at 195 us on the bench."""
    rlc = bank(cans=1, voltage=9.5)
    assert rlc["regime"] == "underdamped"
    assert rlc["zeta"] == pytest.approx(0.85, abs=0.01)
    assert rlc["time_to_peak_s"] * 1e6 == pytest.approx(195.0, abs=2.0)
    assert rlc["peak_current_A"] == pytest.approx(40.0, abs=1.0)


def test_reproduces_measured_ringing_frequency():
    """f0 = 861 Hz at one can, 385 Hz at five (NEXT_SESSION.md section 7)."""
    for cans, expected_hz in ((1, 861.0), (5, 385.0)):
        f0 = bank(cans, voltage=55.0)["omega_0"] / (2 * math.pi)
        assert f0 == pytest.approx(expected_hz, rel=0.01)


def test_loop_resistance_falls_as_cans_are_added():
    """0.164 ohm at 1 can -> 0.126 ohm at 5 (NEXT_SESSION.md section 5)."""
    assert loop_resistance_ohm(1) == pytest.approx(0.164, abs=0.001)
    assert loop_resistance_ohm(5) == pytest.approx(0.126, abs=0.001)
    values = [loop_resistance_ohm(n) for n in range(1, 6)]
    assert values == sorted(values, reverse=True), "R must fall monotonically"


def test_damping_crosses_critical_inside_the_sweep():
    """zeta goes 0.85 -> 1.45 across the bank, crossing 1.0 between 1 and 2 cans.

    The crossing is the shape a correct model predicts and a fixed-R model
    misses, so it is the sweep's most diagnostic feature.
    """
    zetas = [bank(n, 55.0)["zeta"] for n in range(1, 6)]
    assert zetas == sorted(zetas), "zeta must rise monotonically with capacitance"
    assert zetas[0] == pytest.approx(0.85, abs=0.02)
    assert zetas[-1] == pytest.approx(1.45, abs=0.05)
    assert zetas[0] < 1.0 < zetas[1], "the crossing should sit between 1 and 2 cans"


@pytest.mark.parametrize("cans", [1, 2, 3, 4, 5])
def test_every_sweep_point_has_a_real_peak(cans):
    """No sweep point may report I_peak = 0 or t_peak = 0.

    Regression guard: the overdamped branch previously hardcoded both to zero,
    which silently zeroed four of the five points.
    """
    rlc = bank(cans, voltage=55.0)
    assert rlc["peak_current_A"] > 0.0, f"{cans} cans reported no peak current"
    assert rlc["time_to_peak_s"] > 0.0, f"{cans} cans reported no time-to-peak"


def test_peak_current_stays_within_the_switch_rating():
    """The board is designed to ~500 A; the measured circuit must stay under."""
    for cans in range(1, 6):
        assert bank(cans, voltage=55.0)["peak_current_A"] < 500.0


def test_stored_energy_matches_the_full_bank_figure():
    """A full bank at 55 V is 14.4 J (NEXT_SESSION.md section 6)."""
    assert bank(5, 55.0)["stored_energy_J"] == pytest.approx(14.4, rel=0.02)


# -- freewheel decay ----------------------------------------------------------

from simulate_rig_shot import freewheel_current  # noqa: E402

COIL_R_OHM = 0.107     # coil + leads, 4-wire @1kHz
DIODE_VF_V = 0.84      # MBR60100 Schottky, the part actually fitted
I_PEAK_A = 211.0


def test_freewheel_uses_the_coil_resistance_not_the_loop_resistance():
    """Once the switch opens the bank is OUT of the circuit.

    Its ESR therefore stops participating and the current decays through the
    coil alone: tau = L/R_coil = 167us, not the 109us the discharge loop's
    164 mOhm would give. The tail carries a large share of the impulse, so
    reusing the loop figure understates Delta-v by ~15%.
    """
    tau_coil = COIL_L_UH * 1e-6 / COIL_R_OHM
    tau_loop = COIL_L_UH * 1e-6 / LOOP_R_1CAN_OHM
    assert tau_coil * 1e6 == pytest.approx(167.0, abs=2.0)
    assert tau_loop * 1e6 == pytest.approx(109.0, abs=2.0)
    assert tau_coil > 1.5 * tau_loop


def test_diode_drop_drives_the_current_to_exactly_zero():
    """A diode is not a resistor: its forward drop is a constant opposing term.

    L di/dt = -(I*R + Vf) reaches zero in FINITE time, where a pure exponential
    only decays asymptotically. At 211A that truncates the tail at ~557us.
    """
    t_zero = (COIL_L_UH * 1e-6 / COIL_R_OHM) * math.log(
        1 + I_PEAK_A * COIL_R_OHM / DIODE_VF_V)
    assert t_zero * 1e6 == pytest.approx(557.0, abs=15.0)

    assert freewheel_current(I_PEAK_A, t_zero * 0.9, COIL_R_OHM,
                             COIL_L_UH * 1e-6, DIODE_VF_V) > 0.0
    assert freewheel_current(I_PEAK_A, t_zero * 1.1, COIL_R_OHM,
                             COIL_L_UH * 1e-6, DIODE_VF_V) == 0.0
    # A pure exponential is still non-zero well past that point.
    assert freewheel_current(I_PEAK_A, t_zero * 1.1, COIL_R_OHM,
                             COIL_L_UH * 1e-6, 0.0) > 0.0


def test_freewheel_is_monotonic_and_never_negative():
    L = COIL_L_UH * 1e-6
    previous = I_PEAK_A + 1.0
    for i in range(200):
        value = freewheel_current(I_PEAK_A, i * 5e-6, COIL_R_OHM, L, DIODE_VF_V)
        assert 0.0 <= value <= previous
        previous = value


def test_freewheel_starts_at_the_cut_current():
    assert freewheel_current(I_PEAK_A, 0.0, COIL_R_OHM, COIL_L_UH * 1e-6,
                             DIODE_VF_V) == pytest.approx(I_PEAK_A)


def test_the_tail_carries_a_large_share_of_the_impulse():
    """Why the resistance choice matters so much.

    Delta-v goes as the integral of I^2 dt, and the freewheel tail contributes
    a comparable amount to the 191us gate window itself -- so a wrong tail is a
    wrong Delta-v, not a rounding error.
    """
    L = COIL_L_UH * 1e-6
    steps, span = 20000, 1.5e-3
    dt = span / steps

    def tail(resistance, vf):
        return sum(freewheel_current(I_PEAK_A, i * dt, resistance, L, vf) ** 2
                   for i in range(steps)) * dt

    correct = tail(COIL_R_OHM, DIODE_VF_V)
    wrong = tail(LOOP_R_1CAN_OHM, 0.0)
    gate = I_PEAK_A ** 2 * 191e-6          # order-of-magnitude gate contribution

    assert correct > 0.3 * gate, "the tail is not negligible against the gate"
    assert correct > 1.3 * wrong, (
        "using the loop resistance materially understates the tail")
