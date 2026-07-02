"""Unit tests for the Kit extension's pure-physics core (coil_physics.py).

Also cross-checks against scripts/rlc_circuit.py, which implements the same
physics for the headless pipeline (the duplication is tracked tech debt —
these tests are the guard that the two stay in agreement).
"""

import json
import math
from pathlib import Path

import pytest

import coil_physics
from coil_physics import CoilPhysics, gate_crossed
import rlc_circuit

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config" / "coil_params.json").read_text())


@pytest.fixture
def cp():
    """CoilPhysics built from the checked-in config."""
    return CoilPhysics(
        inner_radius=CONFIG["inner_radius_mm"],
        outer_radius=CONFIG["outer_radius_mm"],
        length=CONFIG["length_mm"],
        num_turns=CONFIG["num_turns"],
        wire_diameter=CONFIG["wire_diameter_mm"],
        insulation_thickness=CONFIG["insulation_thickness_mm"],
        capacitance_uF=CONFIG["capacitance_uF"],
        charge_voltage=CONFIG["charge_voltage_V"],
        esr=CONFIG["esr_ohm"],
        wiring_resistance=CONFIG["wiring_resistance_ohm"],
        has_flyback_diode=CONFIG["has_flyback_diode"],
        gates=CONFIG["gate_positions"],
    )


def test_derived_values_match_config(cp):
    """R_dc and L match the values persisted in config/coil_params.json."""
    assert cp.R_dc == pytest.approx(CONFIG["resistance_ohm"], rel=1e-9)
    assert cp.inductance_uH == pytest.approx(CONFIG["inductance_uH"], rel=1e-9)


def test_rlc_regime(cp):
    assert cp.regime == "underdamped"
    assert cp.zeta == pytest.approx(0.34, abs=0.02)
    assert cp.stored_energy == pytest.approx(
        0.5 * CONFIG["capacitance_uF"] * 1e-6 * CONFIG["charge_voltage_V"] ** 2
    )
    assert cp.R_total == pytest.approx(
        cp.R_dc + CONFIG["esr_ohm"] + CONFIG["wiring_resistance_ohm"]
    )


def test_rlc_current_peak_and_clamp(cp):
    # Analytic time of the damped-sine peak
    t_peak = math.atan2(cp.omega_d, cp.alpha) / cp.omega_d
    I_peak_true = cp.rlc_current(t_peak)
    # cp.peak_current is the undamped amplitude V/(omega_d*L); the true damped
    # peak is smaller by exp(-alpha*t_peak)*sin(omega_d*t_peak)
    expected = cp.peak_current * math.exp(-cp.alpha * t_peak) * math.sin(cp.omega_d * t_peak)
    assert I_peak_true == pytest.approx(expected, rel=1e-12)
    assert 0 < I_peak_true < cp.peak_current

    # Local maximum: neighbors are lower
    assert cp.rlc_current(t_peak * 0.9) < I_peak_true
    assert cp.rlc_current(t_peak * 1.1) < I_peak_true

    # Flyback diode: never negative over several oscillation periods
    period = 2 * math.pi / cp.omega_d
    for i in range(500):
        assert cp.rlc_current(i * period / 100) >= 0.0

    # Before trigger: zero
    assert cp.rlc_current(-1e-3) == 0.0


def test_recompute_derived_propagates_overrides(cp):
    L_before = cp.inductance_uH
    I_peak_before = cp.peak_current
    cp.num_turns = 60
    cp.charge_voltage = 300.0
    cp.recompute_derived()
    # More turns -> higher inductance (Wheeler ~ N^2 with weak denominator change)
    assert cp.inductance_uH > L_before
    assert cp.stored_energy == pytest.approx(0.5 * cp.capacitance_uF * 1e-6 * 300.0 ** 2)
    assert cp.peak_current != I_peak_before


def test_gate_crossing():
    assert gate_crossed(-21.0, -19.5, -20.0)      # forward crossing
    assert gate_crossed(-19.5, -21.0, -20.0)      # backward crossing
    assert gate_crossed(-21.0, -20.0, -20.0)      # exact landing counts
    assert not gate_crossed(-25.0, -22.0, -20.0)  # approaching, not crossed
    assert not gate_crossed(-19.0, -18.0, -20.0)  # already past


def test_cross_check_against_rlc_circuit(cp):
    """coil_physics and scripts/rlc_circuit.py must agree on shared physics."""
    winding_params = {
        "inner_radius_mm": CONFIG["inner_radius_mm"],
        "outer_radius_mm": CONFIG["outer_radius_mm"],
        "length_mm": CONFIG["length_mm"],
        "num_turns": CONFIG["num_turns"],
        "wire_diameter_mm": CONFIG["wire_diameter_mm"],
        "insulation_thickness_mm": CONFIG["insulation_thickness_mm"],
    }
    geom = rlc_circuit.compute_winding_geometry(winding_params)
    assert cp.wire_length_mm == pytest.approx(geom["wire_length_mm"], rel=1e-9)
    assert cp.wire_mass_g == pytest.approx(geom["wire_mass_g"], rel=1e-9)
    assert cp.num_layers == geom["num_layers"]
    assert cp.mean_radius == pytest.approx(geom["mean_radius_mm"], rel=1e-9)

    R_dc = rlc_circuit.compute_dc_resistance(
        geom["wire_length_mm"], geom["wire_cross_section_mm2"]
    )
    assert cp.R_dc == pytest.approx(R_dc, rel=1e-9)

    L_uH = rlc_circuit.compute_multilayer_inductance(
        CONFIG["num_turns"], geom["mean_radius_mm"],
        CONFIG["length_mm"], geom["winding_depth_mm"],
    )
    assert cp.inductance_uH == pytest.approx(L_uH, rel=1e-9)

    rlc = rlc_circuit.compute_rlc_params({
        "capacitance_uF": CONFIG["capacitance_uF"],
        "charge_voltage_V": CONFIG["charge_voltage_V"],
        "inductance_uH": cp.inductance_uH,
        "total_resistance_ohm": cp.R_total,
    })
    assert cp.alpha == pytest.approx(rlc["alpha"], rel=1e-12)
    assert cp.omega_0 == pytest.approx(rlc["omega_0"], rel=1e-12)
    assert cp.zeta == pytest.approx(rlc["zeta"], rel=1e-12)
    assert cp.regime == rlc["regime"]
    # rlc_circuit reports the true damped peak; coil_physics.rlc_current at
    # the same instant must match it
    assert cp.rlc_current(rlc["time_to_peak_s"]) == pytest.approx(
        rlc["peak_current_A"], rel=1e-9,
    )
    # Both closed-form currents identical across the pulse
    for i in range(50):
        t = i * rlc["zero_crossing_s"] / 49
        assert cp.rlc_current(t) == pytest.approx(
            rlc_circuit.rlc_current(t, rlc), abs=1e-9,
        )


def test_marble_mass():
    cp = CoilPhysics()
    # 10mm steel sphere: (4/3)*pi*5^3 mm^3 * 7.8e-3 g/mm^3 ~ 4.08 g
    assert cp.marble_mass_kg == pytest.approx(0.00408, rel=0.01)
