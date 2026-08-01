"""Tests for the named rig profiles and their derived circuit values.

The bank helper carries the load here. Capacitor ESR parallels as 1/n, so loop
resistance falls as cans are added and the C-sweep varies R and C together --
the failure mode vbench warns about is a model holding R fixed, which
"will attribute that to capacitance and appear to validate for the wrong
reason". These tests pin the scaling against the measured endpoints.
"""

import json
from pathlib import Path

import pytest

from rig_profile import ProfileError, RigProfile, available_profiles, load_profile

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def vbench():
    return load_profile(ROOT, "vbench_v0")


@pytest.fixture(scope="module")
def legacy():
    return load_profile(ROOT, "legacy_gates_v1")


def test_both_profiles_exist_and_legacy_is_the_default():
    """The default must stay legacy until the new reference run is captured."""
    assert set(available_profiles(ROOT)) == {"legacy_gates_v1", "vbench_v0"}
    doc = json.loads((ROOT / "config" / "rig_profile.json").read_text(encoding="utf-8"))
    assert doc["default_profile"] == "legacy_gates_v1"
    assert load_profile(ROOT).name == "legacy_gates_v1"


def test_legacy_defers_rather_than_duplicating(legacy):
    """Deferring to coil_params.json is what keeps the 300V fixture exact."""
    assert legacy.defers_to_coil_params
    assert legacy.sensing_mode == "gates"
    assert legacy.firing_mode == "position_cutoff"
    with pytest.raises(ProfileError):
        legacy.bank()


def test_vbench_describes_the_measured_rig(vbench):
    assert vbench.sensing_mode == "stations"
    assert vbench.firing_mode == "fixed_on_time"
    assert vbench.sensing["pitch_mm"] == pytest.approx(22.14)
    assert vbench.marble["diameter_mm"] == pytest.approx(12.7)
    assert vbench.circuit["measured"]["inductance_uH"] == pytest.approx(17.9)
    assert vbench.circuit["voltage_max_V"] == pytest.approx(55.0)


def test_bank_esr_parallels_and_loop_resistance_falls(vbench):
    """0.164 ohm at 1 can -> 0.126 ohm at 5, matching the bench."""
    assert vbench.bank(1)["loop_resistance_ohm"] == pytest.approx(0.164, abs=1e-9)
    assert vbench.bank(5)["loop_resistance_ohm"] == pytest.approx(0.126, abs=0.001)

    resistances = [vbench.bank(n)["loop_resistance_ohm"] for n in range(1, 6)]
    assert resistances == sorted(resistances, reverse=True)

    # ESR itself is exactly 1/n of the single-can value.
    for n in range(1, 6):
        assert vbench.bank(n)["esr_ohm"] == pytest.approx(0.048 / n, rel=1e-12)


def test_bank_capacitance_scales_with_cans(vbench):
    for n in range(1, 6):
        assert vbench.bank(n)["capacitance_uF"] == pytest.approx(1909.0 * n)
    assert len(vbench.bank_options()) == 5


def test_bank_rejects_impossible_can_counts(vbench):
    for bad in (0, 6, -1):
        with pytest.raises(ProfileError):
            vbench.bank(bad)


def test_gate_window_subtracts_the_measured_overhead(vbench):
    """The rig LOGS the clamped request but holds the gate for that minus 9us."""
    window = vbench.gate_window_us()
    assert window["on_time_us"] == pytest.approx(200.0)
    assert window["gate_us"] == pytest.approx(191.0)

    # The clamp applies to the request, and the overhead comes off afterwards.
    clamped = vbench.gate_window_us(5000.0)
    assert clamped["on_time_us"] == pytest.approx(2000.0)
    assert clamped["gate_us"] == pytest.approx(1991.0)

    # A request below the overhead cannot produce a negative window.
    assert vbench.gate_window_us(3.0)["gate_us"] == pytest.approx(0.0)


def test_stations_are_mirrored_and_clear_of_the_coil(vbench):
    specs = vbench.station_specs()
    assert set(specs) == {"A", "B"}
    assert specs["A"]["order_rev"] != specs["B"]["order_rev"]

    face_in = vbench.coil["face_in_x_mm"]
    face_out = vbench.coil["face_out_x_mm"]
    for spec in specs.values():
        assert len(spec["channel_x_mm"]) == 5
        for x in spec["channel_x_mm"]:
            assert not (face_in < x < face_out)


def test_station_a_last_channel_matches_the_fire_distance(vbench):
    """The 35mm fire distance must agree with the published channel geometry.

    Station A's channel nearest the coil is at -57.78 and the coil's entry face
    at -22.78, so the gap is exactly the 35.0mm the firmware extrapolates over.
    If these ever disagree the shot fires at the wrong place.
    """
    specs = vbench.station_specs()
    nearest = max(specs["A"]["channel_x_mm"])
    gap = vbench.coil["face_in_x_mm"] - nearest
    assert gap == pytest.approx(vbench.firing["last_channel_to_coil_mm"], abs=1e-9)


def test_channel_spacing_matches_the_declared_pitch(vbench):
    """The published positions must actually be one pitch apart."""
    pitch = vbench.sensing["pitch_mm"]
    for spec in vbench.station_specs().values():
        xs = sorted(spec["channel_x_mm"])
        gaps = [b - a for a, b in zip(xs, xs[1:])]
        for gap in gaps:
            assert gap == pytest.approx(pitch, abs=0.01)


# -- validation ---------------------------------------------------------------

def _vbench_data():
    doc = json.loads((ROOT / "config" / "rig_profile.json").read_text(encoding="utf-8"))
    return doc["profiles"]["vbench_v0"]


def test_rejects_matching_station_order_flags():
    """Both stations reading the same direction means a board is reversed."""
    data = _vbench_data()
    data["sensing"]["stations"]["B"]["order_rev"] = True
    with pytest.raises(ProfileError, match="opposite order_rev"):
        load_profile(ROOT, "vbench_v0", path=_write_tmp(data))


def test_rejects_a_channel_inside_the_coil():
    data = _vbench_data()
    data["sensing"]["stations"]["A"]["channel_x_mm"][-1] = 0.0
    with pytest.raises(ProfileError, match="inside the coil"):
        load_profile(ROOT, "vbench_v0", path=_write_tmp(data))


def test_rejects_voltage_above_the_board_invariant():
    data = _vbench_data()
    data["circuit"]["charge_voltage_V"] = 60.0
    with pytest.raises(ProfileError, match="exceeds the board invariant"):
        load_profile(ROOT, "vbench_v0", path=_write_tmp(data))


def test_rejects_unknown_profile_name():
    with pytest.raises(ProfileError, match="unknown profile"):
        load_profile(ROOT, "does_not_exist")


_TMP = []


def _write_tmp(vbench_data):
    """Write a one-profile doc to a temp file and return its path."""
    import tempfile
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8")
    json.dump({"default_profile": "vbench_v0",
               "profiles": {"vbench_v0": vbench_data}}, handle)
    handle.close()
    _TMP.append(handle.name)
    return handle.name
