"""Tests for the ported rig velocity estimator.

The headline test is `test_replays_recorded_hardware_rolls`: the fixtures in
tests/fixtures/ are real per-channel timestamps captured on the bench
(vbench logs/rolls_*.csv, 20 passes of a 12.7mm steel ball through the IR
array), together with the velocity and residual the rig's own firmware computed
from them. Replaying them through this port and getting the same numbers is a
hardware-derived oracle -- it proves the sim and the rig share one estimator,
which is what makes a sim-vs-real Delta-v gap attributable to physics.

The other tests pin the behaviours that are deliberately quirky, so a future
"cleanup" that silently improves on the rig gets caught.
"""

import csv
import math
from pathlib import Path

import pytest

from coil_sensing import (
    REQUIRED_CHANNELS,
    RESID_WARN_US,
    VirtualStation,
    crossed,
    fit_is_suspect,
    interpolate_crossing_us,
    transit_us_to,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
# The pitch the rig's firmware used when these logs were written.
LOG_PITCH_MM = 22.14


def _load_rolls(name):
    with open(FIXTURES / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _station_from_row(row, pitch_mm=LOG_PITCH_MM):
    """Replay one recorded pass into a station, as the firmware captured it."""
    station = VirtualStation("A", list(range(5)), pitch_mm)
    for i in range(5):
        value = row.get(f"lead{i}_us", "")
        if value not in ("", "None", None):
            station.record(i, float(value))
    return station


@pytest.mark.parametrize("log_name", ["rolls_3v3.csv", "rolls_5v.csv"])
def test_replays_recorded_hardware_rolls(log_name):
    """Reproduce the rig's own v_mps and resid_us from real captured timestamps."""
    rows = _load_rolls(log_name)
    assert rows, f"{log_name} fixture is empty"

    replayed = 0
    for row in rows:
        station = _station_from_row(row)
        assert station.n_captured() == int(float(row["n_ch"]))

        logged_v = row.get("v_mps", "")
        if logged_v in ("", "None", None):
            # A genuinely failed pass: fewer than 2 channels fired, so the rig
            # reported no velocity either. rolls_5v.csv has several -- that
            # dropout is why the array was requalified at 3.3V.
            assert station.n_captured() < 2
            assert station.velocity_mps() is None
            continue

        assert station.velocity_mps() == pytest.approx(float(logged_v), abs=5e-5), (
            f"{log_name} pass {row['pass_id']}: velocity diverged from the rig's fit")

        logged_resid = row.get("resid_us", "")
        if logged_resid not in ("", "None", None):
            assert station.residual_us() == pytest.approx(
                float(logged_resid), rel=1e-4), (
                f"{log_name} pass {row['pass_id']}: residual diverged")
        replayed += 1

    assert replayed, f"{log_name} contributed no usable passes"


def test_velocity_scales_linearly_with_pitch():
    """Every velocity and every Delta-v scales linearly with the pitch constant.

    This is why the 11.0 placeholder mattered so much, and why compare_sim_real
    must hard-fail on shot rows still carrying it. Note the correction factor is
    22.14/11.0 = 2.0127, not exactly 2 -- vbench's "wrong by exactly 2x" is
    round-numbers shorthand.
    """
    row = _load_rolls("rolls_3v3.csv")[0]
    at_correct = _station_from_row(row, pitch_mm=22.14).velocity_mps()
    at_placeholder = _station_from_row(row, pitch_mm=11.0).velocity_mps()
    assert at_correct / at_placeholder == pytest.approx(22.14 / 11.0, rel=1e-12)


def test_constant_velocity_is_recovered_exactly():
    station = VirtualStation("A", list(range(5)), 22.14)
    v_mps = 1.006  # the rig's ideal release speed
    for i in range(5):
        station.record(i, i * (22.14 / 1000.0) / v_mps * 1e6)
    assert station.velocity_mps() == pytest.approx(v_mps, rel=1e-12)
    # A perfect line has no residual.
    assert station.residual_us() == pytest.approx(0.0, abs=1e-6)


def test_direction_does_not_change_the_magnitude():
    """velocity_mps() returns a magnitude, so a wrong order flag still reads right."""
    forward = VirtualStation("A", list(range(5)), 22.14)
    reverse = VirtualStation("B", list(range(5)), 22.14)
    for i in range(5):
        forward.record(i, i * 20000.0)
        reverse.record(i, (4 - i) * 20000.0)
    assert forward.velocity_mps() == pytest.approx(reverse.velocity_mps(), rel=1e-12)


def test_rev_reverses_channels_once():
    station = VirtualStation("A", [-146.34, -124.2, -102.06, -79.92, -57.78],
                             22.14, rev=True)
    assert station.channel_x_mm[0] == pytest.approx(-57.78)
    assert station.channel_x_mm[-1] == pytest.approx(-146.34)


def test_last_channel_is_by_time_not_index():
    """The channel nearest the coil is the last one crossed, whichever index it is."""
    station = VirtualStation("A", [-146.34, -124.2, -102.06, -79.92, -57.78], 22.14)
    for i in range(5):
        station.record(i, (4 - i) * 20000.0)  # crossed in descending index order
    assert station.last_channel_x_mm() == pytest.approx(-146.34)
    assert station.last_tick() == pytest.approx(80000.0)


def test_too_few_channels_yield_no_measurement():
    station = VirtualStation("A", list(range(5)), 22.14)
    assert station.velocity_mps() is None
    station.record(0, 0.0)
    assert station.velocity_mps() is None          # 1 channel: no line
    station.record(1, 20000.0)
    assert station.velocity_mps() is not None      # 2 channels: a line
    assert station.residual_us() is None           # ...but no meaningful residual
    station.record(2, 40000.0)
    assert station.residual_us() is not None


def test_incomplete_capture_is_visible_to_the_caller():
    """The rig aborts below 5 channels; the sim must be able to see that."""
    station = VirtualStation("A", list(range(5)), 22.14)
    for i in range(4):
        station.record(i, i * 20000.0)
    assert not station.complete()
    assert station.n_captured() < REQUIRED_CHANNELS


def test_first_edge_wins():
    station = VirtualStation("A", list(range(5)), 22.14)
    station.record(0, 1000.0)
    station.record(0, 9999.0)
    assert station._ts[0] == pytest.approx(1000.0)


def test_accelerating_pass_inflates_the_residual():
    """A decelerating hand-roll is exactly what RESID_WARN_US is meant to catch."""
    station = VirtualStation("A", list(range(5)), 22.14)
    t, v = 0.0, 1.0
    for i in range(5):
        station.record(i, t)
        t += (22.14 / 1000.0) / v * 1e6
        v *= 0.75
    assert fit_is_suspect(station.residual_us())

    # The recorded hand rolls are all in this regime -- they were rolled by hand
    # down a tilted test segment, so they are accelerating, not constant.
    for row in _load_rolls("rolls_3v3.csv"):
        assert float(row["resid_us"]) > RESID_WARN_US


def test_interpolated_crossing_beats_step_quantisation():
    """500Hz steps would quantise timestamps to ~2mm at 1m/s; interpolation must not."""
    dt_us = 2000.0                      # one 500 Hz physics step
    v_mm_us = 1.006 / 1000.0            # 1.006 m/s in mm/us
    target = -102.06
    # A step that straddles the channel, with the crossing 30% through it.
    prev_x = target - 0.3 * v_mm_us * dt_us
    x = prev_x + v_mm_us * dt_us
    t_us = 500000.0
    crossing = interpolate_crossing_us(prev_x, x, target, t_us, dt_us)
    assert crossing == pytest.approx(t_us - dt_us + 0.3 * dt_us, rel=1e-9)
    # Sampling the step time instead would be off by most of a step.
    assert abs(crossing - t_us) > 0.5 * dt_us


def test_interpolation_rejects_segments_that_do_not_span():
    assert interpolate_crossing_us(0.0, 1.0, 5.0, 100.0, 10.0) is None
    assert interpolate_crossing_us(1.0, 1.0, 1.0, 100.0, 10.0) is None


def test_crossed_detects_both_directions():
    assert crossed(-1.0, 1.0, 0.0)
    assert crossed(1.0, -1.0, 0.0)
    assert not crossed(1.0, 2.0, 0.0)


def test_transit_matches_the_rig_arithmetic():
    """35mm at 1m/s is the rig's ~35ms compute-and-arm budget."""
    assert transit_us_to(35.0, 1.0) == pytest.approx(35000.0, rel=1e-12)
    assert transit_us_to(35.0, 0.0) is None


def test_estimator_is_unbiased_on_a_flat_pass():
    """On the rig's flat zone the fit should recover truth, not merely be close.

    This is the property the flat measurement zone exists to guarantee
    (track/RIG.md "Profile"), so it is worth asserting rather than assuming.
    """
    for v_mps in (0.4, 1.006, 2.5):
        station = VirtualStation("A", list(range(5)), 22.14)
        for i in range(5):
            station.record(i, i * (22.14 / 1000.0) / v_mps * 1e6)
        assert math.isclose(station.velocity_mps(), v_mps, rel_tol=1e-12)
