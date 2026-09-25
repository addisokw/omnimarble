"""The return-leg FiringController (sustain) and the station helpers it uses.

The return leg is the forward leg's mirror at station B, travel toward -x,
timed on the LAST channel pair (v_local) plus the firmware's empirical trim
on top of the mirrored fire offset -- the rule the bench validated after the
constant-acceleration predictor was falsified (2026-09-25). These tests pin
that arithmetic, the wrong-way re-arm, the manual offset, and per-pass reuse.
"""

import pytest

from coil_sensing import FiringController, VirtualStation, return_trim_mm

FIRING = {
    "required_channels": 5,
    "last_channel_to_coil_mm": 35.0,
    "detect_halfwidth_mm": 5.2,
    "fire_offset_mm": 9.0,
    "on_time_us": 1500.0,
    "trigger_lead_us": 0.0,
    "trigger_slip_us": 2000.0,
    "capture_window_ms": 200.0,
    "trigger_timeout_ms": 3000.0,
}
RETURN = {
    "last_channel_to_coil_mm": 35.0,
    "trim_k_mm_mps": 1.7,
    "trim_gate_mm_per_us": -0.004,
    "trim_gate_ref_us": 700.0,
    "trim_max_mm": 16.0,
    "manual_offset_mm": 0.0,
}
A_CHANNELS = [-146.34, -124.2, -102.06, -79.92, -57.78]
B_CHANNELS = [57.78, 79.92, 102.06, 124.2, 146.34]


def _forward_pass(station, v_mps, t0_us=0.0, channels=5):
    """Constant velocity, index-UP in time (toward +x on either station)."""
    step = (22.14 / 1000.0) / v_mps * 1e6
    t = t0_us
    for i in range(channels):
        station.record(i, t)
        t += step
    return t - step


def _return_pass(station, v_mps, t0_us=0.0):
    """Constant velocity toward -x: index-DOWN in time."""
    step = (22.14 / 1000.0) / v_mps * 1e6
    t = t0_us
    for k in range(5):
        station.record(4 - k, t)
        t += step
    return t - step


def _ret_ctl(station, **kw):
    args = dict(leg="ret", profile_return=RETURN, on_us=1500.0)
    args.update(kw)
    return FiringController(FIRING, station, **args)


def test_return_trim_matches_the_twin_and_the_firmware_constants():
    """coil_sensing keeps a local copy of sustain_model.return_trim_mm (the
    twin imports this module, so it cannot go the other way); pin them."""
    from sustain_model import return_trim_mm as twin_trim
    for v, on in ((0.2, 1500), (0.6, 1500), (0.17, 700), (0.01, 700), (0.495, 1500)):
        assert return_trim_mm(v, on) == pytest.approx(twin_trim(v, on), abs=1e-12)
    assert return_trim_mm(0.2, 1500) == pytest.approx(5.3, abs=0.05)
    assert return_trim_mm(0.01, 700) == 16.0


def test_return_leg_fires_at_reach_over_v_local_with_the_trim():
    """A mirrored constant-velocity pass through B: reach = 35 + 5.2 + 9 + trim
    over v_local (the last channel pair), not the fit."""
    v = 0.25
    b = VirtualStation("B", B_CHANNELS, 22.14)
    last = _return_pass(b, v)
    ctl = _ret_ctl(b)
    assert ctl.update(last) == FiringController.ARMED
    assert ctl.pass_direction == -1
    assert ctl.v_in_mps == pytest.approx(v, rel=1e-9)
    assert ctl.v_local_mps == pytest.approx(v, rel=1e-9)
    trim = return_trim_mm(v, 1500.0)
    assert ctl.trim_mm == pytest.approx(trim)
    reach = 35.0 + 5.2 + 9.0 + trim
    assert ctl.reach_mm == pytest.approx(reach)
    assert ctl.fire_at_us == pytest.approx(last + reach / 1000.0 / v * 1e6, rel=1e-9)
    assert ctl.x_target_mm == pytest.approx(22.78 - 9.0 - trim)
    assert ctl.update(ctl.fire_at_us - 1.0) == FiringController.ARMED
    assert ctl.update(ctl.fire_at_us) == FiringController.FIRED


def test_forward_leg_is_unchanged_by_the_return_machinery():
    """The forward reach stays 35 + 5.2 + 9 over the FIT velocity, no trim."""
    v = 1.006
    a = VirtualStation("A", A_CHANNELS, 22.14)
    last = _forward_pass(a, v)
    ctl = FiringController(FIRING, a)
    assert ctl.update(last) == FiringController.ARMED
    assert ctl.leg == "fwd" and ctl.trim_mm == 0.0
    assert ctl.reach_mm == pytest.approx(35.0 + 5.2 + 9.0)
    assert ctl.v_transit_mps == pytest.approx(ctl.v_in_mps)
    assert ctl.fire_at_us == pytest.approx(last + 49.2 / 1000.0 / v * 1e6, rel=1e-9)
    assert ctl.x_target_mm == pytest.approx(-22.78 + 9.0)


def test_return_leg_times_on_the_local_velocity_not_the_fit():
    """A decelerating pass: the fit averages the fast outer channels, the local
    pair is slower, and the firmware times on the local pair."""
    b = VirtualStation("B", B_CHANNELS, 22.14)
    t, v = 0.0, 0.30
    for k in range(5):
        b.record(4 - k, t)
        t += (22.14 / 1000.0) / v * 1e6
        v *= 0.9
    last = b.last_tick()
    ctl = _ret_ctl(b)
    assert ctl.update(last) == FiringController.ARMED
    assert ctl.v_local_mps < ctl.v_in_mps
    assert ctl.v_transit_mps == pytest.approx(ctl.v_local_mps)
    assert ctl.fire_at_us - last == pytest.approx(
        ctl.reach_mm / 1000.0 / ctl.v_local_mps * 1e6, rel=1e-9)


def test_return_trim_shrinks_with_speed_and_gate():
    trims = {}
    for v in (0.2, 0.6):
        b = VirtualStation("B", B_CHANNELS, 22.14)
        last = _return_pass(b, v)
        ctl = _ret_ctl(b)
        ctl.update(last)
        trims[v] = ctl.trim_mm
    assert trims[0.2] > trims[0.6]
    b = VirtualStation("B", B_CHANNELS, 22.14)
    last = _return_pass(b, 0.2)
    ctl700 = _ret_ctl(b, on_us=700.0)
    ctl700.update(last)
    assert ctl700.trim_mm > trims[0.2]          # the gate term is negative beyond 700 us
    assert ctl700.trim_mm == pytest.approx(1.7 / 0.2)


def test_manual_offset_shifts_the_return_fire_point():
    v = 0.25
    delay = {}
    for off in (0.0, 4.0, -3.0):
        b = VirtualStation("B", B_CHANNELS, 22.14)
        last = _return_pass(b, v)
        ctl = _ret_ctl(b, manual_offset_mm=off)
        ctl.update(last)
        delay[off] = ctl.fire_at_us - last
        assert ctl.x_target_mm == pytest.approx(22.78 - 9.0 - ctl.trim_mm - off)
    assert delay[4.0] - delay[0.0] == pytest.approx(4.0 / 1000.0 / v * 1e6, rel=1e-9)
    assert delay[-3.0] - delay[0.0] == pytest.approx(-3.0 / 1000.0 / v * 1e6, rel=1e-9)
    # the profile's manual offset is the default
    b = VirtualStation("B", B_CHANNELS, 22.14)
    last = _return_pass(b, v)
    ctl = _ret_ctl(b, profile_return=dict(RETURN, manual_offset_mm=2.0))
    ctl.update(last)
    assert ctl.manual_offset_mm == 2.0
    assert ctl.fire_at_us - last == pytest.approx(
        delay[0.0] + 2.0 / 1000.0 / v * 1e6, rel=1e-9)


def test_wrong_way_pass_rearms_in_sustain_and_aborts_otherwise():
    """A +x pass through B while the ret leg waits: the ball is past the coil
    and receding, so never fire; sustain clears the station and keeps waiting,
    a single shot gives up."""
    b = VirtualStation("B", B_CHANNELS, 22.14)
    last = _forward_pass(b, 0.4)                # index-up == toward +x at B
    ctl = _ret_ctl(b, rearm_on_reject=True)
    assert ctl.update(last) == FiringController.REARMED
    assert "wrong-way" in ctl.reject_reason
    assert ctl.wrong_way_count == 1 and ctl.rearm_count == 1
    assert b.n_captured() == 0 and ctl.fire_at_us is None
    assert ctl.update(last + 1.0) == FiringController.WAITING
    # the correct pass that follows fires normally
    last2 = _return_pass(b, 0.4, t0_us=last + 2_000_000.0)
    assert ctl.update(last2) == FiringController.ARMED
    assert ctl.fire_at_us > last2

    b2 = VirtualStation("B", B_CHANNELS, 22.14)
    last = _forward_pass(b2, 0.4)
    single = _ret_ctl(b2)
    assert single.update(last) == FiringController.ABORTED
    assert "wrong-way" in single.abort_reason


def test_forward_leg_rejects_a_pass_receding_from_the_coil():
    """The forward leg's mirror: a -x pass through A is a wrong-way pass too."""
    a = VirtualStation("A", A_CHANNELS, 22.14)
    last = _return_pass(a, 0.4)                 # index-down == toward -x at A
    ctl = FiringController(FIRING, a, leg="fwd", rearm_on_reject=True)
    assert ctl.update(last) == FiringController.REARMED
    assert ctl.update(last + 1.0) == FiringController.WAITING


def test_reset_makes_the_controller_reusable_per_pass():
    b = VirtualStation("B", B_CHANNELS, 22.14)
    last = _return_pass(b, 0.3)
    ctl = _ret_ctl(b)
    ctl.update(last)
    ctl.update(ctl.fire_at_us)
    assert ctl.state == FiringController.FIRED
    ctl.reset()
    assert ctl.state == FiringController.WAITING
    assert b.n_captured() == 0 and ctl.fire_at_us is None and ctl.trim_mm == 0.0
    last2 = _return_pass(b, 0.2, t0_us=5_000_000.0)
    assert ctl.update(last2) == FiringController.ARMED
    assert ctl.v_in_mps == pytest.approx(0.2, rel=1e-9)


def test_partial_capture_rearms_once_in_sustain():
    b = VirtualStation("B", B_CHANNELS, 22.14)
    step = (22.14 / 1000.0) / 0.3 * 1e6
    for k in range(4):
        b.record(4 - k, k * step)
    ctl = _ret_ctl(b, rearm_on_reject=True, incomplete_retries=1)
    ctl.update(3 * step)
    assert ctl.update(3 * step + 201_000.0) == FiringController.REARMED
    assert "4/5" in ctl.reject_reason
    # the retry is spent: a second partial capture aborts
    for k in range(4):
        b.record(4 - k, 1_000_000.0 + k * step)
    ctl.update(1_000_000.0 + 3 * step)
    assert ctl.update(1_000_000.0 + 3 * step + 201_000.0) == FiringController.ABORTED


def test_station_local_velocity_and_direction():
    b = VirtualStation("B", B_CHANNELS, 22.14)
    t, v = 0.0, 0.30
    for k in range(5):
        b.record(4 - k, t)
        t += (22.14 / 1000.0) / v * 1e6
        v *= 0.9
    # local = the last interval, crossed at 0.30 * 0.9^3
    assert b.velocity_local_mps() == pytest.approx(0.30 * 0.9 ** 3, rel=1e-9)
    assert b.pass_direction() == -1 and b.travel_direction() == -1
    assert b.raw_ticks()[0][0] == 4 and b.raw_ticks()[-1][0] == 0
    assert b.first_tick() == 0.0
    # a reversed layout flips travel_direction but not pass_direction
    r = VirtualStation("B", B_CHANNELS, 22.14, rev=True)
    for k in range(5):
        r.record(4 - k, k * 1000.0)
    assert r.pass_direction() == -1 and r.travel_direction() == +1
    # stitched capture: not monotonic
    s = VirtualStation("B", B_CHANNELS, 22.14)
    for i, tt in ((0, 0.0), (1, 100.0), (3, 200.0), (2, 300.0), (4, 400.0)):
        s.record(i, tt)
    assert s.pass_direction() == 0
    e = VirtualStation("B", B_CHANNELS, 22.14)
    assert e.velocity_local_mps() is None and e.first_tick() is None
    b.reset()
    assert b.n_captured() == 0 and b.raw_ticks() == []
