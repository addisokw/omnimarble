"""The Kit sustain leg state machine, driven headless with the 1-D twin's kinematics.

The acceptance test that can run without Kit: SustainLoop (the machine the
extension polls per physics step) makes every decision -- leg, arming, bank
release, fire time, skip, v_out -- while sustain_model.integrate_flat moves
the ball and the impulse map supplies the kick, exactly as simulate_sustain
does. At 4 cans / 1500 us / 1.0 A the loop must reproduce the twin's limit
cycle to +/-0.02 m/s; in practice the two agree to well under a mm/s because
they are the same rules, and the tolerance is the Kit acceptance band.
"""

import math
from pathlib import Path

import pytest

from coil_sensing import FiringController, VirtualStation
from impulse_map import load_map
from rig_profile import load_profile
from simulate_sustain import DEFAULT_LOSSES, SustainConfig, default_map_path, run_sustain
from sustain_loop import RECORD_COLUMNS, SustainLoop, SustainSettings, write_records_csv
from sustain_model import (
    BankModel,
    FLAT_ZONE_X_MM,
    TrackLosses,
    bank_capacitance_F,
    excursion_return,
    excursion_time_s,
    integrate_flat,
    kick_dv,
)

ROOT = Path(__file__).resolve().parent.parent
MAP_4CAN_1500 = default_map_path(4, 1500)


@pytest.fixture(scope="module")
def profile():
    return load_profile(ROOT, "vbench_v0")


def _stations(profile):
    return {name: VirtualStation(name, spec["channel_x_mm"], spec["pitch_mm"],
                                 rev=spec["order_rev"])
            for name, spec in profile.station_specs().items()}


def drive_headless(profile, imap, losses, settings, release_v=1.0, dt=5e-4,
                   kick_scale=1.0, max_time_s=200.0):
    """Run SustainLoop on the twin's kinematics. Returns the loop.

    The ball coasts on the flat through integrate_flat; every integrator
    sample is a poll, channel trips are fed to the loop's stations as they
    happen, and once the loop has planned a fire time the segment is cut
    exactly there so the pulse lands where the loop asked, not at the next
    sample. Ramps are the fitted energy-form transfer, as in the twin.
    """
    specs = profile.station_specs()
    channels = {n: s["channel_x_mm"] for n, s in specs.items()}
    half = float(profile.sensing.get("detect_halfwidth_mm", 0.0))
    stations = _stations(profile)
    bank = BankModel(bank_capacitance_F(profile, settings.cans), settings.charge_r_ohm,
                     settings.psu_volts, settings.psu_amps)
    loop = SustainLoop(profile, stations, settings, bank, v_start=settings.psu_volts)

    x, v, direction, t = -FLAT_ZONE_X_MM, float(release_v), +1, 0.0

    def feed(trips, upto_us, cursor):
        while cursor < len(trips) and trips[cursor][0] <= upto_us:
            t_trip, name, idx = trips[cursor]
            stations[name].record(idx, t_trip)
            cursor += 1
        return cursor

    while not loop.finished and t * 1e-6 < max_time_s:
        seg = integrate_flat(losses, x, v, direction, channels, half, t0_us=t, dt=dt)
        trips = sorted((tt, name, idx) for name, lst in seg.trips.items()
                       for idx, tt in lst)
        cursor = 0
        fired = False
        for k in range(1, len(seg.ts)):
            t_k = seg.ts[k]
            cursor = feed(trips, t_k, cursor)
            loop.poll(t_k)
            if loop.finished:
                break
            if loop.state == SustainLoop.ARMED and loop.t_fire_planned_us <= seg.ts[-1]:
                t_f = loop.t_fire_planned_us
                if t_f > t_k:
                    seg2 = integrate_flat(losses, seg.xs[k], seg.vs[k], direction,
                                          channels, half, t0_us=t_k, dt=dt, t_stop_us=t_f)
                    xf, vf = seg2.x_mm, seg2.v_mps
                    if seg2.reason != "t_stop":
                        loop.finish(t_f, "marble %s before the pulse (x = %.1f mm)"
                                    % ("stalled" if seg2.stalled else "left the flat", xf))
                        break
                else:
                    xf, vf = seg.xs[k], seg.vs[k]
                cursor = feed(trips, t_f, cursor)
                ev = loop.poll(t_f)
                assert ev is not None and ev["event"] == "fire", ev
                dv, _ = kick_dv(imap, xf, vf, ev["v_bank"], loop.leg, kick_scale)
                loop.note_delivery(xf, vf, dv)
                loop.shot_done(t_f, bank.v_post_shot(ev["v_bank"], settings.on_time_us,
                                                     settings.cans))
                x, v, t = xf, vf + dv, t_f
                fired = True
                if v <= 0.0:
                    loop.finish(t_f, "the kick reversed the marble")
                break
        if loop.finished or fired:
            continue
        cursor = feed(trips, seg.t_us, cursor)
        loop.poll(seg.t_us)
        if loop.finished:
            break
        if seg.reason == "stalled":
            loop.finish(seg.t_us, "marble stalled on the flat at x = %.1f mm" % seg.x_mm)
            break
        if seg.reason != "x_stop":
            loop.finish(seg.t_us, "marble did not reach the ramp (%s)" % seg.reason)
            break
        x, v, t = seg.x_mm, seg.v_mps, seg.t_us
        side = "far" if direction > 0 else "entry"
        v_back = excursion_return(losses, side, v)
        if v_back is None:
            loop.finish(t, "marble parked on the %s ramp (left the flat at %.3f m/s)"
                        % (side, v))
            break
        t += excursion_time_s(losses, v, v_back) * 1e6
        x, v, direction = direction * FLAT_ZONE_X_MM, v_back, -direction
    if not loop.finished:
        loop.finish(t, "driver time limit")
    return loop


def _needs_bench_inputs():
    if not MAP_4CAN_1500.exists() or not DEFAULT_LOSSES.exists():
        pytest.skip("needs config/impulse_maps/impulse_map_4can_1500us.json and "
                    "config/track_losses.json")


# -- acceptance: the loop reproduces the 1-D twin -------------------------------

def test_loop_reproduces_the_twin_limit_cycle_at_4can_1500us_1A(profile):
    """The Kit acceptance band, +/-0.02 m/s on the cycle, applied to the machine
    the extension will drive. Same map, same losses, same bank."""
    _needs_bench_inputs()
    imap = load_map(MAP_4CAN_1500)
    losses = TrackLosses.load(DEFAULT_LOSSES)
    cfg = SustainConfig(cans=4, gate_us=1500.0, psu_amps=1.0, psu_volts=49.6,
                        charge_r=22.0, cycles=15, max_shots=30)
    twin = run_sustain(profile, imap, TrackLosses.load(DEFAULT_LOSSES), cfg)
    twin_lc = twin.limit_cycle()
    assert twin_lc["n"] >= 10, twin.reason

    settings = SustainSettings(cans=4, on_time_us=1500.0, max_shots=30, max_seconds=120.0,
                               psu_amps=1.0, psu_volts=49.6, charge_r_ohm=22.0)
    loop = drive_headless(profile, imap, losses, settings, release_v=cfg.release_v)
    lc = loop.limit_cycle()
    assert lc["n"] >= 10, loop.reason
    assert lc["v_in"] == pytest.approx(twin_lc["v_in"], abs=0.02), (lc, twin_lc)
    # and, being the same rules, they agree far more tightly than the band
    assert lc["v_in"] == pytest.approx(twin_lc["v_in"], abs=0.003), (lc, twin_lc)
    # kick by kick: legs alternate, delivered positions and trims match
    twin_fired = twin.fired
    loop_fired = loop.fired
    assert len(loop_fired) == len(twin_fired) == 30
    for a, b in zip(loop_fired, twin_fired):
        assert a["leg"] == b["leg"]
        assert a["x_delivered"] == pytest.approx(b["x_delivered"], abs=0.3)
        assert a["trim_mm"] == pytest.approx(b["trim_mm"], abs=0.05)
        assert a["v_bank_at_fire"] == pytest.approx(b["v_bank_at_fire"], abs=0.2)
        assert a["dv_raw"] == pytest.approx(b["dv_raw"], abs=0.005)
    assert loop.reason.startswith("shot count")


def test_loop_reproduces_the_twin_on_the_starved_psu_holdout(profile):
    """0.3 A: the bank releases at the 0.60 floor on tripped channels and the
    cycle still holds -- the held-out run the twin scored a HIT on."""
    _needs_bench_inputs()
    imap = load_map(MAP_4CAN_1500)
    losses = TrackLosses.load(DEFAULT_LOSSES)
    cfg = SustainConfig(cans=4, gate_us=1500.0, psu_amps=0.3, psu_volts=49.6,
                        charge_r=22.0, cycles=15, max_shots=30)
    twin = run_sustain(profile, imap, TrackLosses.load(DEFAULT_LOSSES), cfg)
    settings = SustainSettings(cans=4, on_time_us=1500.0, max_shots=30, max_seconds=120.0,
                               psu_amps=0.3, psu_volts=49.6, charge_r_ohm=22.0)
    loop = drive_headless(profile, imap, losses, settings)
    assert loop.limit_cycle()["v_in"] == pytest.approx(twin.limit_cycle()["v_in"], abs=0.02)
    floors = [r for r in loop.fired if r["v_bank_at_release"] < 0.96 * 49.6 - 1e-6]
    assert floors, "a 0.3 A PSU must be releasing at the floor on some passes"
    for r in floors:
        assert r["v_bank_at_release"] >= 0.60 * 49.6 - 1e-6
        assert r["v_bank_at_fire"] >= r["v_bank_at_release"]


# -- machine behaviour without bench inputs -------------------------------------

class _ConstMap:
    """A map that always kicks by D (rescaled by bank voltage squared)."""

    def __init__(self, dv, v_basis=49.6):
        self.dv = dv
        self.v_basis = v_basis

    def lookup(self, x_mm, v_in):
        return self.dv, False

    def rescale_factor(self, v_bank):
        return (v_bank / self.v_basis) ** 2


def _lossless(alpha=0.8):
    from sustain_model import BSideExcess, Excursion, FlatLoss
    return TrackLosses(FlatLoss(0.0, 0.0), BSideExcess(0.0, 0.0),
                       Excursion(alpha, 0.0), Excursion(alpha, 0.0))


def _fast_settings(**kw):
    d = dict(cans=4, on_time_us=1500.0, max_shots=60, max_seconds=300.0,
             psu_amps=1000.0, psu_volts=49.6, charge_r_ohm=0.01)
    d.update(kw)
    return SustainSettings(**d)


def test_legs_alternate_and_reach_the_analytic_fixed_point(profile):
    """Constant kick D, lossless flat: v* = D (alpha + sqrt(alpha)) / (1 - alpha)."""
    D, alpha = 0.05, 0.8
    loop = drive_headless(profile, _ConstMap(D), _lossless(alpha), _fast_settings(),
                          release_v=0.3, dt=1e-3)
    legs = [r["leg"] for r in loop.fired]
    assert legs[:6] == ["fwd", "ret", "fwd", "ret", "fwd", "ret"]
    v_star = D * (alpha + math.sqrt(alpha)) / (1.0 - alpha)
    assert loop.limit_cycle()["v_in"] == pytest.approx(v_star, abs=0.003)
    assert loop.reason == "shot count (60)"
    # every record is complete
    for r in loop.fired:
        for col in ("x_delivered", "v_bank_at_fire", "v_out_fit", "dv_raw", "t_fire_s"):
            assert r[col] is not None, (col, r)
    # cycles are counted per forward pass at A
    assert loop.cycles[0]["v_in_A"] == pytest.approx(loop.fired[0]["v_in_fit"])
    assert all(c["kicked"] for c in loop.cycles)


def test_shot_budget_and_clock_stop_the_run(profile):
    loop = drive_headless(profile, _ConstMap(0.05), _lossless(0.8),
                          _fast_settings(max_shots=5), release_v=0.3, dt=1e-3)
    assert loop.n_shots == 5 and loop.reason == "shot count (5)"
    loop = drive_headless(profile, _ConstMap(0.05), _lossless(0.8),
                          _fast_settings(max_seconds=3.0), release_v=0.3, dt=1e-3)
    assert loop.reason == "clock (3 s)"
    assert loop.t_end_us * 1e-6 >= 3.0


def test_slipped_pass_is_skipped_and_the_leg_retries(profile):
    """A starved PSU: the return pass comes before the floor; skip, coast, the
    forward pass in between gets no kick, and the return leg fires on the next
    return pass -- after re-arming through the wrong-way pass at B."""
    settings = _fast_settings(psu_amps=0.05, charge_r_ohm=22.0, max_shots=8)
    loop = drive_headless(profile, _ConstMap(0.1), _lossless(0.95), settings,
                          release_v=0.5, dt=1e-3)
    legs = [(r["leg"], r["skipped"]) for r in loop.records]
    assert legs[0] == ("fwd", False)
    assert legs[1] == ("ret", True), legs
    assert "gone" in loop.records[1]["skip_reason"]
    assert legs[2][0] == "ret"
    rearms = [e for e in loop.events if "wrong-way" in e[1]]
    assert rearms, loop.events


def test_records_write_as_csv(profile, tmp_path):
    loop = drive_headless(profile, _ConstMap(0.05), _lossless(0.8),
                          _fast_settings(max_shots=4), release_v=0.3, dt=1e-3)
    path = write_records_csv(tmp_path / "kit_sustain_test.csv", loop.records,
                             {"source": "test"})
    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0] == "# source=test"
    assert text[1] == ",".join(RECORD_COLUMNS)
    assert len(text) == 2 + 4
    lines = loop.summary_lines()
    joined = "\n".join(lines)
    assert "sustain ended: shot count (4)" in joined
    assert "v_in at A per cycle, m/s:" in joined
    assert "far ramp excursion:" in joined and "entry ramp excursion:" in joined


def test_wrong_way_pass_rearms_the_return_leg(profile):
    """Hand-fed: the ret leg watches B; a +x pass through B is refused and
    the station cleared; the -x pass that follows fires."""
    stations = _stations(profile)
    settings = _fast_settings()
    bank = BankModel(7.392e-3, 0.01, 49.6, 1000.0)
    loop = SustainLoop(profile, stations, settings, bank, v_start=49.6)
    loop._new_leg("ret", 0.0)
    b = stations["B"]
    v = 0.4
    step = 22.14e-3 / v * 1e6
    t = 1_000_000.0
    for i in range(5):                       # index-up == toward +x at B
        b.record(i, t + i * step)
    ev = loop.poll(t + 4 * step + 100.0)
    assert ev["event"] == "rearm" and "wrong-way" in ev["reason"]
    assert b.n_captured() == 0 and loop.state == SustainLoop.WAITING
    t = 3_000_000.0
    for i in range(5):                       # index-down == toward -x
        b.record(4 - i, t + i * step)
    ev = loop.poll(t + 4 * step + 100.0)
    assert ev["event"] == "armed"
    ctl = loop.ctl
    assert ctl.leg == "ret" and ctl.trim_mm > 0
    expected = 35.0 + 5.2 + 9.0 + ctl.trim_mm
    assert ctl.reach_mm == pytest.approx(expected)
    assert ev["t_fire_us"] == pytest.approx(t + 4 * step + expected / 1000.0 / v * 1e6)
