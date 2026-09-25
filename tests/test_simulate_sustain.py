"""Closed-loop sustain twin: the cycle loop, the timing rules and the bank.

Constant-dv maps make the fixed point analytic; the bank closed forms are
checked against hand numbers; the floor and skip/retry policies are driven
through the loop with a starved PSU.
"""

import math
import subprocess
import sys
from pathlib import Path

import pytest

from impulse_map import ImpulseMap, SCHEMA
from rig_profile import load_profile
from simulate_sustain import (
    DEFAULT_LOSSES,
    SustainConfig,
    default_map_path,
    print_summary,
    run_sustain,
)
from sustain_model import (
    BankModel,
    BSideExcess,
    COIL_FACE_X_MM,
    Excursion,
    FLAT_ZONE_X_MM,
    FlatLoss,
    TrackLosses,
    bank_retention,
    fire_delay_us,
    fit_kinematics,
    integrate_flat,
    predict_transit_us,
    return_trim_mm,
    station_read,
)

ROOT = Path(__file__).resolve().parent.parent
HALF = 5.2
V_PSU = 49.6


@pytest.fixture(scope="module")
def profile():
    return load_profile(ROOT, "vbench_v0")


@pytest.fixture(scope="module")
def specs(profile):
    s = profile.station_specs()
    for name, spec in s.items():
        spec["name"] = name
    return s


def const_map(dv, v_bank=V_PSU):
    return ImpulseMap({
        "schema": SCHEMA, "cans": 4, "gate_us": 1500.0, "v_bank_V": v_bank,
        "x_mm": [-40.0, 40.0], "v_in_mps": [0.05, 1.5],
        "dv_mps": [[dv, dv], [dv, dv]],
        "voltage_scaling": {"exponent": 2.0, "basis_V": v_bank},
    })


def peaked_map(dv_peak, x_peak=-13.78, width=8.0, v_bank=V_PSU):
    xs = [-40.0 + i * 2.0 for i in range(41)]
    vs = [0.05, 1.5]
    row = [dv_peak * math.exp(-((x - x_peak) / width) ** 2) for x in xs]
    return ImpulseMap({
        "schema": SCHEMA, "cans": 4, "gate_us": 1500.0, "v_bank_V": v_bank,
        "x_mm": xs, "v_in_mps": vs, "dv_mps": [row, row],
        "voltage_scaling": {"exponent": 2.0, "basis_V": v_bank},
    })


def lossless(alpha=0.8):
    return TrackLosses(FlatLoss(0.0, 0.0), BSideExcess(0.0, 0.0),
                       Excursion(alpha, 0.0), Excursion(alpha, 0.0))


def fast_bank(**kw):
    """A PSU that refills the bank instantly, so the map's basis voltage is what fires."""
    d = dict(cans=4, gate_us=1500.0, psu_amps=1000.0, psu_volts=V_PSU, charge_r=0.01,
             arm_latency_ms=0.0, dt=1e-3, cycles=25, max_shots=60)
    d.update(kw)
    return SustainConfig(**d)


# -- fixed point ---------------------------------------------------------------

def test_converges_to_the_analytic_fixed_point(profile):
    """Constant kick D, lossless flat, excursions v^2 -> alpha v^2.

    v* = D (alpha + sqrt(alpha)) / (1 - alpha) at station A.
    """
    D, alpha = 0.05, 0.8
    res = run_sustain(profile, const_map(D), lossless(alpha),
                      fast_bank(release_v=0.3, cycles=25, max_shots=60))
    v_star = D * (alpha + math.sqrt(alpha)) / (1.0 - alpha)
    lc = res.limit_cycle()
    assert lc["n"] >= 20, res.reason
    assert lc["v_in"] == pytest.approx(v_star, abs=0.003)
    assert lc["trend"] in ("HOLDING", "RISING")
    # every fired kick delivered exactly D at the map's own voltage
    for k in res.fired:
        assert k["dv_coil"] == pytest.approx(D, rel=1e-6)
        assert k["v_bank_at_fire"] == pytest.approx(V_PSU, rel=1e-6)
        assert k["dv_raw"] == pytest.approx(D, abs=1e-6)


def test_fixed_point_is_reached_from_above_too(profile):
    D, alpha = 0.05, 0.8
    # the error contracts by alpha per cycle, so from 0.9 it needs ~40
    res = run_sustain(profile, const_map(D), lossless(alpha),
                      fast_bank(release_v=0.9, cycles=40, max_shots=100))
    v_star = D * (alpha + math.sqrt(alpha)) / (1.0 - alpha)
    assert res.limit_cycle()["v_in"] == pytest.approx(v_star, abs=0.003)


def test_marble_parks_when_the_kick_cannot_cover_the_excursion(profile):
    # beta > 0 is what lets the energy form park; alpha alone only shrinks v
    losses = TrackLosses(FlatLoss(0.0, 0.0), BSideExcess(0.0, 0.0),
                         Excursion(0.5, 0.01), Excursion(0.5, 0.01))
    res = run_sustain(profile, const_map(0.01), losses,
                      fast_bank(release_v=0.3, cycles=25, max_shots=60))
    assert "parked" in res.reason, res.reason
    assert res.limit_cycle()["n"] < 25


# -- timing rules --------------------------------------------------------------

def _decelerating_return_read(specs, a=0.3, v_edge=0.35, dt=5e-4):
    losses = TrackLosses(flat=FlatLoss(a, 0.0))
    channels = {n: s["channel_x_mm"] for n, s in specs.items()}
    seg = integrate_flat(losses, FLAT_ZONE_X_MM, v_edge, -1, channels, HALF, dt=dt,
                         stop_after_station="B")
    assert seg.reason == "station"
    read = station_read(seg.trips["B"], specs["B"], -1, HALF)
    return losses, channels, seg, read


def _x_at_fire(losses, channels, seg, t_fire, dt=5e-4):
    seg2 = integrate_flat(losses, seg.x_mm, seg.v_mps, -1, channels, HALF,
                          t0_us=seg.t_us, dt=dt, t_stop_us=t_fire)
    assert seg2.reason == "t_stop"
    return seg2.x_mm


def test_quadratic_rule_lands_on_target_under_constant_deceleration(specs):
    losses, channels, seg, read = _decelerating_return_read(specs)
    reach = 35.0 + HALF + 9.0
    x_target = COIL_FACE_X_MM - 9.0
    delay_q, info_q = fire_delay_us("quadratic", read, reach)
    x_q = _x_at_fire(losses, channels, seg, read.t_last + delay_q)
    assert abs(x_q - x_target) < 0.5, (x_q, x_target)
    assert not info_q["stalled"] and not info_q["clamped"]
    assert info_q["a"] == pytest.approx(-0.3, rel=0.02)


def test_linear_rule_lands_short_by_the_predicted_amount(specs):
    """Constant-velocity extrapolation on a decelerating ball fires early.

    Short by a tau^2/2 plus the local-vs-last velocity mismatch over tau,
    tau = reach / v_local -- the 8-11 mm the bench swept for by hand.
    """
    losses, channels, seg, read = _decelerating_return_read(specs)
    reach = 35.0 + HALF + 9.0
    x_target = COIL_FACE_X_MM - 9.0
    delay_l, _ = fire_delay_us("linear", read, reach)
    x_l = _x_at_fire(losses, channels, seg, read.t_last + delay_l)
    short = x_l - x_target                      # return leg travels toward -x
    v_last = seg.v_at(read.t_last)
    tau = delay_l * 1e-6
    predicted = (0.5 * 0.3 * tau * tau + (read.v_local - v_last) * tau) * 1000.0
    assert short > 3.0
    assert short == pytest.approx(predicted, abs=0.5)
    delay_q, info_q = fire_delay_us("quadratic", read, reach)
    assert delay_q > delay_l
    # later_mm is the extra delay at v_last; the ball slows over it, so it
    # overstates the shortfall it corrects
    assert short < info_q["later_mm"] < 2.0 * short


def test_predict_transit_matches_the_firmware_numbers():
    assert predict_transit_us(40.2, 0.25, 0.0) == (pytest.approx(160800.0), False)
    t, stalled = predict_transit_us(40.2, 0.25, -0.30)
    assert t == pytest.approx(180300.0, abs=50) and not stalled
    t, stalled = predict_transit_us(40.2, 0.25, 0.30)
    assert t == pytest.approx(147700.0, abs=50) and not stalled
    t, _ = predict_transit_us(40.2, 0.25, 1e-9)
    assert t == pytest.approx(160800.0, abs=1)
    assert predict_transit_us(40.2, 0.12, -0.30) == (pytest.approx(400000.0), True)


def test_fit_kinematics_recovers_v_last_and_a(specs):
    _, _, seg, read = _decelerating_return_read(specs)
    v_last, a, resid = fit_kinematics(read.ticks, read.pitch_mm)
    assert v_last == pytest.approx(seg.v_at(read.t_last), rel=0.01)
    assert a == pytest.approx(-0.3, rel=0.01)
    assert resid < 50.0
    # two crossings: exact line, a = 0; constant velocity: |a| tiny
    v2, a2, _ = fit_kinematics(read.ticks[:2], read.pitch_mm)
    assert a2 == 0.0 and v2 > 0
    ticks = [(i, 100000.0 * i) for i in range(5)]
    v, a, r = fit_kinematics(ticks, 22.14)
    assert v == pytest.approx(0.2214) and abs(a) < 1e-9


def test_reverse_order_decelerating_pass_reads_positive_v_negative_a():
    # crossings in descending spatial index (a return pass), slowing down
    ts, t, v = [], 0.0, 0.3
    for k in range(5):
        ts.append((4 - k, t))
        dt = 0.02214 / v
        t += dt * 1e6
        v -= 0.3 * dt
    v_last, a, _ = fit_kinematics(ts, 22.14)
    assert v_last > 0 and a < 0
    assert a == pytest.approx(-0.3, rel=0.05)


def test_return_trim_reproduces_the_firmware_constants():
    assert return_trim_mm(0.2, 1500) == pytest.approx(5.3, abs=0.05)
    assert return_trim_mm(0.6, 1500) == pytest.approx(-0.37, abs=0.05)
    assert return_trim_mm(0.17, 700) == pytest.approx(10.0, abs=0.05)
    assert return_trim_mm(0.01, 700) == 16.0


# -- bank --------------------------------------------------------------------

def test_bank_closed_forms_against_hand_numbers():
    b = BankModel(7.392e-3, 22.0, 49.6, 1.0)
    t = b.t_to(11.0, 47.6)
    assert 0.45 < t < 0.58, t
    assert b.v_after(11.0, t) == pytest.approx(47.6, abs=1e-9)
    b3 = BankModel(7.392e-3, 22.0, 49.6, 0.3)
    t3 = b3.t_to(11.0, 47.6)
    assert 0.9 < t3 < 1.1, t3
    assert b3.v_after(11.0, t3) == pytest.approx(47.6, abs=1e-9)
    # continuous through the knee, monotone, asymptotic to V_psu
    t_knee = (b.v_knee - 11.0) * b.C / b.I
    assert b.v_after(11.0, t_knee - 1e-6) == pytest.approx(b.v_after(11.0, t_knee + 1e-6), abs=1e-3)
    vs = [b.v_after(11.0, 0.05 * i) for i in range(30)]
    assert vs == sorted(vs)
    assert b.v_after(11.0, 100.0) == pytest.approx(49.6)
    assert b.t_to(20.0, 10.0) == 0.0
    assert math.isinf(b.t_to(10.0, 49.6))


def test_bank_retention_table_and_interpolation():
    assert bank_retention(4, 1500) == pytest.approx(0.242)
    assert bank_retention(4, 700) == pytest.approx(0.534)
    assert bank_retention(5, 1500) == pytest.approx(0.314)
    assert bank_retention(3, 700) == pytest.approx(0.41)
    mid = bank_retention(4, 850)
    assert 0.390 < mid < 0.534
    assert bank_retention(4, 850) == pytest.approx(0.462, abs=1e-6)
    b = BankModel(7.392e-3, 22.0, 49.6, 1.0)
    assert b.v_post_shot(47.6, 1500, 4) == pytest.approx(47.6 * 0.242)


# -- policies through the loop ------------------------------------------------

def test_floor_policy_fires_with_what_the_bank_has(profile):
    """A starved PSU: the ball is back before 0.96 V_start; a tripped channel
    releases the wait at the 0.60 floor and the bank keeps charging until the
    pulse, so v_bank_at_fire > v_bank_at_release."""
    cfg = fast_bank(psu_amps=0.2, charge_r=22.0, release_v=0.5, cycles=4,
                    max_shots=8, arm_latency_ms=50.0)
    res = run_sustain(profile, const_map(0.1), lossless(0.9), cfg)
    ret = [k for k in res.kicks if k["leg"] == "ret"]
    assert ret and not ret[0]["skipped"], res.kicks
    k = ret[0]
    assert 0.60 * V_PSU - 1e-6 <= k["v_bank_at_release"] < 0.96 * V_PSU, k
    assert k["v_bank_at_fire"] > k["v_bank_at_release"]
    assert k["v_bank_at_fire"] < V_PSU
    assert k["dv_coil"] == pytest.approx(0.1 * (k["v_bank_at_fire"] / V_PSU) ** 2)


def test_slipped_pass_is_skipped_and_the_leg_retries_next_time(profile):
    """Bank below the floor when the return pass comes: skip, coast, and the
    return leg fires on the NEXT return pass; the forward pass in between
    gets no kick."""
    cfg = fast_bank(psu_amps=0.05, charge_r=22.0, release_v=0.5, cycles=4,
                    max_shots=8, arm_latency_ms=50.0)
    res = run_sustain(profile, const_map(0.1), lossless(0.95), cfg)
    legs = [(k["leg"], k["skipped"]) for k in res.kicks]
    assert legs[0] == ("fwd", False)
    assert legs[1] == ("ret", True), legs
    assert "gone" in res.kicks[1]["skip_reason"]
    assert legs[2][0] == "ret", legs
    assert res.cycles[1]["kicked"] is False
    assert res.cycles[0]["kicked"] is True


def test_late_arming_skips_the_pass(profile):
    cfg = fast_bank(release_v=0.5, cycles=3, max_shots=6, arm_latency_ms=5000.0)
    res = run_sustain(profile, const_map(0.1), lossless(0.95), cfg)
    assert res.kicks[0]["skipped"] is False
    assert res.kicks[1]["skipped"] is True
    assert "armed" in res.kicks[1]["skip_reason"]


def test_forward_kick_lands_short_from_flat_loss(profile):
    losses = TrackLosses(FlatLoss(0.0, 0.8), BSideExcess(0.0, 0.0),
                         Excursion(0.9, 0.0), Excursion(0.9, 0.0))
    res = run_sustain(profile, const_map(0.15), losses,
                      fast_bank(release_v=0.5, cycles=3, max_shots=6))
    fwd = [k for k in res.fired if k["leg"] == "fwd"]
    assert fwd
    for k in fwd:
        assert k["x_target"] == pytest.approx(-COIL_FACE_X_MM + 9.0)
        short = k["x_target"] - k["x_delivered"]
        assert 0.2 < short < 3.0, short


def bench_like_losses():
    """Of the order the 09-24 runs measured: ~0.1 m/s lost per excursion,
    ~15% deceleration through B on the return, 0.2 m/s kicks."""
    return TrackLosses(FlatLoss(0.02, 0.5), BSideExcess(0.25, 0.2),
                       Excursion(0.93, 0.002), Excursion(0.95, 0.002))


def test_levelling_the_b_side_is_monotone(profile):
    losses = bench_like_losses()
    cycles = []
    for scale in (1.0, 0.5, 0.0):
        cfg = fast_bank(release_v=0.5, cycles=12, max_shots=30, b_slope_scale=scale,
                        timing_rule="quadratic", ret_offset=0.0)
        res = run_sustain(profile, peaked_map(0.25), losses, cfg)
        lc = res.limit_cycle()
        assert lc["n"] >= 8, (scale, res.reason)
        cycles.append(lc["v_in"])
    assert cycles[0] < cycles[1] < cycles[2], cycles


def test_quadratic_return_timing_beats_untrimmed_linear_on_a_peaked_map(profile):
    """Untrimmed constant-velocity timing fires the return kick ~7 mm short
    on a decelerating approach; the predictor lands within a few mm (the
    excess ends at the coil face, before the fire point, so constant-a
    extrapolation is not exact there either) and the cycle is faster."""
    losses = bench_like_losses()
    out = {}
    for rule in ("linear", "quadratic"):
        cfg = fast_bank(release_v=0.5, cycles=12, max_shots=30, timing_rule=rule,
                        ret_offset=0.0)
        res = run_sustain(profile, peaked_map(0.25), losses, cfg)
        ret = [k for k in res.fired if k["leg"] == "ret"]
        assert len(ret) >= 6, (rule, res.reason)
        out[rule] = (res.limit_cycle()["v_in"],
                     sum(abs(k["x_delivered"] - k["x_target"]) for k in ret) / len(ret),
                     sum(k["dv_coil"] for k in ret) / len(ret))
    # the linear shortfall shrinks as the cycle speeds up (a fast ball barely
    # decelerates), so compare the two rules rather than pin a distance
    assert out["quadratic"][1] < 1.0, out
    assert out["linear"][1] > max(1.5, 2.0 * out["quadratic"][1]), out
    assert out["quadratic"][2] > out["linear"][2], out
    assert out["quadratic"][0] > out["linear"][0], out


def test_summary_prints_in_the_firmware_format(profile, capsys):
    res = run_sustain(profile, const_map(0.05), lossless(0.8),
                      fast_bank(release_v=0.3, cycles=6, max_shots=12))
    print_summary(res)
    text = capsys.readouterr().out
    assert "v_in at A per cycle, m/s:" in text
    assert any(w in text for w in ("RISING", "FALLING", "HOLDING"))
    assert "far ramp excursion:" in text and "entry ramp excursion:" in text
    assert "[A> kick 1/12" in text and "[<B kick 2/12" in text
    d = res.to_dict()
    assert d["limit_cycle"]["n"] == 6 and len(d["kicks"]) == 12


# -- CLI ---------------------------------------------------------------------

def test_cli_smoke(tmp_path):
    map_path = default_map_path(4, 1500)
    if not map_path.exists() or not DEFAULT_LOSSES.exists():
        pytest.skip("needs config/impulse_maps/impulse_map_4can_1500us.json and "
                    "config/track_losses.json (built on the GPU / from bench data)")
    out = tmp_path / "sustain.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "simulate_sustain.py"),
         "--cans", "4", "--gate", "1500", "--cycles", "3", "--max-shots", "6",
         "--out", str(out)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert "sustain ended" in proc.stdout
