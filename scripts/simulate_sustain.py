"""Closed-loop sustain twin: predict the limit cycle from map + losses.

    uv run python scripts/simulate_sustain.py --cans 4 --gate 1500 --psu-amps 1.0
    uv run python scripts/simulate_sustain.py --cans 4 --sweep-gates 700,1000,1500 --psu-amps 0.3
    uv run python scripts/simulate_sustain.py --cans 4 --gate 1500 --timing-rule quadratic --ret-offset 0

Mirrors vbench firmware/main.py cmd_sustain one pass at a time:

  * legs alternate fwd (trigger A, fire at -22.78 + fire_offset) and ret
    (trigger B, fire at +22.78 - fire_offset - ret_offset); a pass in the
    wrong direction for the awaited leg just coasts;
  * the bank wait releases at recharge_frac * V_start, or at floor_frac *
    V_start once a channel of the trigger station has tripped
    (_wait_bank_ready); the bank keeps charging until the pulse, so
    v_bank_at_fire > v_bank_at_release, which is the number the kick sees;
  * a pass whose fire window has slipped by more than slip_us when the bank
    releases is skipped and the leg stays armed for the next same-direction
    pass ("retry"); a station armed after the ball has already entered it
    (arm_latency_ms) yields a partial capture, also skipped;
  * forward timing is the linear rule on the fit velocity; return timing is
    the chosen rule (linear + today's K/v trim, or the quadratic predictor)
    on the station's own crossings, read through the same estimator the
    firmware uses, so its biases (the return fit reads high) are inherited;
  * the kick is an instantaneous dv from the impulse map at the DELIVERED
    position (mirrored on the return), scaled by (V_fire/V_map)^2 and
    kick_scale; the ball's own speed is what the map's v_in axis sees.

Not modelled: the rescue fire on a < 5-channel pass (a ball that stalls in
a station ends the run), the incoherent-capture gates, and the 50 ms bank
polling granularity.

Output: per-kick records (delivered x included -- the forward kick lands
~1 mm short of -13.78 because the ball slows over the 49 mm reach), per-cycle
v_in at A, the limit cycle, and a summary in the firmware's own format.
"""

import argparse
import dataclasses
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "source" / "extensions" / "omni.marble.coaster"
                       / "omni" / "marble" / "coaster"))

from impulse_map import load_map  # noqa: E402
from rig_profile import load_profile  # noqa: E402
from sustain_model import (  # noqa: E402
    BankModel,
    COIL_FACE_X_MM,
    FLAT_ZONE_X_MM,
    TrackLosses,
    bank_capacitance_F,
    excursion_return,
    excursion_time_s,
    fire_delay_us,
    integrate_flat,
    kick_dv,
    return_trim_mm,
    station_read,
)

DEFAULT_MAP_DIR = ROOT / "config" / "impulse_maps"
DEFAULT_LOSSES = ROOT / "config" / "track_losses.json"


def default_map_path(cans, gate_us):
    return DEFAULT_MAP_DIR / f"impulse_map_{int(cans)}can_{int(gate_us)}us.json"


@dataclasses.dataclass
class SustainConfig:
    cans: int = 4
    gate_us: float = 1500.0
    psu_amps: float = 1.0
    psu_volts: float = 49.6
    charge_r: float = 22.0
    fire_offset: float = 9.0
    ret_offset: object = "auto"        # "auto" (K/v trim under linear, 0 under quadratic) or mm
    timing_rule: str = "linear"        # return-leg rule: linear | quadratic
    b_slope_scale: float = 1.0
    b_conservative: bool = False
    release_v: float = 1.0             # ball speed entering the flat zone at the start, m/s
    recharge_frac: float = 0.96
    floor_frac: float = 0.60
    cycles: int = 15
    max_shots: int = 30
    max_seconds: float = 120.0
    kick_scale: float = 1.0
    shape_csv: str = ""               # measured fire-position curve overriding the map's entry-side shape
    corrections_forward_only: bool = True   # shape/kick_scale were measured on the FORWARD leg; the return leg keeps the frozen mirror
    arm_latency_ms: float = 100.0
    loss_form: str = "energy"          # energy | linear (overrides the table's form)
    dt: float = 5e-4
    slip_us: float = 2000.0
    recharge_timeout_s: float = 30.0
    max_extra_mm: float = 16.0
    last_channel_to_coil_mm: float = 35.0

    def to_dict(self):
        return dataclasses.asdict(self)


class RunResult:
    def __init__(self, cfg):
        self.cfg = cfg
        self.kicks = []          # per attempted kick (fired or skipped)
        self.cycles = []         # per forward pass: {"cycle", "v_in_A", "kicked"}
        self.reason = "unknown"
        self.t_end_s = 0.0
        self.n_shots = 0
        self.v_start = None

    @property
    def fired(self):
        return [k for k in self.kicks if not k["skipped"]]

    def fwd_v_in(self):
        return [k["v_in_fit"] for k in self.fired if k["leg"] == "fwd" and k["v_in_fit"]]

    def limit_cycle(self, n_last=5):
        """Mean v_in at A over the last n_last fired forward kicks, plus the drift."""
        good = self.fwd_v_in()
        if not good:
            return {"n": 0, "v_in": None, "drift": None, "trend": "DEAD"}
        tail = good[-n_last:]
        mean = sum(tail) / len(tail)
        if len(good) >= 2:
            drift = (good[-1] - good[0]) / (len(good) - 1)
            trend = ("RISING" if drift > 0.002 else
                     "FALLING" if drift < -0.002 else "HOLDING")
        else:
            drift, trend = 0.0, "SINGLE"
        tail_drift = ((tail[-1] - tail[0]) / (len(tail) - 1)) if len(tail) >= 2 else 0.0
        return {"n": len(good), "v_in": mean, "v_in_last": good[-1],
                "drift": drift, "tail_drift": tail_drift, "trend": trend,
                "n_tail": len(tail)}

    def excursion_losses(self):
        """As the firmware computes them, from consecutive fired kicks."""
        far, entry = [], []
        shots = self.fired
        for a, b in zip(shots, shots[1:]):
            if a["leg"] == "fwd" and b["leg"] == "ret" and a["v_out_fit"] and b["v_in_fit"]:
                far.append(a["v_out_fit"] - b["v_in_fit"])
            if a["leg"] == "ret" and b["leg"] == "fwd" and a["v_out_fit"] and b["v_in_fit"]:
                entry.append(a["v_out_fit"] - b["v_in_fit"])
        return far, entry

    def to_dict(self):
        return {"config": self.cfg.to_dict(), "reason": self.reason,
                "t_end_s": self.t_end_s, "n_shots": self.n_shots,
                "v_start": self.v_start, "kicks": self.kicks,
                "cycles": self.cycles, "limit_cycle": self.limit_cycle(),
                "excursions": dict(zip(("far", "entry"), self.excursion_losses()))}


def _apply_loss_form(losses, form):
    if form and form != "energy":
        losses.far.form = form
        losses.entry.form = form
    return losses


def run_sustain(profile, imap, losses, cfg, imap_ret=None):
    imap_ret = imap_ret if imap_ret is not None else imap
    """Run the cycle loop. Returns a RunResult."""
    losses = _apply_loss_form(losses, cfg.loss_form)
    specs = profile.station_specs()
    for name, s in specs.items():
        s["name"] = name
    st_in = profile.sensing.get("station_in", "A")
    st_out = profile.sensing.get("station_out", "B")
    channels = {n: s["channel_x_mm"] for n, s in specs.items()}
    half = float(profile.sensing.get("detect_halfwidth_mm", 0.0))
    n_ch = len(channels[st_in])

    bank = BankModel(bank_capacitance_F(profile, cfg.cans), cfg.charge_r,
                     cfg.psu_volts, cfg.psu_amps)
    v_start = cfg.psu_volts
    v_target = cfg.recharge_frac * v_start
    v_floor = cfg.floor_frac * v_start

    res = RunResult(cfg)
    res.v_start = v_start
    t = 0.0                     # us
    x = -FLAT_ZONE_X_MM
    v = float(cfg.release_v)
    direction = +1
    awaited = "fwd"
    t_last_fire = None
    v_post = v_start
    cycle = 0
    reason = None

    def bank_v(t_us):
        if t_last_fire is None:
            return v_start
        return bank.v_after(v_post, (t_us - t_last_fire) * 1e-6)

    while True:
        if t * 1e-6 >= cfg.max_seconds:
            reason = "clock (%.0f s)" % cfg.max_seconds
            break
        pass_leg = "fwd" if direction > 0 else "ret"
        if pass_leg == "fwd":
            cycle += 1
            if cycle > cfg.cycles:
                reason = "cycle limit (%d)" % cfg.cycles
                break
        if res.n_shots >= cfg.max_shots:
            reason = "shot count (%d)" % cfg.max_shots
            break
        trig = st_in if pass_leg == "fwd" else st_out
        out_st = st_out if pass_leg == "fwd" else st_in

        # -- segment 1: coast to the end of the trigger station ------------
        seg1 = integrate_flat(losses, x, v, direction, channels, half, t0_us=t,
                              dt=cfg.dt, stop_after_station=trig,
                              b_slope_scale=cfg.b_slope_scale,
                              b_conservative=cfg.b_conservative)
        if seg1.reason == "stalled":
            reason = "marble stalled on the flat at x = %.1f mm" % seg1.x_mm
            break
        if seg1.reason != "station":
            reason = "marble never completed station %s (%s)" % (trig, seg1.reason)
            break
        read = station_read(seg1.trips[trig], specs[trig], direction, half)
        x, v, t = seg1.x_mm, seg1.v_mps, seg1.t_us
        if pass_leg == "fwd":
            res.cycles.append({"cycle": cycle, "v_in_A": read.v_fit, "kicked": False,
                               "t_s": t * 1e-6})

        kicked = False
        rec = None
        if pass_leg == awaited:
            rec = {"leg": pass_leg, "cycle": cycle, "kick_idx": len(res.kicks) + 1,
                   "t_s": read.t_last * 1e-6, "v_in_fit": read.v_fit,
                   "v_local": read.v_local, "resid_us": read.residual_us,
                   "trim_mm": 0.0, "skipped": False, "skip_reason": None}
            # arming
            t_first = read.ticks[0][1]
            t_armed = (-float("inf") if t_last_fire is None
                       else t_last_fire + cfg.arm_latency_ms * 1000.0)
            # bank release
            if t_last_fire is None:
                t_release = -float("inf")
                v_at_arm = v_start
            else:
                v_at_arm = bank_v(t_armed)
                t96 = t_last_fire + bank.t_to(v_post, v_target) * 1e6
                t60 = t_last_fire + bank.t_to(v_post, v_floor) * 1e6
                t_release = min(t96, max(t60, t_first))
                if t_release - t_armed > cfg.recharge_timeout_s * 1e6:
                    reason = "bank did not recharge"
                    res.kicks.append({**rec, "skipped": True,
                                      "skip_reason": "bank stalled"})
                    break
            rec["v_bank_at_arm"] = v_at_arm
            rec["v_bank_at_release"] = bank_v(max(t_release, t_first))
            # timing
            reach = cfg.last_channel_to_coil_mm + half + cfg.fire_offset
            if pass_leg == "fwd":
                delay, info = fire_delay_us("linear", read, reach,
                                            v_linear=read.v_fit)
                x_target = -COIL_FACE_X_MM + cfg.fire_offset
            else:
                if cfg.ret_offset == "auto":
                    trim = (return_trim_mm(read.v_local or read.v_fit, cfg.gate_us)
                            if cfg.timing_rule == "linear" else 0.0)
                else:
                    trim = float(cfg.ret_offset)
                rec["trim_mm"] = trim
                reach += trim
                delay, info = fire_delay_us(cfg.timing_rule, read, reach,
                                            max_extra_mm=cfg.max_extra_mm)
                x_target = COIL_FACE_X_MM - cfg.fire_offset - trim
            rec.update({"x_target": x_target, "reach_mm": reach,
                        "transit_us": delay, "v_transit": info["v_used"],
                        "accel_mps2": info["a"], "stalled": info["stalled"],
                        "clamped": info.get("clamped", False),
                        "later_mm": info.get("later_mm", 0.0)})
            t_fire = read.t_last + delay
            rec["late_us"] = t_fire - t_release if t_release > -float("inf") else None
            if t_first < t_armed:
                rec["skipped"] = True
                rec["skip_reason"] = "station armed %.0f ms after the ball entered it" % (
                    (t_armed - t_first) * 1e-3)
            elif t_release > t_fire + cfg.slip_us:
                rec["skipped"] = True
                rec["skip_reason"] = "fire window %.0f ms gone when the bank released" % (
                    (t_release - t_fire) * 1e-3)
            if not rec["skipped"]:
                t_fire_actual = max(t_fire, t_release)
                # -- segment 2: coast to the pulse -----------------------
                seg2 = integrate_flat(losses, x, v, direction, channels, half,
                                      t0_us=t, dt=cfg.dt, t_stop_us=t_fire_actual,
                                      b_slope_scale=cfg.b_slope_scale,
                                      b_conservative=cfg.b_conservative)
                x, v, t = seg2.x_mm, seg2.v_mps, seg2.t_us
                if seg2.reason != "t_stop":
                    reason = "marble %s before the pulse (x = %.1f mm)" % (
                        "stalled" if seg2.stalled else "left the flat", x)
                    res.kicks.append({**rec, "skipped": True, "skip_reason": reason})
                    break
                v_fire = bank_v(t_fire_actual)
                if pass_leg == "ret" and cfg.corrections_forward_only:
                    dv, clamped = kick_dv(imap_ret, x, v, v_fire, pass_leg, 1.0)
                else:
                    dv, clamped = kick_dv(imap, x, v, v_fire, pass_leg, cfg.kick_scale)
                rec.update({"x_delivered": x, "v_true_at_fire": v,
                            "v_bank_at_fire": v_fire, "dv_coil": dv,
                            "map_clamped": clamped, "t_fire_s": t_fire_actual * 1e-6})
                v = v + dv
                v_post = bank.v_post_shot(v_fire, cfg.gate_us, cfg.cans)
                t_last_fire = t_fire_actual
                res.n_shots += 1
                kicked = True
                if v <= 0.0:
                    reason = "the kick reversed the marble (dv %.3f at x = %.1f)" % (dv, x)
                    rec["v_out_fit"] = None
                    rec["dv_raw"] = None
                    res.kicks.append(rec)
                    break

        # -- segment 3: coast to the far edge, reading the out station ------
        seg3 = integrate_flat(losses, x, v, direction, channels, half, t0_us=t,
                              dt=cfg.dt, b_slope_scale=cfg.b_slope_scale,
                              b_conservative=cfg.b_conservative)
        x, v, t = seg3.x_mm, seg3.v_mps, seg3.t_us
        if rec is not None:
            out_read = (station_read(seg3.trips[out_st], specs[out_st], direction, half)
                        if len(seg3.trips[out_st]) >= 2 else None)
            rec["v_out_fit"] = out_read.v_fit if out_read else None
            rec["dv_raw"] = ((rec["v_out_fit"] - rec["v_in_fit"])
                             if (rec["v_out_fit"] and rec["v_in_fit"]) else None)
            res.kicks.append(rec)
        if kicked:
            awaited = "ret" if pass_leg == "fwd" else "fwd"
            if pass_leg == "fwd":
                res.cycles[-1]["kicked"] = True
        if seg3.reason == "stalled":
            reason = "marble stalled on the flat at x = %.1f mm" % x
            break
        if seg3.reason != "x_stop":
            reason = "marble did not reach the ramp (%s)" % seg3.reason
            break

        # -- excursion ------------------------------------------------------
        side = "far" if direction > 0 else "entry"
        v_back = excursion_return(losses, side, v)
        if v_back is None:
            reason = "marble parked on the %s ramp (left the flat at %.3f m/s)" % (side, v)
            break
        t += excursion_time_s(losses, v, v_back) * 1e6
        x = direction * FLAT_ZONE_X_MM
        v = v_back
        direction = -direction

    res.reason = reason or "unknown"
    res.t_end_s = t * 1e-6
    return res


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _f(value, spec="%.3f", none="--"):
    return none if value is None else spec % value


def print_summary(res, cfg=None, file=None):
    cfg = cfg or res.cfg
    out = file or sys.stdout
    p = lambda s="": print(s, file=out)  # noqa: E731
    p("  " + "=" * 58)
    p("  SUSTAIN twin: %d can(s) | on=%d us | offset %+.2f mm | ret %s | rule %s"
      % (cfg.cans, cfg.gate_us, cfg.fire_offset, str(cfg.ret_offset), cfg.timing_rule))
    p("  PSU %.1f V / %.2f A through %.0f ohm; refiring at %.2f V, floor %.2f V"
      % (cfg.psu_volts, cfg.psu_amps, cfg.charge_r,
         cfg.recharge_frac * res.v_start, cfg.floor_frac * res.v_start))
    p("  b_slope_scale %.2f | kick_scale %.2f | release %.3f m/s"
      % (cfg.b_slope_scale, cfg.kick_scale, cfg.release_v))
    p("  " + "=" * 58)
    n_fired = 0
    for k in res.kicks:
        tag = "A>" if k["leg"] == "fwd" else "<B"
        if k["skipped"]:
            p("  [%s pass skipped @ %.1fs: %s] v_in %s local %s"
              % (tag, k["t_s"], k["skip_reason"], _f(k["v_in_fit"]), _f(k["v_local"])))
            continue
        n_fired += 1
        p("  [%s kick %d/%d @ %.1fs] v_in %s (local %s%s) x %+.2f -> %+.2f mm"
          "%s bank %s -> %s V dv_coil %+.3f v_out %s raw %s"
          % (tag, n_fired, cfg.max_shots, k["t_fire_s"], _f(k["v_in_fit"]),
             _f(k["v_local"]),
             (", a %+.2f" % k["accel_mps2"]) if k["leg"] == "ret" and cfg.timing_rule == "quadratic" else "",
             k["x_target"], k["x_delivered"],
             (" trim %+.1f" % k["trim_mm"]) if k["leg"] == "ret" else "",
             _f(k["v_bank_at_release"], "%.1f"), _f(k["v_bank_at_fire"], "%.1f"),
             k["dv_coil"], _f(k["v_out_fit"]), _f(k["dv_raw"], "%+.3f")))
    p("  " + "-" * 58)
    p("  sustain ended: %s" % res.reason)
    p("  %d shots in %.1f s." % (res.n_shots, res.t_end_s))
    good = res.fwd_v_in()
    if len(good) >= 2:
        p("  v_in at A per cycle, m/s:")
        for i in range(0, len(good), 8):
            p("    " + "  ".join("%.3f" % v for v in good[i:i + 8]))
        lc = res.limit_cycle()
        p("  %s: %.3f -> %.3f m/s, %+.4f m/s per cycle over %d cycles."
          % (lc["trend"], good[0], good[-1], lc["drift"], len(good)))
        p("  limit cycle (last %d): %.3f m/s" % (lc["n_tail"], lc["v_in"]))
    far, entry = res.excursion_losses()
    for name, lst in (("far ramp", far), ("entry ramp", entry)):
        if lst:
            p("  %s excursion: %s mm/s lost (n=%d)"
              % (name, "/".join("%.0f" % (1000 * d) for d in lst), len(lst)))
    fwd = [k for k in res.fired if k["leg"] == "fwd"]
    if fwd:
        short = sum(k["x_target"] - k["x_delivered"] for k in fwd) / len(fwd)
        p("  forward kick delivered %.2f mm short of target on average (flat loss over the reach)"
          % short)


def _parse_ret_offset(s):
    return "auto" if str(s).lower() == "auto" else float(s)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    d = SustainConfig()
    parser.add_argument("--rig-profile", default="vbench_v0")
    parser.add_argument("--cans", type=int, default=d.cans)
    parser.add_argument("--gate", type=float, default=d.gate_us)
    parser.add_argument("--psu-amps", type=float, default=d.psu_amps)
    parser.add_argument("--psu-volts", type=float, default=d.psu_volts)
    parser.add_argument("--charge-r", type=float, default=d.charge_r)
    parser.add_argument("--fire-offset", type=float, default=d.fire_offset)
    parser.add_argument("--ret-offset", default="auto",
                        help="'auto' (K/v trim under linear, 0 under quadratic) or mm")
    parser.add_argument("--timing-rule", choices=("linear", "quadratic"),
                        default=d.timing_rule)
    parser.add_argument("--b-slope-scale", type=float, default=d.b_slope_scale)
    parser.add_argument("--b-conservative", action="store_true")
    parser.add_argument("--release-v", type=float, default=d.release_v,
                        help="ball speed entering the flat zone at the start, m/s")
    parser.add_argument("--recharge-frac", type=float, default=d.recharge_frac)
    parser.add_argument("--floor-frac", type=float, default=d.floor_frac)
    parser.add_argument("--cycles", type=int, default=d.cycles)
    parser.add_argument("--max-shots", type=int, default=d.max_shots)
    parser.add_argument("--max-seconds", type=float, default=d.max_seconds)
    parser.add_argument("--kick-scale", type=float, default=d.kick_scale)
    parser.add_argument("--shape-csv", default=d.shape_csv,
                        help="sweep_firepos CSV (offset_mm, coil_dv_mm_s) whose "
                             "normalised curve replaces the map's entry-side "
                             "shape (declared correction, see impulse_map."
                             "with_measured_shape)")
    parser.add_argument("--arm-latency-ms", type=float, default=d.arm_latency_ms)
    parser.add_argument("--loss-form", choices=("energy", "linear"), default=d.loss_form)
    parser.add_argument("--dt", type=float, default=d.dt)
    parser.add_argument("--impulse-map", type=Path, default=None,
                        help="default config/impulse_maps/impulse_map_<cans>can_<gate>us.json")
    parser.add_argument("--losses", type=Path, default=DEFAULT_LOSSES)
    parser.add_argument("--sweep-gates", default=None,
                        help="comma-separated gates; needs a map per gate")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="results/sustain_<tag>.json")
    parser.add_argument("--quiet", action="store_true")
    return parser


def config_from_args(args, gate_us=None):
    return SustainConfig(
        cans=args.cans, gate_us=float(gate_us if gate_us is not None else args.gate),
        psu_amps=args.psu_amps, psu_volts=args.psu_volts, charge_r=args.charge_r,
        fire_offset=args.fire_offset, ret_offset=_parse_ret_offset(args.ret_offset),
        timing_rule=args.timing_rule, b_slope_scale=args.b_slope_scale,
        b_conservative=args.b_conservative, release_v=args.release_v,
        recharge_frac=args.recharge_frac, floor_frac=args.floor_frac,
        cycles=args.cycles, max_shots=args.max_shots, max_seconds=args.max_seconds,
        kick_scale=args.kick_scale, shape_csv=args.shape_csv,
        arm_latency_ms=args.arm_latency_ms,
        loss_form=args.loss_form, dt=args.dt)



def load_shape_points(path, ref_offset=9.0, face_x=-22.78):
    """(x_mm, ratio_to_ref) from a sweep_firepos CSV, pairs averaged per offset."""
    import csv as _csv
    by = {}
    with open(path, newline="") as f:
        for r in _csv.DictReader(f):
            by.setdefault(float(r["offset_mm"]), []).append(float(r["coil_dv_mm_s"]))
    means = {o: sum(v) / len(v) for o, v in by.items()}
    ref = means[ref_offset]
    return [(face_x + o, means[o] / ref) for o in sorted(means)]


def main(argv=None):
    args = build_parser().parse_args(argv)
    profile = load_profile(ROOT, args.rig_profile)
    if not args.losses.exists():
        raise SystemExit(f"no loss table at {args.losses} (run fit_track_losses.py)")
    losses_doc = TrackLosses.load(args.losses)

    gates = ([float(g) for g in args.sweep_gates.split(",")] if args.sweep_gates
             else [args.gate])
    results = []
    for gate in gates:
        map_path = args.impulse_map if (args.impulse_map and not args.sweep_gates) \
            else default_map_path(args.cans, gate)
        if not map_path.exists():
            raise SystemExit(f"no impulse map at {map_path} (run impulse_map.py)")
        imap = load_map(map_path)
        imap_base = imap
        if args.shape_csv:
            imap = imap.with_measured_shape(load_shape_points(args.shape_csv))
            print(f"entry-side shape from {args.shape_csv} ({len(imap.shape_source)} points)")
        if imap.cans != args.cans:
            print(f"WARNING: map {map_path} is for {imap.cans} cans, run is {args.cans}")
        cfg = config_from_args(args, gate)
        losses = TrackLosses.from_dict(losses_doc.to_dict())
        res = run_sustain(profile, imap, losses, cfg, imap_ret=imap_base)
        if not args.quiet:
            print_summary(res, cfg)
        results.append((gate, map_path, res))

    if len(results) > 1:
        print()
        print("  %6s %8s %6s %8s %8s %8s" % ("gate", "cycle", "kicks", "slowret", "fwdkick", "end"))
        for gate, _, res in results:
            lc = res.limit_cycle()
            ret = [k["dv_raw"] for k in res.fired if k["leg"] == "ret" and k["dv_raw"] is not None]
            fwd = [k["dv_raw"] for k in res.fired if k["leg"] == "fwd" and k["dv_raw"] is not None]
            print("  %6.0f %8s %6d %8s %8s  %s" % (
                gate, _f(lc["v_in"]), res.n_shots,
                _f(sum(ret) / len(ret) if ret else None, "%+.3f"),
                _f(sum(fwd) / len(fwd) if fwd else None, "%+.3f"),
                res.reason))

    tag = args.tag or "%dcan_%dus_%gA" % (args.cans, gates[0], args.psu_amps)
    out = args.out or (ROOT / "results" / f"sustain_{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "profile": profile.name,
        "losses": {"path": str(args.losses).replace("\\", "/"),
                   "meta": losses_doc.meta},
        "runs": [{"gate_us": gate, "impulse_map": str(mp).replace("\\", "/"),
                  **res.to_dict()} for gate, mp, res in results],
    }
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
