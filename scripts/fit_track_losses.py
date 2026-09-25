"""Fit the track-loss table for the sustain twin from bench data.

Inputs:
  data/roll_baselines.csv                    no-kick rolls A -> B: v_in, dv over 204 mm
  ../omnimarble-vbench/logs/sustain_kicks.csv  one row per kick (host/sustainlog.py)

Output: config/track_losses.json (scripts/sustain_model.TrackLosses), with a
meta block recording n per fit, rms, the speed range each fit covers, the
csv sha256 and the date -- so a prediction can say what it was fitted on.

Four fits, in order, each through the model's OWN forward functions so the
simulator reproduces the fitted numbers by construction:

  flat      dv/dt = -(a0 + k v^2) on the three roll baselines.
  b_side    extra deceleration (g_B + k_B v^2) for -x travel on the B side,
            from the return passes' (v_local / v_fit) ratio at station B
            (a decelerating pass makes the fit read high) plus the no-kick
            return leg 0.643 -> 0.315 m/s (B fit -> A fit), weight 3.
  far       v_back^2 = alpha v_out^2 - beta between the forward kick's v_out
            at B and the next return kick's v_in at B, both mapped to the
            ball's true speed at the flat-zone EDGE through flat/b_side.
  entry     same between the return kick's v_out at A and the next forward
            v_in at A.

    uv run python scripts/fit_track_losses.py --kicks ../omnimarble-vbench/logs/sustain_kicks.csv
"""

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from sustain_model import (  # noqa: E402
    BSideExcess,
    Excursion,
    FLAT_ZONE_X_MM,
    FlatLoss,
    TrackLosses,
    integrate_flat,
    station_read,
)

DEFAULT_KICKS = ROOT.parent / "omnimarble-vbench" / "logs" / "sustain_kicks.csv"
DEFAULT_BASELINES = ROOT / "data" / "roll_baselines.csv"
DEFAULT_OUT = ROOT / "config" / "track_losses.json"
STATION_DISTANCE_M = 0.20412
NOKICK_LEG = (0.643, 0.315)     # 09-24 withheld return kick: B fit -> A fit
FIT_DT_S = 2e-3                 # coarse is fine: v changes ~1e-3 per step


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _num(value):
    if value is None:
        return None
    s = str(value).strip()
    if s in ("", "None", "nan", "-", "--"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_kicks(path):
    """Rows of sustain_kicks.csv with leg in {fwd, ret}, numbers parsed.

    Summary rows (no leg / no kick_idx) are dropped. Missing columns are
    tolerated as None so a schema that grows does not break the fit.
    """
    with open(path, newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    rows = []
    for r in raw:
        leg = (r.get("leg") or "").strip()
        if leg not in ("fwd", "ret"):
            continue
        idx = _num(r.get("kick_idx") if r.get("kick_idx") not in (None, "") else r.get("kick"))
        if idx is None:
            continue
        rows.append({
            "run_id": (r.get("run_id") or "").strip(),
            "on_us": _num(r.get("on_us")),
            "cans": _num(r.get("cans")),
            "leg": leg,
            "kick_idx": int(idx),
            "bank_pre": _num(r.get("bank_pre")),
            "v_in": _num(r.get("v_in")),
            "v_out": _num(r.get("v_out")),
            "dv": _num(r.get("dv")),
            "resid_us": _num(r.get("resid_us")),
            "v_local": _num(r.get("v_local")),
            "trim_mm": _num(r.get("trim_mm")),
        })
    return rows


def pair_cycles(rows):
    """{"far": [(fwd.v_out, ret.v_in)], "entry": [(ret.v_out, fwd.v_in)]}.

    Consecutive kicks within a run only (kick_idx i and i+1): a skipped pass
    between two kicks would put an unmeasured excursion in the pair.
    """
    far, entry = [], []
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run_id"], []).append(r)
    for run_rows in by_run.values():
        run_rows.sort(key=lambda r: r["kick_idx"])
        for a, b in zip(run_rows, run_rows[1:]):
            if b["kick_idx"] != a["kick_idx"] + 1:
                continue
            if a["leg"] == "fwd" and b["leg"] == "ret" and a["v_out"] and b["v_in"]:
                far.append((a["v_out"], b["v_in"]))
            if a["leg"] == "ret" and b["leg"] == "fwd" and a["v_out"] and b["v_in"]:
                entry.append((a["v_out"], b["v_in"]))
    return {"far": far, "entry": entry}


def load_baselines(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        v = _num(r.get("v_in_mps"))
        dv = _num(r.get("dv_mps"))
        if v is None or dv is None:
            continue
        out.append((v, dv))
    return out


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------
def _rms(values):
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


def fit_flat(baselines, distance_m=STATION_DISTANCE_M):
    """FlatLoss from (v_in, dv over distance_m) points; a0, k >= 0.

    Uses the closed-form v_after_distance, bounded least squares.
    """
    from scipy.optimize import least_squares

    def resid(p):
        flat = FlatLoss(p[0], p[1])
        return [flat.v_after_distance(v, distance_m) - v - dv for v, dv in baselines]

    v_mean = sum(v for v, _ in baselines) / len(baselines)
    k0 = max(1e-3, -sum(dv for _, dv in baselines) / len(baselines)
             / (v_mean * distance_m))
    sol = least_squares(resid, [0.0, k0], bounds=([0.0, 0.0], [10.0, 50.0]))
    flat = FlatLoss(sol.x[0], sol.x[1])
    meta = {"n": len(baselines), "rms_mps": _rms(list(sol.fun)),
            "v_range": [min(v for v, _ in baselines), max(v for v, _ in baselines)],
            "distance_m": distance_m}
    return flat, meta


def fit_excursion(pairs):
    """Excursion from (v_out, v_back) pairs: linear LS in (v_out^2, v_back^2).

    Also fits the linear alternative v_back = v_out + c0 + c1 v_out and
    stores it on the same object (selected by `form`).
    """
    n = len(pairs)
    if n < 2:
        raise ValueError("need at least two pairs to fit an excursion")
    xs = [a * a for a, _ in pairs]
    ys = [b * b for _, b in pairs]
    alpha, negbeta = _linfit(xs, ys)
    beta = -negbeta
    clamped = False
    if beta < 0.0:
        # A negative beta means the excursion would hand energy back at
        # zero speed -- unphysical, and what a narrow v_out cluster does to
        # a two-parameter fit. Refit alpha through the origin instead.
        beta = 0.0
        alpha = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
        clamped = True
    exc = Excursion(alpha, beta)
    c1p, c0 = _linfit([a for a, _ in pairs], [b for _, b in pairs])
    exc.c0 = c0
    exc.c1 = c1p - 1.0
    res = []
    for a, b in pairs:
        vb = exc.v_back(a)
        res.append((vb if vb is not None else 0.0) - b)
    meta = {"n": n, "rms_mps": _rms(res), "beta_clamped_to_zero": clamped,
            "v_out_range": [min(a for a, _ in pairs), max(a for a, _ in pairs)],
            "v_back_range": [min(b for _, b in pairs), max(b for _, b in pairs)]}
    return exc, meta


def _linfit(xs, ys):
    """(slope, intercept) least squares."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if den == 0.0:
        return 0.0, sy / n
    slope = (n * sxy - sx * sy) / den
    return slope, (sy - slope * sx) / n


class ReturnPassModel:
    """Forward function: a return pass from the far edge through B (and on to A).

    Everything the B-side fit and the pair-to-edge conversions need, built on
    integrate_flat/station_read so the simulator and the fitter agree.
    """

    def __init__(self, specs, halfwidth_mm, dt=FIT_DT_S, b_slope_scale=1.0,
                 trigger_in="A", trigger_out="B"):
        self.specs = specs
        self.half = float(halfwidth_mm)
        self.dt = dt
        self.b_slope_scale = b_slope_scale
        self.A = trigger_in
        self.B = trigger_out
        self.channels = {n: s["channel_x_mm"] for n, s in specs.items()}

    def read_B(self, losses, v_edge):
        seg = integrate_flat(losses, FLAT_ZONE_X_MM, v_edge, -1, self.channels,
                             self.half, dt=self.dt, stop_after_station=self.B,
                             b_slope_scale=self.b_slope_scale)
        if seg.reason != "station":
            return None
        return station_read(seg.trips[self.B], self.specs[self.B], -1, self.half)

    def read_B_and_A(self, losses, v_edge):
        seg = integrate_flat(losses, FLAT_ZONE_X_MM, v_edge, -1, self.channels,
                             self.half, dt=self.dt, stop_after_station=self.A,
                             b_slope_scale=self.b_slope_scale)
        if seg.reason != "station":
            return None, None
        return (station_read(seg.trips[self.B], self.specs[self.B], -1, self.half),
                station_read(seg.trips[self.A], self.specs[self.A], -1, self.half))

    def v_edge_for_B_fit(self, losses, v_fit_target, tol=1e-4, max_iter=40):
        """Edge speed whose return pass reads v_fit_target at B.

        Bracketed bisection on a MONOTONE function: a faster edge speed always
        reads faster at B, and a ball that stalls before B reads as "too slow".
        The secant this replaced started at 1.05-1.4x the target, which for a
        slow ball under a real B-side loss is below the stall speed, so the
        fit scored every pass under ~0.22 m/s as a stall (2026-09-24).
        """
        def read(ve):
            r = self.read_B(losses, ve)
            return r.v_fit if (r is not None and r.v_fit) else None

        lo, hi = v_fit_target, v_fit_target * 1.5
        # grow hi until the pass reads faster than the target
        for _ in range(12):
            f = read(hi)
            if f is not None and f >= v_fit_target:
                break
            lo, hi = hi, hi * 1.5
        else:
            return hi
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            f = read(mid)
            if f is None or f < v_fit_target:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return hi


def fit_b_side(ret_rows, flat, nokick=NOKICK_LEG, weight=3.0, specs=None,
               halfwidth_mm=5.2, dt=FIT_DT_S, ratio_only=False):
    """BSideExcess (g_B, k_B >= 0) from return-pass station reads.

    ret_rows: dicts with v_in (the B fit) and v_local; rows lacking either
    are ignored. For each row the edge speed is solved so the model's B fit
    matches the row's, and the residual is the model's v_local minus the
    measured one. The no-kick leg (v_fit_B, v_fit_A) adds a residual on the
    A fit with the given weight -- it is the only direct measurement of the
    whole B-side stretch. Uses scipy.optimize.least_squares.
    """
    from scipy.optimize import least_squares

    if specs is None:
        specs = _default_specs()
    model = ReturnPassModel(specs, halfwidth_mm, dt=dt)
    rows = [r for r in ret_rows if r.get("v_in") and r.get("v_local")]

    def losses_for(p):
        return TrackLosses(flat=flat, b_side=BSideExcess(p[0], p[1]))

    def resid(p):
        losses = losses_for(p)
        out = []
        for r in rows:
            ve = model.v_edge_for_B_fit(losses, r["v_in"])
            rd = model.read_B(losses, ve)
            out.append(((rd.v_local if rd and rd.v_local else 0.0) - r["v_local"]))
        if nokick is not None and not ratio_only and weight > 0.0:
            out.append(weight * (nokick_pred(losses) - nokick[1]))
        return out

    def nokick_pred(losses):
        """Model's v_fit at A for the no-kick return leg entering at nokick[0]."""
        ve = model.v_edge_for_B_fit(losses, nokick[0])
        _, rA = model.read_B_and_A(losses, ve)
        return rA.v_fit if rA is not None and rA.v_fit else 0.0

    if not rows and (nokick is None or ratio_only):
        raise ValueError("nothing to fit the B-side excess on")
    sol = least_squares(resid, [0.5, 0.5], bounds=([0.0, 0.0], [20.0, 50.0]),
                        diff_step=[0.05, 0.05], xtol=1e-4, ftol=1e-5)
    b = BSideExcess(sol.x[0], sol.x[1])
    res = list(sol.fun)
    fitted = losses_for(sol.x)
    nk_pred = nokick_pred(fitted) if nokick is not None else None
    meta = {"n_rows": len(rows), "nokick": list(nokick) if nokick else None,
            "nokick_weight": weight,
            "rms_v_local_mps": _rms(res[:len(rows)]) if rows else None,
            # the single no-kick leg, scored against the fit (a constraint only
            # when weight > 0): model's v_fit at A vs the observed value
            "nokick_pred_vA_mps": nk_pred,
            "nokick_resid_mps": (nk_pred - nokick[1]) if nokick is not None else None,
            "v_fit_range": ([min(r["v_in"] for r in rows), max(r["v_in"] for r in rows)]
                            if rows else None)}
    return b, meta


def _default_specs():
    from rig_profile import load_profile
    sys.path.insert(0, str(ROOT / "source" / "extensions" / "omni.marble.coaster"
                           / "omni" / "marble" / "coaster"))
    profile = load_profile(ROOT, "vbench_v0")
    return _named_specs(profile)


def _named_specs(profile):
    specs = profile.station_specs()
    for name, s in specs.items():
        s["name"] = name
    return specs


def pairs_to_edge(pairs, side, losses, specs, halfwidth_mm=5.2, dt=FIT_DT_S):
    """Map station-read pairs to the ball's true speeds at the flat-zone edge.

    far:   (fwd v_out fit at B, ret v_in fit at B) -> (v at +edge leaving,
           v at +edge returning). Leaving: flat loss over the 69 mm from B's
           centre to the edge. Returning: solved through the B-side model.
    entry: (ret v_out fit at A, fwd v_in fit at A) -> (v at -edge leaving,
           v at -edge returning), flat loss both ways (A is on the flat side).
    """
    model = ReturnPassModel(specs, halfwidth_mm, dt=dt)
    centre = abs(float(specs["B" if side == "far" else "A"]["channel_x_mm"][2]))
    d_m = (FLAT_ZONE_X_MM - centre) / 1000.0
    out = []
    for v_out_fit, v_back_fit in pairs:
        v_edge_out = losses.flat.v_after_distance(v_out_fit, d_m)
        if side == "far":
            v_edge_back = model.v_edge_for_B_fit(losses, v_back_fit)
        else:
            v_edge_back = losses.flat.v_before_distance(v_back_fit, d_m)
        out.append((v_edge_out, v_edge_back))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def fit_all(kick_rows, baselines, specs, halfwidth_mm=5.2, dt=FIT_DT_S,
            nokick=NOKICK_LEG, weight=3.0):
    flat, m_flat = fit_flat(baselines)
    ret_rows = [r for r in kick_rows if r["leg"] == "ret"]
    b_side, m_b = fit_b_side(ret_rows, flat, nokick=nokick, weight=weight,
                             specs=specs, halfwidth_mm=halfwidth_mm, dt=dt)
    partial = TrackLosses(flat=flat, b_side=b_side)
    pairs = pair_cycles(kick_rows)
    far_edge = pairs_to_edge(pairs["far"], "far", partial, specs, halfwidth_mm, dt)
    entry_edge = pairs_to_edge(pairs["entry"], "entry", partial, specs, halfwidth_mm, dt)
    far, m_far = fit_excursion(far_edge)
    entry, m_entry = fit_excursion(entry_edge)
    meta = {"flat": m_flat, "b_side": m_b, "far": m_far, "entry": m_entry}
    return TrackLosses(flat=flat, b_side=b_side, far=far, entry=entry, meta=meta), meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kicks", type=Path, default=DEFAULT_KICKS)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rig-profile", default="vbench_v0")
    parser.add_argument("--dt", type=float, default=FIT_DT_S)
    parser.add_argument("--nokick", default="%g,%g" % NOKICK_LEG,
                        help="v_fit at B, v_fit at A of the no-kick return leg; "
                             "'none' to drop the constraint")
    parser.add_argument("--nokick-weight", type=float, default=0.0,
                        help="weight of the single no-kick leg in the B-side "
                             "fit; 0 (default) reports it as a held-out check "
                             "-- 15 return passes read a speed-independent "
                             "v_local/v_fit ratio that no constant-g term "
                             "can give, and this one leg pulled g to 1.6 m/s2")
    parser.add_argument("--run-id", default=None,
                        help="fit only these run_ids (comma-separated; default: every run in the csv)")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT / "source" / "extensions" / "omni.marble.coaster"
                           / "omni" / "marble" / "coaster"))
    from rig_profile import load_profile
    profile = load_profile(ROOT, args.rig_profile)
    specs = _named_specs(profile)
    half = float(profile.sensing.get("detect_halfwidth_mm", 0.0))

    if not args.kicks.exists():
        raise SystemExit(f"no kick table at {args.kicks} (run host/sustainlog.py first)")
    rows = load_kicks(args.kicks)
    if args.run_id:
        wanted = set(x.strip() for x in args.run_id.split(","))
        rows = [r for r in rows if r["run_id"] in wanted]
    baselines = load_baselines(args.baselines)
    nokick = None if args.nokick.lower() == "none" else tuple(
        float(v) for v in args.nokick.split(","))
    print(f"{len(rows)} kick rows, {len(baselines)} roll baselines")

    losses, meta = fit_all(rows, baselines, specs, half, args.dt, nokick,
                           args.nokick_weight)
    losses.ramp_angle_deg = float(profile.track.get("ramp_angle_deg", 55.0))
    losses.meta.update({
        "kicks_csv": {"path": str(args.kicks).replace("\\", "/"),
                      "sha256": sha256_of(args.kicks)},
        "baselines_csv": {"path": str(args.baselines).replace("\\", "/"),
                          "sha256": sha256_of(args.baselines)},
        "date": _dt.date.today().isoformat(),
        "profile": profile.name,
        "run_id": args.run_id,
    })
    losses.save(args.out)
    print(json.dumps(losses.to_dict(), indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
