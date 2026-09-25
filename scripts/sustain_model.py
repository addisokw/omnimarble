"""Shared 1-D physics for the closed-loop sustain twin.

Pure Python on purpose: the loss fitter (scripts/fit_track_losses.py), the
cycle simulator (scripts/simulate_sustain.py) and the tests all import this,
and none of them should need torch or numpy to answer "what will the next
sustain run do". scipy is allowed only inside the fitter.

Coordinates follow the rig: x in mm along the track, coil-centred, station A
at negative x and station B at positive x; the flat zone spans +/-171.34 mm
and the ramps lie beyond it. Speeds are in m/s, accelerations in m/s^2,
timestamps in microseconds (the firmware's ticks_us), so a recorded pass can
be replayed through the same code.

What lives here:

  FlatLoss / BSideExcess / Excursion / TrackLosses  -- the loss model the
      fitter writes to config/track_losses.json and the simulator reads.
  integrate_flat()   -- one segment of coasting on the flat, returning the
      station channel trips exactly as the rig would latch them (leading
      edge, half-width upstream of the sensor).
  station_read()     -- a coil_sensing.VirtualStation built from those trips,
      so the twin reads the SAME fit/local velocities the firmware does,
      biases included (the return fit reads high on a decelerating pass).
  fire_delay_us()    -- the firmware's fire-time rules: `linear` (constant
      velocity) and `quadratic` (the Part A kinematic predictor: centred,
      scaled quadratic fit, stable-root transit, stall branch). This MUST stay
      the same math as firmware/sensors.py fit_kinematics/predict_transit_us.
  return_trim_mm()   -- today's empirical K/v + gate-term trim, kept so the
      twin can reproduce the 2026-09-24 runs before the predictor ships.
  BankModel          -- CC-then-RC recharge with closed forms, and the
      post-shot voltage from the captures' V(cut)/V(on) ratios.
  kick_dv()          -- the kick as an instantaneous dv from an impulse map,
      mirrored on the return and rescaled by (V_fire/V_map)^2.
"""

import json
import math
from pathlib import Path

# The Kit extension package is where the rig's estimator lives; scripts/ and
# tests/ both put it on sys.path (tests/conftest.py), and this module is
# imported from scripts/ so the same insertion is repeated here defensively.
import sys as _sys

_ROOT = Path(__file__).resolve().parent.parent
_COASTER = (_ROOT / "source" / "extensions" / "omni.marble.coaster" / "omni"
            / "marble" / "coaster")
if str(_COASTER) not in _sys.path:
    _sys.path.insert(0, str(_COASTER))

from coil_sensing import VirtualStation, interpolate_crossing_us  # noqa: E402

GRAVITY_MPS2 = 9.81
COIL_FACE_X_MM = 22.78
FLAT_ZONE_X_MM = 171.34
STATION_SPACING_MM = 204.12          # A centre to B centre
DEFAULT_DT_S = 5e-4                  # integrator step; velocity changes ~1e-4 per step

# Small-signal (100 Hz LCR) bank capacitance per can count, vbench
# firmware/config.py BANK_*CAN_UF_100HZ. The recharge is a slow (RC ~0.16 s)
# process, so the small-signal value is the right one here -- not the pulse
# value the discharge model uses. A profile entry `capacitance_100hz_uF` in
# pulse_measured_by_cans (Part C5) outranks this table.
BANK_C_100HZ_UF = {
    1: 1909.0,      # per-can nominal small-signal; no 1-can sustain has been run
    2: 3767.5,
    3: 5586.0,
    4: 7392.0,
    5: 9140.0,
}

# V(cut)/V(on) measured from the blank captures at 49 V, by gate. The bank's
# post-shot voltage is what the recharge starts from, so this sets the cycle
# time budget. Between gates the ratio is interpolated linearly.
BANK_RETENTION = {
    4: {100: 0.960, 200: 0.895, 300: 0.813, 500: 0.661, 700: 0.534,
        1000: 0.390, 1500: 0.242},
    5: {100: 0.969, 200: 0.915, 300: 0.852, 500: 0.720, 700: 0.608,
        1000: 0.469, 1500: 0.314},
}
# 3 cans: one point (700 us -> 0.41); other gates follow the 4-can gate shape
# scaled to pass through it.
BANK_RETENTION_3CAN_700 = 0.41


# ---------------------------------------------------------------------------
# loss model
# ---------------------------------------------------------------------------
class FlatLoss:
    """dv/dt = -(a0 + k v^2) on the flat, both directions.

    a0 in m/s^2 (rolling resistance, ~0), k in 1/m (drag-like). Roll
    baselines over the 204 mm between stations: -0.034 @0.206, -0.050 @0.245,
    -0.113 @0.79 m/s.
    """

    def __init__(self, a0=0.0, k=0.0):
        self.a0 = float(a0)
        self.k = float(k)

    def decel(self, v):
        return self.a0 + self.k * v * v

    def v_after_distance(self, v_in, distance_m):
        """Closed form: a0 + k v^2 decays as exp(-2 k s) with distance.

        Returns 0.0 if the ball would stop inside the distance.
        """
        return _v_after_distance(self.a0, self.k, v_in, distance_m)

    def v_before_distance(self, v_out, distance_m):
        """Inverse of v_after_distance: the speed distance_m upstream."""
        if distance_m <= 0.0:
            return float(v_out)
        if self.k > 1e-12:
            e2 = (self.a0 + self.k * v_out * v_out) * math.exp(2.0 * self.k * distance_m) - self.a0
            return math.sqrt(e2 / self.k)
        return math.sqrt(v_out * v_out + 2.0 * self.a0 * distance_m)

    def to_dict(self):
        return {"a0_mps2": self.a0, "k_per_m": self.k}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("a0_mps2", 0.0), d.get("k_per_m", 0.0))


def _v_after_distance(a0, k, v_in, distance_m):
    if distance_m <= 0.0:
        return float(v_in)
    if k > 1e-12:
        e2 = (a0 + k * v_in * v_in) * math.exp(-2.0 * k * distance_m) - a0
        return math.sqrt(e2 / k) if e2 > 0.0 else 0.0
    e2 = v_in * v_in - 2.0 * a0 * distance_m
    return math.sqrt(e2) if e2 > 0.0 else 0.0


class BSideExcess:
    """Extra deceleration for travel toward -x between x_from and x_to.

    The return pass decelerates ~15% through station B on every crossing
    (the B-side approach is not the mirror of A's). Modelled as an extra
    g + k v^2 acting on the B side, scaled by `b_slope_scale` at run time
    (1 = as built, 0 = levelled). Dissipative by default: it acts only on
    travel toward -x. With `conservative=True` the g term acts as a slope
    instead -- it slows travel toward -x and SPEEDS travel toward +x by the
    same amount, which is the other bound on what levelling would buy.
    """

    def __init__(self, g=0.0, k=0.0, x_from=COIL_FACE_X_MM, x_to=FLAT_ZONE_X_MM):
        self.g = float(g)
        self.k = float(k)
        self.x_from = float(x_from)
        self.x_to = float(x_to)

    def inside(self, x_mm):
        return self.x_from <= x_mm <= self.x_to

    def to_dict(self):
        return {"g_mps2": self.g, "k_per_m": self.k,
                "x_from_mm": self.x_from, "x_to_mm": self.x_to}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("g_mps2", 0.0), d.get("k_per_m", 0.0),
                   d.get("x_from_mm", COIL_FACE_X_MM),
                   d.get("x_to_mm", FLAT_ZONE_X_MM))


class Excursion:
    """Speed transfer over a ramp excursion, applied at the flat-zone edge.

    Energy form (default): v_back^2 = alpha v_out^2 - beta; parks (returns
    None) if the right-hand side is <= 0. Linear alternative behind
    `form="linear"`: v_back = v_out + c0 + c1 v_out, parks if <= 0.
    """

    def __init__(self, alpha=1.0, beta=0.0, form="energy", c0=0.0, c1=0.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.form = form
        self.c0 = float(c0)
        self.c1 = float(c1)

    def v_back(self, v_out):
        if self.form == "linear":
            vb = v_out + self.c0 + self.c1 * v_out
            return vb if vb > 0.0 else None
        e2 = self.alpha * v_out * v_out - self.beta
        return math.sqrt(e2) if e2 > 0.0 else None

    def to_dict(self):
        return {"alpha": self.alpha, "beta_m2ps2": self.beta, "form": self.form,
                "c0_mps": self.c0, "c1": self.c1}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("alpha", 1.0), d.get("beta_m2ps2", 0.0),
                   d.get("form", "energy"), d.get("c0_mps", 0.0),
                   d.get("c1", 0.0))


class TrackLosses:
    """The fitted loss table: flat, B-side excess, far and entry excursions."""

    def __init__(self, flat=None, b_side=None, far=None, entry=None, meta=None,
                 ramp_angle_deg=55.0):
        self.flat = flat or FlatLoss()
        self.b_side = b_side or BSideExcess()
        self.far = far or Excursion()
        self.entry = entry or Excursion()
        self.meta = dict(meta or {})
        self.ramp_angle_deg = float(ramp_angle_deg)

    def to_dict(self):
        return {
            "schema": "track_losses_v1",
            "flat": self.flat.to_dict(),
            "b_side": self.b_side.to_dict(),
            "far": self.far.to_dict(),
            "entry": self.entry.to_dict(),
            "ramp_angle_deg": self.ramp_angle_deg,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(FlatLoss.from_dict(d.get("flat", {})),
                   BSideExcess.from_dict(d.get("b_side", {})),
                   Excursion.from_dict(d.get("far", {})),
                   Excursion.from_dict(d.get("entry", {})),
                   d.get("meta", {}), d.get("ramp_angle_deg", 55.0))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n",
                        encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def decel_mps2(losses, x_mm, v_mps, direction, b_slope_scale=1.0,
               b_conservative=False):
    """Deceleration along the direction of travel (positive = slowing).

    `direction` is +1 for travel toward +x, -1 toward -x; v_mps is the speed
    magnitude. The B-side excess applies to -x travel inside its window; in
    conservative mode its g term reverses sign for +x travel (a slope helps
    the ball on the way out as much as it hurts on the way back).
    """
    d = losses.flat.decel(v_mps)
    b = losses.b_side
    if b_slope_scale and b.inside(x_mm):
        if direction < 0:
            d += b_slope_scale * (b.g + b.k * v_mps * v_mps)
        elif b_conservative:
            d -= b_slope_scale * b.g
    return d


def excursion_return(losses, side, v_out):
    """Speed re-entering the flat zone after the far ("far") or entry ramp."""
    exc = losses.far if side == "far" else losses.entry
    return exc.v_back(v_out)


def excursion_time_s(losses, v_out, v_back):
    """Time spent beyond the flat-zone edge: up the ramp and back.

    A rolling ball on a ramp at angle theta decelerates at (5/7) g sin theta,
    so the excursion takes v_out/a up and v_back/a down. The ramp transition
    itself is folded into the fitted alpha/beta, not timed separately.
    """
    a = (5.0 / 7.0) * GRAVITY_MPS2 * math.sin(math.radians(losses.ramp_angle_deg))
    return (v_out + (v_back or 0.0)) / a


# ---------------------------------------------------------------------------
# flat-zone integration and the station read
# ---------------------------------------------------------------------------
class FlatSegment:
    """Result of integrate_flat: end state, trips, and x(t)/v(t) samplers."""

    def __init__(self, x_mm, v_mps, t_us, direction, trips, ts, xs, vs, reason):
        self.x_mm = x_mm
        self.v_mps = v_mps
        self.t_us = t_us
        self.direction = direction
        self.trips = trips              # {station: [(idx, t_us), ...] in time order}
        self.ts = ts
        self.xs = xs
        self.vs = vs
        self.reason = reason            # "x_stop" | "t_stop" | "station" | "stalled" | "max_time"

    @property
    def stalled(self):
        return self.reason == "stalled"

    def _bracket(self, t_us):
        ts = self.ts
        if t_us <= ts[0]:
            return 0, 0, 0.0
        if t_us >= ts[-1]:
            return len(ts) - 1, len(ts) - 1, 0.0
        lo, hi = 0, len(ts) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ts[mid] <= t_us:
                lo = mid
            else:
                hi = mid
        f = (t_us - ts[lo]) / (ts[hi] - ts[lo])
        return lo, hi, f

    def x_at(self, t_us):
        lo, hi, f = self._bracket(t_us)
        return self.xs[lo] + f * (self.xs[hi] - self.xs[lo])

    def v_at(self, t_us):
        lo, hi, f = self._bracket(t_us)
        return self.vs[lo] + f * (self.vs[hi] - self.vs[lo])


def integrate_flat(losses, x0_mm, v0_mps, direction, channels, halfwidth_mm,
                   t0_us=0.0, dt=DEFAULT_DT_S, x_stop_mm=None, t_stop_us=None,
                   stop_after_station=None, b_slope_scale=1.0,
                   b_conservative=False, max_time_s=20.0):
    """Coast along the flat from (x0, v0) in `direction` until a stop condition.

    channels: {station_name: [channel_x_mm, ...]} in the station's index
    order. A channel trips when the ball's LEADING EDGE reaches it, i.e. its
    centre is at ch_x - direction*halfwidth -- the same convention as
    simulate_rig_shot.simulate_shot, and the reason the reach constants carry
    the half-width.

    Stops at the first of: x passing x_stop_mm (default: the flat-zone edge
    ahead), t reaching t_stop_us, `stop_after_station` having tripped all its
    channels, the ball stalling (v <= 0), or max_time_s. Midpoint integration;
    trips are interpolated within the step so dt does not quantise them.
    """
    if x_stop_mm is None:
        x_stop_mm = direction * FLAT_ZONE_X_MM
    x = float(x0_mm)
    v = float(v0_mps)
    t = float(t0_us)
    trips = {name: [] for name in channels}
    got = {name: [False] * len(chs) for name, chs in channels.items()}
    trip_x = {name: [cx - direction * halfwidth_mm for cx in chs]
              for name, chs in channels.items()}
    ts, xs, vs = [t], [x], [v]
    dt_us = dt * 1e6
    steps = int(max_time_s / dt)
    reason = "max_time"

    # Already at or past the stop?
    if (x - x_stop_mm) * direction >= 0.0:
        return FlatSegment(x, v, t, direction, trips, ts, xs, vs, "x_stop")
    if t_stop_us is not None and t >= t_stop_us:
        return FlatSegment(x, v, t, direction, trips, ts, xs, vs, "t_stop")
    if v <= 0.0:
        return FlatSegment(x, 0.0, t, direction, trips, ts, xs, vs, "stalled")

    for _ in range(steps):
        h = dt
        h_us = dt_us
        if t_stop_us is not None and t + h_us > t_stop_us:
            h_us = t_stop_us - t
            h = h_us * 1e-6
            if h <= 0.0:
                reason = "t_stop"
                break
        # midpoint step in speed; position from the mid-speed
        d1 = decel_mps2(losses, x, v, direction, b_slope_scale, b_conservative)
        v_mid = v - 0.5 * h * d1
        if v_mid <= 0.0:
            v_new = 0.0
        else:
            x_mid = x + direction * v_mid * 1000.0 * 0.5 * h
            d2 = decel_mps2(losses, x_mid, v_mid, direction, b_slope_scale,
                            b_conservative)
            v_new = v - h * d2
        if v_new <= 0.0:
            v_new = 0.0
        x_new = x + direction * 0.5 * (v + v_new) * 1000.0 * h
        t_new = t + h_us

        # sensing, interpolated within the step
        for name, txs in trip_x.items():
            g = got[name]
            for idx, tx in enumerate(txs):
                if g[idx]:
                    continue
                if (x < tx <= x_new) or (x_new <= tx < x):
                    tc = interpolate_crossing_us(x, x_new, tx, t_new, h_us)
                    if tc is not None:
                        g[idx] = True
                        trips[name].append((idx, tc))

        x, v, t = x_new, v_new, t_new
        ts.append(t)
        xs.append(x)
        vs.append(v)

        if v <= 0.0:
            reason = "stalled"
            break
        if (x - x_stop_mm) * direction >= 0.0:
            reason = "x_stop"
            break
        if t_stop_us is not None and t >= t_stop_us - 1e-9:
            reason = "t_stop"
            break
        if (stop_after_station is not None
                and all(got[stop_after_station])):
            reason = "station"
            break

    return FlatSegment(x, v, t, direction, trips, ts, xs, vs, reason)


class StationRead:
    """What the firmware would know after a pass through one station."""

    def __init__(self, station, ticks, direction, halfwidth_mm):
        self.station = station
        self.ticks = list(ticks)               # [(idx, t_us)] in time order
        self.direction = direction
        self.halfwidth_mm = halfwidth_mm
        self.n = len(ticks)
        self.pitch_mm = station.pitch_mm
        self.v_fit = station.velocity_mps()
        self.residual_us = station.residual_us()
        self.t_last = station.last_tick()
        x_last = station.last_channel_x_mm()
        self.x_last_centre = (None if x_last is None
                              else x_last - direction * halfwidth_mm)
        self.v_local = None
        if self.n >= 2:
            (i1, t1), (i2, t2) = self.ticks[-2], self.ticks[-1]
            g = t2 - t1
            if g > 0 and i1 != i2:
                self.v_local = abs(i2 - i1) * (self.pitch_mm / 1000.0) / (g / 1e6)

    @property
    def complete(self):
        return self.station.complete()


def station_read(trips, spec, direction, halfwidth_mm):
    """Build a VirtualStation from trips and read it the way the firmware does.

    spec is one entry of RigProfile.station_specs(); trips is the list of
    (idx, t_us) integrate_flat produced for that station (any order).
    """
    station = VirtualStation(spec.get("name", "?"), spec["channel_x_mm"],
                             spec["pitch_mm"], rev=spec.get("order_rev", False))
    station.detect_halfwidth_mm = halfwidth_mm
    ticks = sorted(trips, key=lambda p: p[1])
    for idx, t_us in ticks:
        station.record(idx, t_us)
    return StationRead(station, ticks, direction, halfwidth_mm)


# ---------------------------------------------------------------------------
# fire timing -- the same math as the firmware predictor (Part A)
# ---------------------------------------------------------------------------
def _solve3(S, b):
    """3x3 normal equations by pivoted elimination (host/rolllog.py _quadfit)."""
    M = [row[:] + [bi] for row, bi in zip(S, b)]
    for c in range(3):
        p = max(range(c, 3), key=lambda k: abs(M[k][c]))
        M[c], M[p] = M[p], M[c]
        if M[c][c] == 0:
            return None
        for k in range(3):
            if k != c:
                f = M[k][c] / M[c][c]
                for j in range(c, 4):
                    M[k][j] -= f * M[c][j]
    return [M[i][3] / M[i][i] for i in range(3)]


def fit_kinematics(ticks, pitch_mm):
    """(v_last_mps, a_mps2, resid_us) from crossings, or None.

    Distance along the direction of travel s_k = |i_k - i_first| * pitch, so
    v is positive on both legs and a < 0 means decelerating. Time is centred
    and span-scaled, u = (t - t_first)/T - 0.5, and the fit is
    s = c0 + c1 u + c2 u^2: v_last = ds/dt at u = +0.5 = (c1 + c2)/T,
    a = 2 c2 / T^2. n = 2 is an exact line (a = 0), n = 3 an exact parabola,
    n >= 4 leaves a residual (RMS position residual expressed in us at v_last).
    """
    ticks = sorted(ticks, key=lambda p: p[1])
    n = len(ticks)
    if n < 2:
        return None
    i0, t0 = ticks[0]
    T = ticks[-1][1] - t0
    if T <= 0:
        return None
    T_s = T * 1e-6
    s = [abs(i - i0) * pitch_mm / 1000.0 for i, _ in ticks]
    if n == 2:
        v = s[1] / T_s
        return (v, 0.0, 0.0)
    u = [(t - t0) / T - 0.5 for _, t in ticks]
    S = [[sum(ui ** (p + q) for ui in u) for q in range(3)] for p in range(3)]
    b = [sum(ui ** p * si for ui, si in zip(u, s)) for p in range(3)]
    c = _solve3(S, b)
    if c is None:
        return None
    c0, c1, c2 = c
    v_last = (c1 + c2) / T_s
    a = 2.0 * c2 / (T_s * T_s)
    resid_us = 0.0
    if n >= 4 and v_last > 0:
        ss = 0.0
        for ui, si in zip(u, s):
            r = si - (c0 + c1 * ui + c2 * ui * ui)
            ss += r * r
        resid_us = math.sqrt(ss / n) / v_last * 1e6
    return (v_last, a, resid_us)


def predict_transit_us(reach_mm, v_mps, a_mps2):
    """(transit_us, stalled): time for the ball to cover reach_mm from v, a.

    Stable root tau = 2R / (v + sqrt(v^2 + 2 a R)) -- reduces to R/v at a = 0
    with no small-a guard. If the discriminant is negative the ball stalls
    short of the reach: fire at the stall time -v/a and flag it (the coil
    pulls a ball that is still upstream; the safe error direction).
    """
    R = reach_mm / 1000.0
    if v_mps <= 0.0:
        return (float("inf"), True)
    disc = v_mps * v_mps + 2.0 * a_mps2 * R
    if disc < 0.0:
        return (-v_mps / a_mps2 * 1e6, True)
    return (2.0 * R / (v_mps + math.sqrt(disc)) * 1e6, False)


def fire_delay_us(rule, read, reach_mm, v_linear=None, max_extra_mm=16.0):
    """Delay after the last crossing at which to fire. Returns (delay_us, info).

    rule="linear": reach / v with v = v_linear if given (the forward leg
    times on the fit velocity) else read.v_local (the return leg today).
    rule="quadratic": the Part A predictor on the station's own crossings,
    clamped to at most max_extra_mm beyond the constant-velocity answer
    (None = no clamp). info carries v_used, a, stalled, clamped, linear_us.
    """
    if read.n < 2:
        raise ValueError("cannot time a shot from fewer than two crossings")
    v_lin = v_linear if v_linear else read.v_local
    if not v_lin or v_lin <= 0.0:
        raise ValueError("no usable velocity for the linear rule")
    linear_us = reach_mm / 1000.0 / v_lin * 1e6
    if rule == "linear":
        return linear_us, {"rule": rule, "v_used": v_lin, "a": 0.0,
                           "stalled": False, "clamped": False,
                           "linear_us": linear_us}
    if rule != "quadratic":
        raise ValueError(f"unknown timing rule {rule!r}")
    kin = fit_kinematics(read.ticks, read.pitch_mm)
    if kin is None:
        return linear_us, {"rule": "linear-fallback", "v_used": v_lin, "a": 0.0,
                           "stalled": False, "clamped": False,
                           "linear_us": linear_us}
    v_last, a, resid = kin
    transit, stalled = predict_transit_us(reach_mm, v_last, a)
    clamped = False
    if max_extra_mm is not None:
        cap = (reach_mm + max_extra_mm) / 1000.0 / v_last * 1e6
        if transit > cap:
            transit, clamped = cap, True
    return transit, {"rule": rule, "v_used": v_last, "a": a, "stalled": stalled,
                     "clamped": clamped, "linear_us": linear_us,
                     "kin_resid_us": resid,
                     "later_mm": (transit - linear_us) * 1e-6 * v_last * 1000.0}


def return_trim_mm(v_local, on_us, K=1.7, gate_mm_per_us=-0.004, ref_us=700,
                   max_mm=16.0):
    """Today's firmware return trim: K / v_local plus a gate term, clamped.

    Reproduces SUSTAIN_RET_TRIM_* as of 2026-09-24 so the twin can be scored
    against those runs; the quadratic rule replaces it.
    """
    gate_mm = gate_mm_per_us * ((on_us or ref_us) - ref_us)
    return min(max_mm, K / max(v_local, 0.02) + gate_mm)


# ---------------------------------------------------------------------------
# bank
# ---------------------------------------------------------------------------
def bank_retention(cans, gate_us):
    """V(cut)/V(on) for a gate, interpolated in gate from the capture table."""
    if cans >= 5:
        table = BANK_RETENTION[5]
        scale = 1.0
    elif cans == 4:
        table = BANK_RETENTION[4]
        scale = 1.0
    else:
        table = BANK_RETENTION[4]
        scale = BANK_RETENTION_3CAN_700 / BANK_RETENTION[4][700]
    gates = sorted(table)
    g = float(gate_us)
    if g <= gates[0]:
        r = table[gates[0]]
    elif g >= gates[-1]:
        # extrapolate on the last segment, floored at zero
        g0, g1 = gates[-2], gates[-1]
        r = table[g1] + (table[g1] - table[g0]) / (g1 - g0) * (g - g1)
    else:
        for g0, g1 in zip(gates, gates[1:]):
            if g0 <= g <= g1:
                f = (g - g0) / (g1 - g0)
                r = table[g0] + f * (table[g1] - table[g0])
                break
    return max(0.0, min(1.0, r * scale))


class BankModel:
    """Recharge through the charge resistor from a current-limited PSU.

    Below the knee V_psu - I R the PSU's current limit sets the rate
    (V = V0 + I t / C); above it the resistor does (RC toward V_psu). Both
    branches have closed forms, so the simulator never integrates the bank.
    Sanity: 1500 us at 1.0 A recovers 11 -> 47.6 V in ~0.5 s at 4 cans
    (C 7.392 mF, R 22, V_psu 49.6); at 0.3 A ~1.0 s.
    """

    def __init__(self, C_F, R_charge_ohm, V_psu, I_limit_A):
        self.C = float(C_F)
        self.R = float(R_charge_ohm)
        self.V_psu = float(V_psu)
        self.I = float(I_limit_A)
        self.tau = self.R * self.C
        self.v_knee = self.V_psu - self.I * self.R

    def v_after(self, v0, dt_s):
        if dt_s <= 0.0:
            return float(v0)
        v = float(v0)
        t = float(dt_s)
        if v < self.v_knee:
            t_cc = (self.v_knee - v) * self.C / self.I
            if t <= t_cc:
                return v + self.I * t / self.C
            v = self.v_knee
            t -= t_cc
        if self.tau <= 0.0:
            return self.V_psu
        return self.V_psu - (self.V_psu - v) * math.exp(-t / self.tau)

    def t_to(self, v0, v_target):
        """Seconds from v0 to v_target; 0 if already there, inf if unreachable."""
        if v_target <= v0:
            return 0.0
        if v_target >= self.V_psu:
            return float("inf")
        t = 0.0
        v = float(v0)
        if v < self.v_knee:
            v_stop = min(v_target, self.v_knee)
            t += (v_stop - v) * self.C / self.I
            v = v_stop
        if v_target > v:
            if self.tau <= 0.0:
                return t
            t += -self.tau * math.log((self.V_psu - v_target) / (self.V_psu - v))
        return t

    def v_post_shot(self, v_fire, gate_us, cans):
        return v_fire * bank_retention(cans, gate_us)


def bank_capacitance_F(profile, cans):
    """Small-signal bank capacitance from the profile if C5 has added it, else the table."""
    entry = None
    if profile is not None:
        entry = (profile.circuit.get("pulse_measured_by_cans", {})
                 .get(str(cans), {}))
    uF = entry.get("capacitance_100hz_uF") if entry else None
    if uF is None:
        uF = BANK_C_100HZ_UF.get(cans)
        if uF is None:
            uF = BANK_C_100HZ_UF[1] * cans
    return float(uF) * 1e-6


# ---------------------------------------------------------------------------
# kick
# ---------------------------------------------------------------------------
def kick_dv(imap, x_mm, v_mps, v_fire, leg, kick_scale=1.0):
    """Instantaneous dv from the map at the delivered position.

    The return leg uses the mirrored position (x -> -x). The map was built at
    its own bank voltage; the impulse goes as I^2 ~ V^2, so it is rescaled by
    (V_fire/V_map)^2. Returns (dv, clamped) where clamped flags a lookup
    outside the map's grid.
    """
    if leg == "ret":
        dv, clamped = imap.lookup(-x_mm, v_mps)
    else:
        dv, clamped = imap.lookup(x_mm, v_mps)
    return kick_scale * imap.rescale_factor(v_fire) * dv, clamped
