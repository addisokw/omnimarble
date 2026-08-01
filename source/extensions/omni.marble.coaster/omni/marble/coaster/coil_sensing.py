"""Virtual IR velocity stations mirroring the vbench rig's sensing.

The bench rig measures velocity with two 5-channel IR reflective arrays -- one
before the coil (station A, which also triggers the shot) and one after it
(station B). Velocity is a least-squares fit of channel position against
first-edge time, in ../omnimarble-vbench/firmware/sensors.py.

This module is a deliberate LINE-FOR-LINE PORT of that estimator, not an
improvement on it. The sim's job is to predict what the rig will measure, so it
has to inherit the rig's measurement biases too -- otherwise a sim-vs-real gap
cannot be attributed to physics rather than instrumentation. Three quirks are
preserved on purpose:

  * The fit assumes a UNIFORM pitch and uses `index * pitch`, even though the
    sim knows each channel's true x. Using the true positions would make the
    sim a better estimator than the rig and hide real pitch error.
  * `t0` is the timestamp of the LOWEST SPATIAL INDEX, which for a station the
    marble crosses in descending order is the last channel crossed, not the
    first. The slope is unaffected; `velocity_mps` takes the magnitude.
  * `residual_us` returns None below 3 channels -- two points fit a line
    exactly, so the residual would be a meaningless zero.

Timestamps are microseconds (float) to match the firmware's `ticks_us`, which
also lets recorded hardware logs be replayed through this code unchanged.

Pure Python: no carb/omni imports, so it is unit-testable and usable from the
headless mirror as well as the Kit extension.
"""

import math

# Fewer than this many channels at the trigger station and the shot is
# abandoned: with a channel missing, the last crossing is not necessarily the
# one nearest the coil, so the transit distance would be wrong by a whole pitch
# or more (firmware/main.py:89-96).
REQUIRED_CHANNELS = 5

# Above this the straight-line fit is not describing the motion -- either one
# channel is mistiming or the marble is accelerating through a zone that is
# meant to be flat (firmware/config.py:163).
RESID_WARN_US = 2000.0


class VirtualStation:
    """One 5-channel IR array, recording first-edge crossing times.

    channel_x_mm are the channels' true positions in coil-centred coordinates
    (x = travel), ordered as the rig's firmware indexes them. `rev` reverses
    them once at construction so every index below is a spatial position, as
    the firmware does in Station.__init__.
    """

    def __init__(self, name, channel_x_mm, pitch_mm, rev=False):
        self.name = name
        self.channel_x_mm = (list(reversed(channel_x_mm)) if rev
                             else list(channel_x_mm))
        self.rev = rev
        self.pitch_mm = float(pitch_mm)
        self.n = len(self.channel_x_mm)
        self._ts = [0.0] * self.n      # first-edge timestamp, us
        self._got = [False] * self.n

    def reset(self):
        self._ts = [0.0] * self.n
        self._got = [False] * self.n

    def record(self, index, t_us):
        """Latch channel `index`'s first edge. Later edges are ignored."""
        if not self._got[index]:
            self._ts[index] = float(t_us)
            self._got[index] = True

    def n_captured(self):
        return sum(1 for g in self._got if g)

    def complete(self):
        return self.n_captured() == self.n

    def last_tick(self):
        """Timestamp of the most recent crossing, or None.

        This is what the firmware extrapolates the fire time from, so it is
        the latest edge in TIME -- not the highest spatial index.
        """
        best = None
        for i in range(self.n):
            if self._got[i] and (best is None or self._ts[i] > best):
                best = self._ts[i]
        return best

    def last_channel_x_mm(self):
        """x of the channel crossed last -- the one nearest the coil."""
        best_i, best_t = None, None
        for i in range(self.n):
            if self._got[i] and (best_t is None or self._ts[i] > best_t):
                best_i, best_t = i, self._ts[i]
        return None if best_i is None else self.channel_x_mm[best_i]

    def _fit(self):
        """LS (slope, intercept, m, xs, ys) of position[m] vs time[s], or None."""
        got = [(i, self._ts[i]) for i in range(self.n) if self._got[i]]
        if len(got) < 2:
            return None
        pitch_m = self.pitch_mm / 1000.0
        t0 = got[0][1]
        xs, ys = [], []
        for (i, ts) in got:
            xs.append((ts - t0) / 1000000.0)
            ys.append(i * pitch_m)
        m = len(xs)
        sx = sum(xs)
        sy = sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        denom = m * sxx - sx * sx
        if denom == 0:
            return None
        slope = (m * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / m
        return (slope, intercept, m, xs, ys)

    def velocity_mps(self):
        """Fitted speed as a MAGNITUDE, or None if fewer than 2 channels fired."""
        f = self._fit()
        if f is None:
            return None
        return abs(f[0])

    def velocity_mm_s(self):
        v = self.velocity_mps()
        return None if v is None else v * 1000.0

    def residual_us(self):
        """RMS of the fit residuals in microseconds -- the honest quality number."""
        f = self._fit()
        if f is None:
            return None
        slope, intercept, m, xs, ys = f
        if m < 3 or slope == 0:
            return None
        ss = 0.0
        for x, y in zip(xs, ys):
            r = (x - (y - intercept) / slope) * 1000000.0
            ss += r * r
        return (ss / m) ** 0.5


def interpolate_crossing_us(prev_x, x, target_x, t_us, dt_us):
    """Sub-step crossing time for a channel at `target_x`, in microseconds.

    The physics steps at 500 Hz, so sampling the step time directly would
    quantise every timestamp to 2 ms -- about 2 mm at 1 m/s against a 22.14 mm
    pitch, a ~9% per-channel error that would swamp the real measurement. The
    marble's position is known continuously, so interpolate linearly within the
    step instead. That leaves the sim's estimator limited by the same things
    the rig's is (pitch, the fit, the ball's motion) rather than by the solver
    step rate.

    Returns None if the segment does not span target_x.
    """
    if x == prev_x:
        return None
    frac = (target_x - prev_x) / (x - prev_x)
    if not (0.0 <= frac <= 1.0):
        return None
    return (t_us - dt_us) + frac * dt_us


def crossed(prev_x, x, target_x):
    """True if the segment [prev_x, x] spans target_x, in either direction."""
    return (prev_x < target_x <= x) or (x <= target_x < prev_x)


def transit_us_to(distance_mm, velocity_mps):
    """Time to cover `distance_mm` at a constant `velocity_mps`.

    The rig computes its fire delay exactly this way -- a constant-velocity
    extrapolation from the last channel crossed to the coil's entry face
    (firmware/main.py:112-117). The marble is really still decelerating
    slightly, so this is biased; that bias is part of what the sim reproduces.
    """
    if not velocity_mps:
        return None
    return distance_mm / 1000.0 / velocity_mps * 1e6


def fit_is_suspect(residual_us_value):
    return residual_us_value is not None and residual_us_value > RESID_WARN_US


class FiringController:
    """Decides when to fire, from the trigger station alone.

    A port of the rig's firmware/main.py _wait_and_time_trigger, restructured
    as a poll-per-physics-step state machine instead of a blocking sleep.

    The rig waits for the marble to CLEAR station A, fits v_in from the full
    five-channel pass, and then times the pulse so it lands as the marble
    reaches the coil's ENTRY FACE. It used to fire as soon as two channels had
    triggered, which put the marble 124 mm short of the coil with the pulse
    long over before it arrived.

    THE CONTROLLER IS GIVEN ONLY THE STATION, NEVER THE MARBLE'S TRUE STATE.
    That is deliberate and structural. The fire time comes from a
    constant-velocity extrapolation off the estimator's fit, while the marble
    is really still decelerating slightly -- so the rig fires marginally late.
    Handing the controller ground truth would quietly erase a real bias the
    twin exists to predict.
    """

    WAITING = "waiting"       # no channel has fired yet
    CAPTURING = "capturing"   # some fired, waiting for the rest
    ARMED = "armed"           # fire time computed, counting down
    FIRED = "fired"
    ABORTED = "aborted"

    def __init__(self, profile_firing, station):
        self.firing = profile_firing
        self.station = station
        self.required = int(profile_firing.get("required_channels", REQUIRED_CHANNELS))
        self.distance_mm = float(profile_firing["last_channel_to_coil_mm"])
        self.lead_us = float(profile_firing.get("trigger_lead_us", 0.0))
        self.slip_us = float(profile_firing.get("trigger_slip_us", 2000.0))
        self.capture_window_us = float(
            profile_firing.get("capture_window_ms", 200.0)) * 1000.0
        self.timeout_us = float(
            profile_firing.get("trigger_timeout_ms", 3000.0)) * 1000.0

        self.state = self.WAITING
        self.abort_reason = None
        self.fire_at_us = None
        self.v_in_mps = None
        self.residual_us = None
        self.slack_us = None          # how much margin the shot had
        self._first_seen_us = None
        self._start_us = None

    def _abort(self, reason):
        self.state = self.ABORTED
        self.abort_reason = reason
        return self.ABORTED

    def update(self, now_us):
        """Advance the state machine. Returns the state after this poll."""
        if self.state in (self.FIRED, self.ABORTED):
            return self.state

        if self._start_us is None:
            self._start_us = now_us

        captured = self.station.n_captured()

        if self.state == self.WAITING:
            if captured == 0:
                if now_us - self._start_us > self.timeout_us:
                    return self._abort("no marble seen before the trigger timeout")
                return self.state
            self._first_seen_us = now_us
            self.state = self.CAPTURING

        if self.state == self.CAPTURING:
            if captured < self.required:
                if now_us - self._first_seen_us > self.capture_window_us:
                    # A partial capture cannot time a shot: with a channel
                    # missing, the last crossing is not necessarily the one
                    # nearest the coil, so the transit distance would be wrong
                    # by a whole pitch or more.
                    return self._abort(
                        f"only {captured}/{self.required} channels at the "
                        f"trigger station")
                return self.state

            self.v_in_mps = self.station.velocity_mps()
            if not self.v_in_mps:
                return self._abort("no velocity from the trigger station")
            self.residual_us = self.station.residual_us()

            transit_us = transit_us_to(self.distance_mm, self.v_in_mps)
            elapsed_us = now_us - self.station.last_tick()
            remain = transit_us - self.lead_us - elapsed_us
            self.slack_us = remain
            if remain < -self.slip_us:
                return self._abort(
                    f"missed the window by {-remain:.0f}us; the marble is "
                    f"already past the coil")
            self.fire_at_us = now_us + max(remain, 0.0)
            self.state = self.ARMED

        if self.state == self.ARMED and now_us >= self.fire_at_us:
            self.state = self.FIRED

        return self.state

    @property
    def fit_suspect(self):
        return fit_is_suspect(self.residual_us)


def _mean(values):
    return math.fsum(values) / len(values)
