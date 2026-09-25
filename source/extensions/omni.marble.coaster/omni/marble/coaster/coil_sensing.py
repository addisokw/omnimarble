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
        self.detect_halfwidth_mm = 0.0

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

    def raw_ticks(self):
        """[(index, t_us)] of the captured channels in TIME order.

        The firmware's Station.raw_ticks(): what pass_direction() and
        velocity_local_mps() read, so those two are ports of the same code.
        """
        got = [(i, self._ts[i]) for i in range(self.n) if self._got[i]]
        got.sort(key=lambda p: p[1])
        return got

    def first_tick(self):
        """Timestamp of the earliest crossing, or None."""
        got = self.raw_ticks()
        return got[0][1] if got else None

    def pass_direction(self):
        """+1 / -1 / 0: the SPATIAL order of the crossings in TIME order.

        +1 means strictly increasing index (the ball ran index-up through the
        array), -1 strictly decreasing, 0 anything else -- and zero is a
        finding, not a failure: crossings that are not monotonic cannot be
        one transit, whatever a fitted slope says. A port of the firmware's
        Station.pass_direction(); which sign is "toward the coil" depends on
        the station's channel order, so the caller (FiringController) turns
        this into a travel direction with channel_x_mm.
        """
        got = self.raw_ticks()
        if len(got) < 2:
            return 0
        if all(got[k + 1][0] > got[k][0] for k in range(len(got) - 1)):
            return 1
        if all(got[k + 1][0] < got[k][0] for k in range(len(got) - 1)):
            return -1
        return 0

    def travel_direction(self):
        """+1 if the pass moved toward +x, -1 toward -x, 0 if incoherent.

        pass_direction() in index space, mapped through the channel layout:
        with channel_x_mm increasing with index the two agree; on a station
        whose indices run against x they are opposite.
        """
        order = self.pass_direction()
        if order == 0 or self.n < 2:
            return 0
        layout = 1 if self.channel_x_mm[-1] > self.channel_x_mm[0] else -1
        return order * layout

    def velocity_local_mps(self):
        """Speed over the LAST inter-crossing interval, or None.

        The firmware's Station.velocity_local_mps(): the ball's speed at the
        station's most recently crossed end -- on a triggering pass the end
        nearest the coil. This is what the return leg times its transit on:
        the full-station fit reads high on the decelerating B-side approach
        (it averages in the fast outer channels), and the bench found the
        last interval closer to the speed that governs the remaining travel.
        Noisier than the fit (one interval instead of five points), so the
        forward leg keeps the fit.
        """
        got = self.raw_ticks()
        if len(got) < 2:
            return None
        (i1, t1), (i2, t2) = got[-2], got[-1]
        g = t2 - t1
        if g <= 0 or i1 == i2:
            return None
        return abs(i2 - i1) * (self.pitch_mm / 1000.0) / (g / 1e6)

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


def crossing_from_velocity_us(x, target_x, velocity_mm_s, t_us):
    """Crossing time inferred from the ball's SPEED rather than the step.

    Needed in Kit, where the marble's transform is read from USD and USD only
    syncs at the render tick: the position sits frozen for ~16 physics steps
    and then jumps ~20mm at once. interpolate_crossing_us() would place that
    crossing inside the 2ms step the jump was reported in, which is wrong by
    most of a 32ms window -- and against a 22.14mm pitch that is nearly a whole
    channel.

    Velocity does stay current, so back the crossing out of it instead:
    exact under constant velocity, which the rig's flat measurement zone is
    built to provide.

    Returns None if the ball is not moving.
    """
    if not velocity_mm_s:
        return None
    return t_us - ((x - target_x) / velocity_mm_s) * 1e6


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


# The coil's faces, mm from its centre (rig_geometry.json face_in_x = -22.78;
# the coil is symmetric). The fire offsets are measured from the face on each
# leg's own side, so this turns an offset into an absolute delivered x.
COIL_FACE_X_MM = 22.78

# The firmware's empirical return trim (vbench firmware/config.py
# SUSTAIN_RET_TRIM_*): mm on top of the mirrored fire offset, K / v_local plus
# a per-us gate term beyond the reference on-time, clamped. Kept as a local
# copy rather than imported from scripts/sustain_model.py, which imports THIS
# module; tests/test_coil_sensing.py pins the two against each other.
RET_TRIM_K_MM_MPS = 1.7
RET_TRIM_GATE_MM_PER_US = -0.004
RET_TRIM_GATE_REF_US = 700.0
RET_TRIM_MAX_MM = 16.0


def return_trim_mm(v_local, on_us, K=RET_TRIM_K_MM_MPS,
                   gate_mm_per_us=RET_TRIM_GATE_MM_PER_US,
                   ref_us=RET_TRIM_GATE_REF_US, max_mm=RET_TRIM_MAX_MM):
    """The return leg's trim, exactly as firmware/main.py _wait_trigger_once.

    The constant-acceleration predictor that was to replace this was
    falsified on the bench (2026-09-25): the deceleration measured through
    station B is local to the station and does not continue to the coil, so
    the return leg is back to constant velocity over the last channel pair
    plus this trim.
    """
    gate_mm = gate_mm_per_us * ((on_us or ref_us) - ref_us)
    return min(max_mm, K / max(v_local, 0.02) + gate_mm)


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
    REARMED = "rearmed"       # a pass was rejected and the station cleared;
                              # back to WAITING on the next poll (sustain only)

    def __init__(self, profile_firing, station, leg="fwd", profile_return=None,
                 on_us=None, manual_offset_mm=None, rearm_on_reject=False,
                 incomplete_retries=1, coil_face_x_mm=COIL_FACE_X_MM):
        """leg="fwd" is the shot the rig has always taken: station A, travel
        toward +x, reach `last_channel_to_coil_mm` (+ half-width + offset).
        leg="ret" is its mirror on the return: station B, travel toward -x,
        reach from `profile_return` (SENSOR_B_FIRST_TO_COIL_MM) plus the
        empirical trim and the manual offset (firmware/main.py
        _wait_trigger_once, the `elif leg == "ret":` branch).

        rearm_on_reject=True (sustain) clears the station and keeps waiting
        on a wrong-way or incomplete pass instead of aborting, as the
        firmware's sustain mode does; a single shot still aborts.
        """
        self.firing = profile_firing
        self.station = station
        self.leg = leg
        self.profile_return = dict(profile_return or {})
        self.rearm_on_reject = bool(rearm_on_reject)
        self.incomplete_retries = int(incomplete_retries)
        self.coil_face_x_mm = float(coil_face_x_mm)
        self.required = int(profile_firing.get("required_channels", REQUIRED_CHANNELS))
        # The channel fired on the ball's LEADING EDGE, so its CENTRE was still
        # a half-width upstream of the sensor -- the centre has that much
        # further to travel than the sensor-to-coil geometry says.
        #
        # `detect_halfwidth_mm` had been sitting in the profile UNREFERENCED.
        # The firmware made the same omission, so the twin agreed with itself
        # and the error was invisible until the rig was fired: two shots landed
        # at 4.4-6.0 mm/s against a predicted 15.7, and the 5.2 mm offset
        # accounts for essentially all of it. The coil face sits on the steep
        # flank of the impulse-vs-position curve, so a few mm early is worth
        # most of the dv.
        self.detect_halfwidth_mm = float(
            profile_firing.get("detect_halfwidth_mm", 0.0))
        # fire_offset_mm delivers the centre PAST the coil face toward the
        # centre: +9 is the impulse optimum measured over 20 releases
        # (x = -13.78 against a model peak of -13.5) and is the firmware's
        # boot default (FIRE_OFFSET_DEFAULT_MM). Zero here would model the
        # rig as it was before that measurement, not as it operates.
        self.fire_offset_mm = float(profile_firing.get("fire_offset_mm", 0.0))
        # Reach BASE per leg: the coil-nearest channel of the leg's own
        # trigger station to the coil face on that side. The return side's is
        # a separate constant (SENSOR_B_FIRST_TO_COIL_MM), assumed symmetric
        # rather than measured, so it is carried separately in the profile.
        if leg == "ret":
            base_mm = float(self.profile_return.get(
                "last_channel_to_coil_mm",
                profile_firing["last_channel_to_coil_mm"]))
            self.required_direction = -1
        else:
            base_mm = float(profile_firing["last_channel_to_coil_mm"])
            self.required_direction = +1
        self.base_mm = base_mm
        # The forward reach is fixed; the return reach also carries the
        # per-pass trim, so it is only known once v_local is.
        self.distance_mm = base_mm + self.detect_halfwidth_mm + self.fire_offset_mm
        self.on_us = (float(on_us) if on_us is not None
                      else float(profile_firing.get("on_time_us", 0.0)) or None)
        self.manual_offset_mm = (
            float(manual_offset_mm) if manual_offset_mm is not None
            else float(self.profile_return.get("manual_offset_mm", 0.0)))
        self.trim_k = float(self.profile_return.get("trim_k_mm_mps", RET_TRIM_K_MM_MPS))
        self.trim_gate = float(self.profile_return.get(
            "trim_gate_mm_per_us", RET_TRIM_GATE_MM_PER_US))
        self.trim_ref_us = float(self.profile_return.get(
            "trim_gate_ref_us", RET_TRIM_GATE_REF_US))
        self.trim_max_mm = float(self.profile_return.get("trim_max_mm", RET_TRIM_MAX_MM))
        self.lead_us = float(profile_firing.get("trigger_lead_us", 0.0))
        self.slip_us = float(profile_firing.get("trigger_slip_us", 2000.0))
        self.capture_window_us = float(
            profile_firing.get("capture_window_ms", 200.0)) * 1000.0
        self.timeout_us = float(
            profile_firing.get("trigger_timeout_ms", 3000.0)) * 1000.0

        self.rearm_count = 0
        self.wrong_way_count = 0
        self.reject_reason = None      # why the last pass was rejected
        self._retries_left = self.incomplete_retries
        self._clear()

    def _clear(self):
        self.state = self.WAITING
        self.abort_reason = None
        self.fire_at_us = None
        self.v_in_mps = None
        self.v_local_mps = None
        self.v_transit_mps = None
        self.trim_mm = 0.0
        self.reach_mm = None
        self.transit_us = None
        self.residual_us = None
        self.slack_us = None          # how much margin the shot had
        self.pass_direction = 0
        self._first_seen_us = None
        self._start_us = None

    def reset(self):
        """Back to WAITING with the station cleared: a fresh pass."""
        self.station.reset()
        self._retries_left = self.incomplete_retries
        self.rearm_count = 0
        self.wrong_way_count = 0
        self.reject_reason = None
        self._clear()

    def rearm(self, reason):
        """Reject the captured pass, clear the station and keep waiting."""
        self.station.reset()
        self.rearm_count += 1
        self.reject_reason = reason
        self._clear()
        self.state = self.REARMED
        return self.REARMED

    def _abort(self, reason):
        self.state = self.ABORTED
        self.abort_reason = reason
        return self.ABORTED

    def _reject(self, reason, retry_ok=True):
        if self.rearm_on_reject and retry_ok:
            return self.rearm(reason)
        return self._abort(reason)

    @property
    def x_target_mm(self):
        """Where the pulse is timed to put the ball's centre, coil-centred x."""
        if self.leg == "ret":
            return (self.coil_face_x_mm - self.fire_offset_mm - self.trim_mm
                    - self.manual_offset_mm)
        return -self.coil_face_x_mm + self.fire_offset_mm

    def update(self, now_us):
        """Advance the state machine. Returns the state after this poll."""
        if self.state in (self.FIRED, self.ABORTED):
            return self.state
        if self.state == self.REARMED:
            self.state = self.WAITING

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
                    # by a whole pitch or more. Sustain re-arms (bounded, as
                    # the firmware's SUSTAIN_INCOMPLETE_RETRIES); a single
                    # shot is abandoned.
                    reason = (f"only {captured}/{self.required} channels at the "
                              f"trigger station")
                    if self._retries_left > 0:
                        self._retries_left -= 1
                        return self._reject(reason)
                    return self._abort(reason)
                return self.state

            # Direction, read off the crossing pattern. On a wrong-way pass
            # the marble reached the coil BEFORE this station, so it is past
            # the coil and receding: the last crossing is the channel
            # furthest from the coil and a pulse would brake it. There is no
            # wrong-way shot to time better; sustain waits for the next pass.
            self.pass_direction = self.station.travel_direction()
            if self.pass_direction == 0:
                return self._reject("incoherent capture: crossings are not one "
                                    "monotonic transit")
            if self.pass_direction != self.required_direction:
                self.wrong_way_count += 1
                v = self.station.velocity_mps() or 0.0
                return self._reject(f"wrong-way pass ({v:.3f} m/s, receding "
                                    f"from the coil)")

            self.v_in_mps = self.station.velocity_mps()
            if not self.v_in_mps:
                return self._abort("no velocity from the trigger station")
            self.residual_us = self.station.residual_us()
            self.v_local_mps = self.station.velocity_local_mps()

            if self.leg == "ret":
                # The bench-validated rule: constant velocity over the last
                # channel pair plus the empirical trim, on top of the
                # mirrored fire offset and the manual offset.
                self.v_transit_mps = self.v_local_mps or self.v_in_mps
                self.trim_mm = return_trim_mm(
                    self.v_transit_mps, self.on_us, self.trim_k, self.trim_gate,
                    self.trim_ref_us, self.trim_max_mm)
                self.reach_mm = (self.base_mm + self.detect_halfwidth_mm
                                 + self.fire_offset_mm + self.trim_mm
                                 + self.manual_offset_mm)
            else:
                self.v_transit_mps = self.v_in_mps
                self.reach_mm = self.distance_mm

            transit_us = transit_us_to(self.reach_mm, self.v_transit_mps)
            self.transit_us = transit_us
            elapsed_us = now_us - self.station.last_tick()
            remain = transit_us - self.lead_us - elapsed_us
            self.slack_us = remain
            if remain < -self.slip_us:
                return self._reject(
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
