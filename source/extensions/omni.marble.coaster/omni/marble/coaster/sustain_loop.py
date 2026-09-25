"""Sustain mode: the leg state machine the Kit extension drives per step.

A port of vbench firmware/main.py cmd_sustain restructured as a poll-per-
physics-step machine, the way FiringController restructures
_wait_trigger_once. The rules are the firmware's:

  * legs alternate fwd (trigger station A, travel toward +x) and ret
    (trigger B, travel toward -x); a pass the wrong way for the awaited leg
    re-arms the station instead of firing (the ball is past the coil and
    receding -- a pulse would brake it);
  * the trigger station is armed BEFORE the bank is waited on; the bank
    releases at recharge_frac x V_start, or at floor_frac x V_start once a
    channel of the trigger station has tripped (_wait_bank_ready), and the
    pulse lands at max(fire time, release time);
  * a pass whose fire window is more than trigger_slip_us gone when the
    bank releases is skipped and the leg stays armed for the next
    same-direction pass;
  * the shot budget and the clock are checked before every kick, not every
    cycle -- they bound discharges, not laps;
  * after the pulse, v_out is the OUT station's own fit (the same estimator
    as v_in), and the next leg's trigger is that same station, re-armed.

Everything the loop knows about the ball comes from the two VirtualStations
and the caller's fire-time delivery note; it never sees the true state. Bank
voltage between shots is the closed-form BankModel from whatever post-shot
voltage the caller reports (the coupled RLC ODE's in Kit; the retention
table's in the 1-D twin).

Pure Python: no carb/omni. tests/test_sustain_loop.py drives it with the 1-D
twin's own kinematics (sustain_model.integrate_flat + the impulse map) and
requires the limit cycle simulate_sustain.py predicts; the extension drives
it with PhysX and the PINN.
"""

import csv
import math
from pathlib import Path

try:
    from .coil_sensing import FiringController      # inside the Kit package
except ImportError:                                  # tests / scripts: on sys.path
    from coil_sensing import FiringController


class SustainSettings:
    """The `sustain <on_us> <max_shots> <max_seconds>` arguments plus the
    bench PSU, as the extension's settings supply them."""

    def __init__(self, cans=4, on_time_us=1500.0, max_shots=30, max_seconds=120.0,
                 psu_amps=1.0, psu_volts=49.6, charge_r_ohm=22.0,
                 recharge_frac=0.96, floor_frac=0.60, recharge_timeout_s=30.0,
                 pass_timeout_ms=20000.0, out_timeout_ms=5000.0,
                 manual_offset_mm=None, incomplete_retries=1):
        self.cans = int(cans)
        self.on_time_us = float(on_time_us)
        self.max_shots = int(max_shots)
        self.max_seconds = float(max_seconds)
        self.psu_amps = float(psu_amps)
        self.psu_volts = float(psu_volts)
        self.charge_r_ohm = float(charge_r_ohm)
        self.recharge_frac = float(recharge_frac)
        self.floor_frac = float(floor_frac)
        self.recharge_timeout_s = float(recharge_timeout_s)
        self.pass_timeout_ms = float(pass_timeout_ms)
        self.out_timeout_ms = float(out_timeout_ms)
        self.manual_offset_mm = manual_offset_mm      # None: the profile's
        self.incomplete_retries = int(incomplete_retries)

    def to_dict(self):
        return dict(vars(self))


# Column order of the per-shot CSV. Names match simulate_sustain's kick
# records where the two overlap, so one parser reads both.
RECORD_COLUMNS = (
    "leg", "cycle", "kick_idx", "t_s", "t_fire_s", "v_in_fit", "v_local",
    "resid_us", "trim_mm", "x_target", "reach_mm", "transit_us", "v_transit",
    "x_delivered", "v_true_at_fire", "v_bank_at_release", "v_bank_at_fire",
    "dv_coil", "v_out_fit", "dv_raw", "late_us", "skipped", "skip_reason",
)


class SustainLoop:
    WAITING = "waiting"       # trigger station armed, watching for the pass
    ARMED = "armed"           # pass captured; counting down to the pulse
    FIRING = "firing"         # pulse in progress; waiting for shot_done()
    POST_SHOT = "post_shot"   # reading v_out at the out station

    def __init__(self, profile, stations, settings, bank, v_start, t0_us=0.0):
        self.profile = profile
        self.stations = stations
        self.settings = settings
        self.bank = bank
        self.v_start = float(v_start)
        self.v_target = settings.recharge_frac * self.v_start
        self.v_floor = settings.floor_frac * self.v_start
        self.t0_us = float(t0_us)
        self.st_in = profile.sensing.get("station_in", "A")
        self.st_out = profile.sensing.get("station_out", "B")
        self.firing = profile.firing
        self.firing_return = getattr(profile, "firing_return", {}) or {}
        self.slip_us = float(self.firing.get("trigger_slip_us", 2000.0))

        self.records = []          # per attempted kick (fired or skipped)
        self.cycles = []           # per captured forward pass at A
        self.events = []           # (t_s, text) for the console
        self.n_shots = 0
        self.cycle = 0
        self.finished = False
        self.reason = None
        self.t_end_us = None
        self.t_last_fire_us = None
        self.t_post_us = None
        self.v_post = self.v_start
        self.t_fire_planned_us = None
        self.rec = None
        self.ctl = None
        self.leg = None
        self.state = None
        self._new_leg("fwd", self.t0_us)

    # -- plumbing ---------------------------------------------------------
    def trigger_station(self, leg=None):
        leg = leg or self.leg
        return self.st_in if leg == "fwd" else self.st_out

    def out_station(self, leg=None):
        leg = leg or self.leg
        return self.st_out if leg == "fwd" else self.st_in

    def _log(self, t_us, text):
        self.events.append(((t_us - self.t0_us) * 1e-6, text))

    def _finish(self, t_us, reason):
        self.finished = True
        self.reason = reason
        self.t_end_us = t_us
        return {"event": "done", "reason": reason}

    def finish(self, t_us, reason):
        """The caller ends the run (marble parked, stalled, budget of the host)."""
        return self._finish(t_us, reason)

    def _new_leg(self, leg, now_us):
        self.leg = leg
        for st in self.stations.values():
            st.reset()
        trig = self.stations[self.trigger_station(leg)]
        self.ctl = FiringController(
            self.firing, trig, leg=leg, profile_return=self.firing_return,
            on_us=self.settings.on_time_us,
            manual_offset_mm=self.settings.manual_offset_mm,
            rearm_on_reject=True,
            incomplete_retries=self.settings.incomplete_retries)
        # In sustain the pass timeout is the firmware's SUSTAIN_PASS_TIMEOUT_MS:
        # no pass in this long means the marble has died somewhere clear of
        # the sensors.
        self.ctl.timeout_us = self.settings.pass_timeout_ms * 1000.0
        self.state = self.WAITING
        self.t_fire_planned_us = None
        self.rec = None
        self.t_leg_start_us = now_us

    # -- bank ---------------------------------------------------------------
    def bank_v(self, t_us):
        """Bank voltage at t from the closed-form recharge."""
        if self.t_post_us is None:
            return self.v_start
        return self.bank.v_after(self.v_post, max(0.0, (t_us - self.t_post_us) * 1e-6))

    def release_time_us(self, t_first_us):
        """When _wait_bank_ready releases for a pass whose first trip was t_first.

        recharge_frac x V_start unconditionally, or floor_frac x V_start
        once the trigger station has tripped -- whichever comes first.
        """
        if self.t_post_us is None:
            return -float("inf")
        t96 = self.t_post_us + self.bank.t_to(self.v_post, self.v_target) * 1e6
        t60 = self.t_post_us + self.bank.t_to(self.v_post, self.v_floor) * 1e6
        return min(t96, max(t60, t_first_us))

    # -- the machine -----------------------------------------------------------
    def poll(self, now_us):
        """Advance. Returns an event dict or None.

        Events: {"event": "fire", ...} -- the caller must apply the pulse now
        and later call shot_done(); "skip" / "rearm" -- a pass was rejected;
        "leg" -- v_out read, next leg armed; "done" -- the run is over.
        """
        if self.finished:
            return None

        if self.state == self.WAITING:
            elapsed_s = (now_us - self.t0_us) * 1e-6
            if elapsed_s >= self.settings.max_seconds:
                return self._finish(now_us, "clock (%.0f s)" % self.settings.max_seconds)
            state = self.ctl.update(now_us)
            if state == FiringController.REARMED:
                self._log(now_us, "%s pass rejected at %s: %s -- re-armed"
                          % (self.leg, self.trigger_station(), self.ctl.reject_reason))
                return {"event": "rearm", "reason": self.ctl.reject_reason}
            if state == FiringController.ABORTED:
                return self._finish(now_us, "shot did not complete (%s)"
                                    % self.ctl.abort_reason)
            if state == FiringController.ARMED:
                return self._arm(now_us)
            return None

        if self.state == self.ARMED:
            if now_us >= self.t_fire_planned_us:
                self.state = self.FIRING
                v_fire = self.bank_v(now_us)
                self.n_shots += 1
                self.rec.update({"t_fire_s": (now_us - self.t0_us) * 1e-6,
                                 "v_bank_at_fire": v_fire,
                                 "kick_idx": len(self.records) + 1})
                if self.leg == "fwd" and self.cycles:
                    self.cycles[-1]["kicked"] = True
                self.t_last_fire_us = now_us
                return {"event": "fire", "leg": self.leg, "v_bank": v_fire,
                        "record": self.rec}
            return None

        if self.state == self.POST_SHOT:
            out = self.stations[self.out_station()]
            timed_out = (now_us - self.t_last_fire_us) > self.settings.out_timeout_ms * 1000.0
            if out.complete() or timed_out:
                v_out = out.velocity_mps() if out.n_captured() >= 2 else None
                self.rec["v_out_fit"] = v_out
                v_in = self.rec.get("v_in_fit")
                self.rec["dv_raw"] = (v_out - v_in) if (v_out and v_in) else None
                self.records.append(self.rec)
                rec = self.rec
                self._log(now_us, "[%s kick %d/%d @ %.0fs] v_in %s v_out %s raw %s"
                          % ("A>" if self.leg == "fwd" else "<B", self.n_shots,
                             self.settings.max_shots, (now_us - self.t0_us) * 1e-6,
                             _f(rec["v_in_fit"]), _f(v_out), _f(rec["dv_raw"], "%+.3f")))
                if self.n_shots >= self.settings.max_shots:
                    return self._finish(now_us, "shot count (%d)" % self.settings.max_shots)
                nxt = "ret" if self.leg == "fwd" else "fwd"
                self._new_leg(nxt, now_us)
                return {"event": "leg", "leg": nxt, "record": rec}
            return None

        return None

    def _arm(self, now_us):
        ctl = self.ctl
        trig = ctl.station
        t_first = trig.first_tick()
        if self.leg == "fwd":
            self.cycle += 1
            self.cycles.append({"cycle": self.cycle, "v_in_A": ctl.v_in_mps,
                                "kicked": False, "t_s": (now_us - self.t0_us) * 1e-6})
        rec = {
            "leg": self.leg, "cycle": self.cycle, "kick_idx": None,
            "t_s": (trig.last_tick() - self.t0_us) * 1e-6,
            "v_in_fit": ctl.v_in_mps, "v_local": ctl.v_local_mps,
            "resid_us": ctl.residual_us, "trim_mm": ctl.trim_mm,
            "x_target": ctl.x_target_mm, "reach_mm": ctl.reach_mm,
            "transit_us": ctl.transit_us, "v_transit": ctl.v_transit_mps,
            "x_delivered": None, "v_true_at_fire": None,
            "v_bank_at_release": None, "v_bank_at_fire": None, "dv_coil": None,
            "v_out_fit": None, "dv_raw": None, "late_us": None,
            "t_fire_s": None, "skipped": False, "skip_reason": None,
        }
        t_release = self.release_time_us(t_first)
        if t_release - now_us > self.settings.recharge_timeout_s * 1e6:
            rec.update({"skipped": True, "skip_reason": "bank stalled"})
            self.records.append(rec)
            return self._finish(now_us, "bank did not recharge")
        t_fire = ctl.fire_at_us
        rec["late_us"] = (t_fire - t_release) if t_release > -float("inf") else None
        if t_release > t_fire + self.slip_us:
            gone_ms = (t_release - t_fire) * 1e-3
            rec.update({"skipped": True,
                        "skip_reason": "fire window %.0f ms gone when the bank released"
                        % gone_ms})
            self.records.append(rec)
            self._log(now_us, "[%s pass skipped: %s] v_in %s local %s"
                      % ("A>" if self.leg == "fwd" else "<B", rec["skip_reason"],
                         _f(rec["v_in_fit"]), _f(rec["v_local"])))
            ctl.rearm(rec["skip_reason"])
            return {"event": "skip", "record": rec}
        self.t_fire_planned_us = max(t_fire, t_release)
        rec["v_bank_at_release"] = self.bank_v(max(t_release, t_first))
        self.rec = rec
        self.state = self.ARMED
        if self.leg == "ret":
            self._log(now_us, "return trim %+.1f mm (empirical, %.1f / v, v_local %s)"
                      % (ctl.trim_mm, ctl.trim_k, _f(ctl.v_local_mps)))
        return {"event": "armed", "record": rec,
                "t_fire_us": self.t_fire_planned_us}

    def note_delivery(self, x_mm, v_mps, dv_coil=None):
        """Where the ball actually was when the pulse landed (caller's truth)."""
        if self.rec is not None:
            self.rec["x_delivered"] = x_mm
            self.rec["v_true_at_fire"] = v_mps
            if dv_coil is not None:
                self.rec["dv_coil"] = dv_coil

    def note_impulse(self, dv_coil):
        """The coil's own impulse over the shot, m/s (the twin's dv_coil)."""
        if self.rec is not None:
            self.rec["dv_coil"] = dv_coil

    def shot_done(self, now_us, v_post):
        """The pulse is over; the bank sits at v_post and starts recharging."""
        self.v_post = float(v_post)
        self.t_post_us = float(now_us)
        self.state = self.POST_SHOT

    # -- reporting ------------------------------------------------------------
    @property
    def fired(self):
        return [r for r in self.records if not r["skipped"]]

    def fwd_v_in(self):
        return [r["v_in_fit"] for r in self.fired if r["leg"] == "fwd" and r["v_in_fit"]]

    def limit_cycle(self, n_last=5):
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
        return {"n": len(good), "v_in": mean, "v_in_last": good[-1],
                "drift": drift, "trend": trend, "n_tail": len(tail)}

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

    def summary_lines(self):
        """The run summary in cmd_sustain's own format."""
        s = self.settings
        t_end = ((self.t_end_us if self.t_end_us is not None else self.t0_us)
                 - self.t0_us) * 1e-6
        lines = ["  " + "-" * 58,
                 "  sustain ended: %s" % (self.reason or "running"),
                 "  %d shots in %.1f s." % (self.n_shots, t_end)]
        good = self.fwd_v_in()
        if len(good) >= 2:
            lines.append("  v_in at A per cycle, m/s:")
            for i in range(0, len(good), 8):
                lines.append("    " + "  ".join("%.3f" % v for v in good[i:i + 8]))
            lc = self.limit_cycle()
            lines.append("  %s: %.3f -> %.3f m/s, %+.4f m/s per cycle over %d cycles."
                         % (lc["trend"], good[0], good[-1], lc["drift"], len(good)))
            lines.append("  limit cycle (last %d): %.3f m/s" % (lc["n_tail"], lc["v_in"]))
        far, entry = self.excursion_losses()
        for name, lst in (("far ramp", far), ("entry ramp", entry)):
            if lst:
                lines.append("  %s excursion: %s mm/s lost (n=%d)"
                             % (name, "/".join("%.0f" % (1000 * d) for d in lst), len(lst)))
        lines.append("  settings: %d can(s) | on=%.0f us | PSU %.1f V / %.2f A through %.0f ohm"
                     % (s.cans, s.on_time_us, s.psu_volts, s.psu_amps, s.charge_r_ohm))
        return lines


def _f(value, spec="%.3f", none="--"):
    return none if value is None else spec % value


def write_records_csv(path, records, meta=None):
    """Per-shot records to CSV with '# key=value' header lines, like the
    trajectory writer. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        for key, value in (meta or {}).items():
            f.write(f"# {key}={value}\n")
        writer = csv.writer(f)
        writer.writerow(RECORD_COLUMNS)
        for rec in records:
            row = []
            for col in RECORD_COLUMNS:
                v = rec.get(col)
                if isinstance(v, bool):
                    v = int(v)
                elif isinstance(v, float):
                    v = round(v, 6) if math.isfinite(v) else ""
                elif v is None:
                    v = ""
                row.append(v)
            writer.writerow(row)
    return path
