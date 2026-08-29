"""Named rig profiles: which physical configuration the sim is imitating.

Two profiles exist and they describe genuinely different machines:

  legacy_gates_v1  six idealised IR gates, position-based cutoff, a 470uF bank
                   and a 10mm marble. Frozen -- the committed 300V trajectory
                   fixture was captured under it and must keep reproducing.

  vbench_v0        the as-built bench rig: two 5-channel IR stations at
                   published positions, a fixed on-time cutoff, a measured
                   1909uF/can bank and a 12.7mm steel ball. Generated from the
                   rig by scripts/import_rig_geometry.py.

Rebasing config/coil_params.json in place would have made the fixture
unreproducible, which is why these coexist rather than one replacing the other.

The bank helper is the part worth reading: capacitor ESR PARALLELS as 1/n, so
loop resistance FALLS as cans are added and the C-sweep varies R and C
together. vbench's own warning (NEXT_SESSION.md section 5) is that "a model
treating R as fixed will attribute that to capacitance and appear to validate
for the wrong reason".

Pure Python -- no carb/omni -- so the headless mirror and the tests can use it.
"""

import json
from pathlib import Path

DEFAULT_PROFILE_PATH = "config/rig_profile.json"


class ProfileError(ValueError):
    """A profile is missing, malformed, or internally inconsistent."""


class RigProfile:
    """One named configuration, with the derived circuit values."""

    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.description = data.get("description", "")
        self.sensing = data.get("sensing", {})
        self.firing = data.get("firing", {})
        self.coil = data.get("coil", {})
        self.circuit = data.get("circuit", {})
        self.marble = data.get("marble", {})
        self.track = data.get("track", {})

    # -- shape -----------------------------------------------------------
    @property
    def sensing_mode(self):
        return self.sensing.get("mode", "gates")

    @property
    def firing_mode(self):
        return self.firing.get("mode", "position_cutoff")

    @property
    def defers_to_coil_params(self):
        """True when the profile takes its circuit from config/coil_params.json.

        The legacy profile does this deliberately: deferring rather than
        duplicating is what guarantees it stays byte-identical to the
        configuration the fixture was captured under.
        """
        return "circuit_from" in self.data

    # -- bank ------------------------------------------------------------
    def bank(self, cans=None):
        """Circuit values for a given can count.

        Returns capacitance, the bank's parallel ESR, and the total loop
        resistance -- the last being what actually sets the pulse.
        """
        if self.defers_to_coil_params:
            raise ProfileError(
                f"profile {self.name!r} takes its circuit from "
                f"{self.data['circuit_from']}; bank() does not apply")

        circuit = self.circuit
        if cans is None:
            cans = int(circuit.get("cans_populated", 1))
        positions = int(circuit.get("bank_positions", 1))
        if not 1 <= cans <= positions:
            raise ProfileError(
                f"cans={cans} outside 1..{positions} for profile {self.name!r}")

        can_uF = float(circuit["can_capacitance_uF"])
        can_esr = float(circuit["can_esr_ohm"])
        measured = circuit.get("measured", {})

        # ESR parallels down as 1/n. The measured loop resistance was taken at
        # one can, so scale only its ESR share and hold the rest fixed.
        esr = can_esr / cans
        loop_at_1can = measured.get("loop_resistance_ohm")
        if loop_at_1can is None:
            loop_r = float(measured.get("coil_resistance_ohm", 0.0)) + esr
        else:
            loop_r = (float(loop_at_1can) - can_esr) + esr

        # Prefer the bench's per-configuration large-signal constants over the
        # n x per-can scaling. Electrolytic droop makes pulse capacitance
        # SUB-LINEAR in can count (each can carries less current, so each
        # droops less: C_pulse/C_smallsig measured 0.859 / 0.931 / 0.958 at
        # 1/2/3 cans), which the linear scaling cannot express -- it
        # understated C by 6.5% at 2 cans and 8% at 3, making the simulated
        # discharge too fast exactly where the on-time sweeps probe it. The
        # loop resistance in the same table is the value the SAME fit was
        # pinned to, so the pair is self-consistent; mixing table C with
        # scaled R would rebuild the inconsistency this exists to remove.
        pulse = circuit.get("pulse_measured_by_cans", {}).get(str(cans))
        capacitance_uF = can_uF * cans
        if pulse is not None:
            capacitance_uF = float(pulse["capacitance_uF"])
            loop_r = float(pulse["loop_resistance_ohm"])

        # Once the switch opens the bank is OUT of the circuit -- the current
        # freewheels through the coil and the diode alone, so the bank's ESR no
        # longer participates and the decay is slower than the discharge loop
        # would suggest (tau 167us on the coil's own 107 mOhm, against 109us if
        # you wrongly reuse the 164 mOhm loop figure). The tail carries a large
        # share of the impulse, so this distinction is worth real Delta-v.
        #
        # The coil's resistance is frequency dependent (107 mOhm at 1kHz,
        # 140 at 10kHz); the ring-down sits near 1/(2*pi*tau) ~ 950 Hz, so the
        # 1kHz figure is the right one here.
        freewheel_r = float(measured.get("coil_resistance_ohm", loop_r))
        # A scope-fitted tail time constant outranks the derived R: tau is what
        # the marble actually feels, and the 2026-09 captures showed the
        # L/R_coil construction was wrong on BOTH inputs (see freewheel_note in
        # the profile). Expressed as an equivalent R so freewheel_current()
        # keeps its diode-drop term unchanged.
        tau_us = measured.get("freewheel_tau_us")
        if tau_us:
            freewheel_r = float(measured["inductance_uH"]) * 1e-6                 / (float(tau_us) * 1e-6)

        return {
            "cans": cans,
            "capacitance_uF": capacitance_uF,
            "esr_ohm": esr,
            "loop_resistance_ohm": loop_r,
            "freewheel_resistance_ohm": freewheel_r,
            "diode_vf_V": float(self.circuit.get("diode_vf_V", 0.84)),
            "inductance_uH": float(measured["inductance_uH"]),
        }

    def bank_options(self):
        positions = int(self.circuit.get("bank_positions", 1))
        return [self.bank(n) for n in range(1, positions + 1)]

    # -- sensing ---------------------------------------------------------
    def station_specs(self):
        """(name, channel_x_mm, pitch_mm, order_rev) for each station."""
        if self.sensing_mode != "stations":
            raise ProfileError(
                f"profile {self.name!r} has sensing mode {self.sensing_mode!r}")
        pitch = float(self.sensing["pitch_mm"])
        return {
            name: {
                "channel_x_mm": list(st["channel_x_mm"]),
                "pitch_mm": pitch,
                "order_rev": bool(st["order_rev"]),
                "role": st.get("role", ""),
            }
            for name, st in self.sensing["stations"].items()
        }

    def gate_window_us(self, requested_us=None):
        """PHYSICAL gate window: the requested on-time, clamped, minus overhead.

        The rig LOGS the clamped request but the switch is actually held for
        that minus FIRE_ON_US_OVERHEAD. Conflating the two is a silent 4.5%
        error at the 200us operating point, so both are returned.
        """
        firing = self.firing
        if requested_us is None:
            requested_us = float(firing.get("on_time_us", 0.0))
        applied = max(0.0, min(float(requested_us),
                               float(firing.get("on_time_max_us", requested_us))))
        overhead = float(firing.get("on_time_overhead_us", 0.0))
        return {"on_time_us": applied, "gate_us": max(0.0, applied - overhead)}


def _validate(name, profile):
    """Invariants worth failing loudly on rather than debugging in a sweep."""
    circuit = profile.circuit
    if circuit:
        v = float(circuit.get("charge_voltage_V", 0.0))
        v_max = float(circuit.get("voltage_max_V", 0.0))
        if v_max and v > v_max:
            raise ProfileError(
                f"{name}: charge_voltage_V {v} exceeds the board invariant {v_max}")

    if profile.sensing_mode == "stations":
        stations = profile.sensing.get("stations", {})
        if len(stations) < 2:
            raise ProfileError(f"{name}: expected two stations")

        # order_rev is NOT constrained to be opposite between the stations.
        # This used to require exactly that, on the reasoning that the mounts
        # sit on opposite flanks so one board must be turned around. The flanks
        # are right, the inference was not: the rig was built with mirror-image
        # mount PARTS, so both arrays sit the same way round relative to the
        # track axis. Measured on the rig 2026-08-23, both stations read
        # channel-increasing WITH travel -- see vbench firmware/config.py for
        # the three independent confirmations from the first roll.
        #
        # There is nothing to validate here from geometry alone: the pair
        # depends on which mount parts were printed, and only a roll settles it.

        # No channel may sit inside the coil former, or it would be measuring
        # the ball while the field is acting on it.
        face_in = profile.coil.get("face_in_x_mm")
        face_out = profile.coil.get("face_out_x_mm")
        if face_in is not None and face_out is not None:
            for st_name, st in stations.items():
                for x in st["channel_x_mm"]:
                    if face_in < x < face_out:
                        raise ProfileError(
                            f"{name}: station {st_name} channel at x={x} is "
                            f"inside the coil ({face_in}..{face_out})")

        n = int(profile.sensing.get("n_channels", 0))
        for st_name, st in stations.items():
            if len(st["channel_x_mm"]) != n:
                raise ProfileError(
                    f"{name}: station {st_name} has {len(st['channel_x_mm'])} "
                    f"channels, expected {n}")
    return profile


def load_profile(root, name=None, path=None):
    """Load one profile by name, defaulting to the file's default_profile."""
    root = Path(root)
    profile_path = Path(path) if path else root / DEFAULT_PROFILE_PATH
    if not profile_path.exists():
        raise ProfileError(f"no rig profile file at {profile_path}")

    doc = json.loads(profile_path.read_text(encoding="utf-8"))
    profiles = doc.get("profiles", {})
    if name is None:
        name = doc.get("default_profile")
    if name not in profiles:
        raise ProfileError(
            f"unknown profile {name!r}; have {sorted(profiles)}")
    return _validate(name, RigProfile(name, profiles[name]))


def available_profiles(root, path=None):
    root = Path(root)
    profile_path = Path(path) if path else root / DEFAULT_PROFILE_PATH
    if not profile_path.exists():
        return []
    doc = json.loads(profile_path.read_text(encoding="utf-8"))
    return sorted(doc.get("profiles", {}))
