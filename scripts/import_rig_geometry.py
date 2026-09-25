"""Generate the `vbench_v0` profile in config/rig_profile.json from the rig itself.

The bench rig publishes track/rig_geometry.json explicitly as "the sim
contract" (vbench track/RIG.md), and firmware/config.py holds the measured
circuit constants. Both are generated or measured upstream, so retyping them
here would guarantee drift. This script imports them instead and records a hash
of each source, so a later run can tell you the rig moved under you.

    uv run python scripts/import_rig_geometry.py            # write the profile
    uv run python scripts/import_rig_geometry.py --check     # verify, exit 1 on drift

The vbench checkout is expected as a sibling of this repo; override with
--vbench. Nothing in vbench is written or modified -- this is read-only on that
side.

TRAP, handled here: rig_geometry.json also carries `firmware_pitch_mm: 11.0`
and `firmware_pitch_matches: false`, both hardcoded in track/rig.py and stale
since the firmware was corrected to 22.14. The authoritative value is
`sensor_module.sensor_pitch_mm`, and this script cross-checks it against
firmware/config.py rather than trusting either alone.
"""

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILE_PATH = ROOT / "config" / "rig_profile.json"
DEFAULT_VBENCH = ROOT.parent / "omnimarble-vbench"

# Module-level constants read out of firmware/config.py. Parsed with ast, never
# imported -- that file targets MicroPython and pulls in `machine` at runtime.
FIRMWARE_CONSTANTS = (
    "SENSOR_PITCH_MM", "STATION_IN", "STATION_OUT", "STATION_ORDER_REV",
    "SENSOR_A_LAST_TO_COIL_MM", "SENSOR_RESID_WARN_US",
    "CAPTURE_WINDOW_MS", "SHOT_TRIGGER_TIMEOUT_MS",
    "FIRE_DEFAULT_ON_US", "FIRE_MAX_ON_US", "FIRE_ON_US_OVERHEAD",
    "FIRE_TRIGGER_LEAD_US", "FIRE_TRIGGER_SLIP_US",
    "VBANK_MAX_V", "COIL_N_TURNS", "COIL_L_UH_NOMINAL", "COIL_R_MOHM_NOMINAL",
    "LOOP_R_MOHM_MEASURED", "BANK_UNIT_UF", "BANK_UNIT_ESR_MOHM",
    "BANK_POSITIONS",
)

# Large-signal bank values, preferred when the firmware defines them. The
# bench C and loop R are small-signal readings; the sim models a 200 A shot,
# under which the electrolytic loses ~12% of its capacitance and gains ~30%
# ESR. L is NOT in this list -- it was confirmed at 1 kHz and 10 kHz and does
# not move. Optional, so an older firmware/config.py still imports.
OPTIONAL_CONSTANTS = ("BANK_UNIT_UF_PULSE", "LOOP_R_MOHM_PULSE",
                      "BANK_2CAN_UF_PULSE", "LOOP_R_MOHM_PULSE_2CAN",
                      "BANK_3CAN_UF_PULSE", "LOOP_R_MOHM_PULSE_3CAN",
                      "BANK_4CAN_UF_PULSE", "LOOP_R_MOHM_PULSE_4CAN",
                      "BANK_5CAN_UF_PULSE", "LOOP_R_MOHM_PULSE_5CAN",
                      # small-signal (100 Hz LCR at J2) bank C per config:
                      # what the charge resistor sees, so the sustain twin's
                      # recharge model uses these, not the pulse values
                      "BANK_2CAN_UF_100HZ", "BANK_3CAN_UF_100HZ",
                      "BANK_4CAN_UF_100HZ", "BANK_5CAN_UF_100HZ",
                      "SENSOR_DETECT_HALFWIDTH_MM",
                      # the measured impulse-curve optimum the firmware boots
                      # with; without it here the profile silently reverted
                      # to 0 on regeneration (2026-09-24)
                      "FIRE_OFFSET_DEFAULT_MM",
                      # return leg (sustain): B's coil-nearest channel to the
                      # coil face, the empirical trim the bench validated
                      # after the kinematic predictor was falsified (09-25),
                      # the manual offset, and the bank release policy
                      "SENSOR_B_FIRST_TO_COIL_MM", "COIL_FACE_X_MM",
                      "SUSTAIN_RET_TRIM_K_MM_MPS", "SUSTAIN_RET_TRIM_GATE_MM_PER_US",
                      "SUSTAIN_RET_TRIM_GATE_REF_US", "SUSTAIN_RET_TRIM_MAX_MM",
                      "RET_OFFSET_DEFAULT_MM",
                      "SUSTAIN_RECHARGE_FRAC", "SUSTAIN_FIRE_FLOOR_FRAC",
                      "SUSTAIN_MAX_SHOTS", "SUSTAIN_MAX_SECONDS",
                      "SUSTAIN_PASS_TIMEOUT_MS", "SUSTAIN_RECHARGE_TIMEOUT_S")

# The fitted track-loss table (scripts/fit_track_losses.py) is embedded in the
# profile so the Kit extension carries the SAME numbers the 1-D twin runs on,
# with the file's hash so drift between the two is visible.
LOSSES_PATH = ROOT / "config" / "track_losses.json"

# Speed at which the PhysX-damping fallback is matched to the fitted flat law.
# PhysX linear damping is dv/dt = -d v; the fitted law is a0 + k v^2, so no
# single d reproduces it -- the equivalent is taken at the sustain cycle speed.
DAMPING_MATCH_V_MPS = 0.2


def _pulse_entry(fw, n, c_key, r_key):
    """One pulse_measured_by_cans row; carries the 100 Hz C when measured."""
    entry = {
        "capacitance_uF": float(fw[c_key]),
        "loop_resistance_ohm": float(fw[r_key]) / 1000.0,
    }
    hz_key = "BANK_UNIT_UF" if n == 1 else "BANK_%dCAN_UF_100HZ" % n
    if hz_key in fw:
        entry["capacitance_100hz_uF"] = float(fw[hz_key])
    return entry


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _firing_return(fw, sensor):
    """The return leg's constants and the sustain release policy."""
    return {
        "_note": (
            "Return leg (sustain): B's coil-nearest channel to the B-side coil "
            "face, assumed symmetric with A's (SENSOR_B_FIRST_TO_COIL_MM). The "
            "fire time is constant velocity over the last channel pair plus the "
            "EMPIRICAL trim min(trim_max, K / v_local + gate * (on_us - ref)) on "
            "top of the mirrored fire offset and the manual offset. The "
            "constant-acceleration predictor was falsified on the bench "
            "2026-09-25: the slowing through station B is local to the station "
            "and does not continue to the coil."),
        "trigger_station": fw["STATION_OUT"],
        "required_channels": int(sensor["channels"]),
        "last_channel_to_coil_mm": float(
            fw.get("SENSOR_B_FIRST_TO_COIL_MM", fw["SENSOR_A_LAST_TO_COIL_MM"])),
        "coil_face_x_mm": float(fw.get("COIL_FACE_X_MM", 22.78)),
        "trim_k_mm_mps": float(fw.get("SUSTAIN_RET_TRIM_K_MM_MPS", 0.0)),
        "trim_gate_mm_per_us": float(fw.get("SUSTAIN_RET_TRIM_GATE_MM_PER_US", 0.0)),
        "trim_gate_ref_us": float(fw.get("SUSTAIN_RET_TRIM_GATE_REF_US", 700)),
        "trim_max_mm": float(fw.get("SUSTAIN_RET_TRIM_MAX_MM", 16.0)),
        "manual_offset_mm": float(fw.get("RET_OFFSET_DEFAULT_MM", 0.0)),
        "sustain": {
            "recharge_frac": float(fw.get("SUSTAIN_RECHARGE_FRAC", 0.96)),
            "floor_frac": float(fw.get("SUSTAIN_FIRE_FLOOR_FRAC", 0.60)),
            "max_shots": int(fw.get("SUSTAIN_MAX_SHOTS", 30)),
            "max_seconds": float(fw.get("SUSTAIN_MAX_SECONDS", 120)),
            "pass_timeout_ms": float(fw.get("SUSTAIN_PASS_TIMEOUT_MS", 20000)),
            "recharge_timeout_s": float(fw.get("SUSTAIN_RECHARGE_TIMEOUT_S", 30)),
        },
    }


def _track_losses():
    """config/track_losses.json embedded verbatim, plus the PhysX modes."""
    if not LOSSES_PATH.exists():
        return None
    table = json.loads(LOSSES_PATH.read_text(encoding="utf-8"))
    flat = table.get("flat", {})
    a0 = float(flat.get("a0_mps2", 0.0))
    k = float(flat.get("k_per_m", 0.0))
    v = DAMPING_MATCH_V_MPS
    d_equiv = (a0 + k * v * v) / v if v > 0 else 0.0
    return {
        "_note": (
            "The fitted loss table (scripts/fit_track_losses.py) the 1-D sustain "
            "twin runs on, embedded so the Kit extension carries the same numbers. "
            "mode 'fitted': the flat-zone drag and the B-side excess are applied as "
            "an explicit axial force in the physics step and PhysX damping is zero; "
            "'physx': PhysX damping only (the pre-sustain behaviour). ramps "
            "'physx': the collidable STL ramps do the excursions; 'fitted': the "
            "energy-form transfer v_back^2 = alpha v_out^2 - beta is applied at "
            "the flat edge instead, to reproduce the twin exactly."),
        "mode": "fitted",
        "ramps": "physx",
        "source": {
            "path": str(LOSSES_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(LOSSES_PATH),
        },
        "schema": table.get("schema", "track_losses_v1"),
        "flat": table.get("flat", {}),
        "b_side": table.get("b_side", {}),
        "far": table.get("far", {}),
        "entry": table.get("entry", {}),
        "ramp_angle_deg": table.get("ramp_angle_deg", 55.0),
        "meta": table.get("meta", {}),
        "physx_damping": {
            "_note": (
                "PhysxRigidBodyAPI linear/angular damping per loss mode, 1/s. "
                "'fitted' is zero: the forces carry the loss. 'physx' matches "
                "dv/dt = -d v to the fitted a0 + k v^2 at %.2f m/s (the cycle "
                "speed); it is wrong everywhere else, which is why 'fitted' is "
                "the default. Angular damping is zero in both: on a rolling ball "
                "it brakes through the contact by an amount PhysX does not "
                "document, so it would be a second unfitted loss."
                % DAMPING_MATCH_V_MPS),
            "fitted": {"linear_per_s": 0.0, "angular_per_s": 0.0},
            "physx": {"linear_per_s": round(d_equiv, 4), "angular_per_s": 0.0,
                      "matched_at_mps": DAMPING_MATCH_V_MPS},
        },
        # Rolling coupling: a force at the centre of a ball rolling without
        # slip accelerates it at F / (m + I / r^2) = (5/7) F / m, so the
        # fitted (linear-speed) deceleration needs 7/5 of m a. Verified only
        # against the twin's numbers in a Kit run, not derived from PhysX.
        "rolling_inertia_factor": 1.4,
    }


def parse_firmware_config(path):
    """Pull module-level literal constants out of firmware/config.py via ast."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in (
                    FIRMWARE_CONSTANTS + OPTIONAL_CONSTANTS):
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    missing = set(FIRMWARE_CONSTANTS) - set(found)
    if missing:
        raise SystemExit(
            f"firmware/config.py is missing expected constants: {sorted(missing)}\n"
            "The firmware moved; update FIRMWARE_CONSTANTS before importing.")
    return found


def build_profile(vbench):
    geom_path = vbench / "track" / "rig_geometry.json"
    fw_path = vbench / "firmware" / "config.py"
    for path in (geom_path, fw_path):
        if not path.exists():
            raise SystemExit(f"not found: {path}\nIs --vbench pointing at the checkout?")

    geom = json.loads(geom_path.read_text(encoding="utf-8"))
    fw = parse_firmware_config(fw_path)

    sensor = geom["sensor_module"]
    pitch = float(sensor["sensor_pitch_mm"])

    # Cross-check the two independent sources rather than trusting either. This
    # is what catches a shots.csv taken with the old 11.0 placeholder, where
    # every velocity is out by 2x.
    if abs(pitch - float(fw["SENSOR_PITCH_MM"])) > 1e-9:
        raise SystemExit(
            f"pitch disagreement: rig_geometry.json says {pitch}, "
            f"firmware/config.py says {fw['SENSOR_PITCH_MM']}.\n"
            "Resolve upstream before importing -- every velocity scales with it.")
    if not sensor.get("sensor_pitch_trusted", False):
        raise SystemExit("rig_geometry.json marks sensor_pitch_mm untrusted")

    stations = {}
    for name, station in geom["stations"].items():
        stations[name] = {
            "role": station["role"],
            "centre_x_mm": station["centre_x"],
            "channel_x_mm": list(station["channel_x"]),
            "clear_of_coil_mm": station["clear_of_coil_mm"],
            "order_rev": bool(fw["STATION_ORDER_REV"][name]),
        }

    # No constraint on the order_rev pair. It was once required to differ,
    # inferring from the opposite-flank mounts that one board must be reversed.
    # Mirror-image mount parts mean both arrays sit the same way round instead,
    # and the rig measured both False on 2026-08-23. Whatever the firmware says
    # is the ground truth -- it is set from a roll, not derived.

    coil = geom["coil"]
    ball = geom["ball"]
    profile_bank_esr = float(fw["BANK_UNIT_ESR_MOHM"]) / 1000.0
    losses = _track_losses()

    return {
        "description": (
            "The as-built vbench rig: measured circuit, two 5-channel IR "
            "stations, fixed on-time cutoff, 12.7mm steel ball."),
        "provenance": {
            "rig_geometry": {
                "path": "track/rig_geometry.json",
                "sha256": sha256(geom_path),
            },
            "firmware_config": {
                "path": "firmware/config.py",
                "sha256": sha256(fw_path),
            },
            "note": (
                "Generated by scripts/import_rig_geometry.py -- do not hand-edit. "
                "Re-run with --check to detect upstream drift."),
        },
        "sensing": {
            "mode": "stations",
            "n_channels": int(sensor["channels"]),
            "pitch_mm": pitch,
            # rig_geometry.json's own note still claims the firmware says 11.0.
            # That was true when track/rig.py was written and is not now -- the
            # cross-check above just proved both sources agree -- so record the
            # resolution rather than propagate a stale warning into the sim.
            "pitch_provenance": (
                sensor.get("sensor_pitch_note", "")
                + f" [RESOLVED at import: firmware/config.py reads "
                  f"{fw['SENSOR_PITCH_MM']}, matching. The 'STILL SAYS 11.0' "
                  f"clause above is stale text hardcoded in track/rig.py.]"),
            "station_in": fw["STATION_IN"],
            "station_out": fw["STATION_OUT"],
            "stations": stations,
            # Half the optical chord a 12.7mm ball presents, measured at 3.3V
            # (vbench docs/IR_BOARD_ROLL_TEST.md): 9.68mm wide.
            # Half the ball's OPTICAL width at the trigger station. The
            # channel fires on the leading edge, so the centre is this far
            # upstream when it does. Was hardcoded to 9.68/2 (the Leonardo
            # jig figure); the rig's station A measures 10.4, and A is what
            # times the shot.
            "detect_halfwidth_mm": float(
                fw.get("SENSOR_DETECT_HALFWIDTH_MM", 9.68 / 2.0)),
            "resid_warn_us": float(fw["SENSOR_RESID_WARN_US"]),
        },
        "firing": {
            # Duplicated from sensing on purpose: the firing controller needs it
            # to reach the ball's CENTRE rather than the sensor it tripped.
            "detect_halfwidth_mm": float(
                fw.get("SENSOR_DETECT_HALFWIDTH_MM", 9.68 / 2.0)),
            "mode": "fixed_on_time",
            "trigger_station": fw["STATION_IN"],
            "required_channels": int(sensor["channels"]),
            "last_channel_to_coil_mm": float(fw["SENSOR_A_LAST_TO_COIL_MM"]),
            "fire_offset_mm": float(fw.get("FIRE_OFFSET_DEFAULT_MM", 0.0)),
            "on_time_us": float(fw["FIRE_DEFAULT_ON_US"]),
            "on_time_max_us": float(fw["FIRE_MAX_ON_US"]),
            "on_time_overhead_us": float(fw["FIRE_ON_US_OVERHEAD"]),
            "trigger_lead_us": float(fw["FIRE_TRIGGER_LEAD_US"]),
            "trigger_slip_us": float(fw["FIRE_TRIGGER_SLIP_US"]),
            "capture_window_ms": float(fw["CAPTURE_WINDOW_MS"]),
            "trigger_timeout_ms": float(fw["SHOT_TRIGGER_TIMEOUT_MS"]),
        },
        "firing_return": _firing_return(fw, sensor),
        "coil": {
            "num_turns": int(fw["COIL_N_TURNS"]),
            "loop_center_radius_mm": 15.0,
            "length_mm": 30.0,
            "bore_radius_mm": float(geom["bores"]["coil_mm"]) / 2.0,
            "face_in_x_mm": coil["face_in_x"],
            "face_out_x_mm": coil["face_out_x"],
            "former_length_mm": coil["former_length_mm"],
        },
        "circuit": {
            "can_capacitance_uF": float(
                fw.get("BANK_UNIT_UF_PULSE", fw["BANK_UNIT_UF"])),
            # Per-configuration large-signal pairs, taken from the firmware's
            # measured constants where they exist. See rig_profile.bank() for
            # why n x per-can cannot express the droop trend these carry.
            "pulse_measured_by_cans": {
                "_note": 'Per-configuration LARGE-SIGNAL constants from bench blank-fire fits, R pinned at the measured ESR convention (vbench firmware/config.py BANK_*_UF_PULSE / LOOP_R_MOHM_PULSE_*). Electrolytic droop makes pulse C sub-linear in can count (C_pulse/C_100Hz = 0.859/0.931/0.958 at 1/2/3), so n x can_uF understates C by 6-8% at 2-3 cans and overstates the waveform speed exactly where the marble on-time data probes it. Configs absent here fall back to the n x can_uF scaling. The 3-can C carries a caveat: no single linear C fits that discharge (rms 0.015); 5350 is the best single-number summary.',
                **{str(n): _pulse_entry(fw, n, c_key, r_key)
                for n, c_key, r_key in (
                    (1, "BANK_UNIT_UF_PULSE", "LOOP_R_MOHM_PULSE"),
                    (2, "BANK_2CAN_UF_PULSE", "LOOP_R_MOHM_PULSE_2CAN"),
                    (3, "BANK_3CAN_UF_PULSE", "LOOP_R_MOHM_PULSE_3CAN"),
                    (4, "BANK_4CAN_UF_PULSE", "LOOP_R_MOHM_PULSE_4CAN"),
                    (5, "BANK_5CAN_UF_PULSE", "LOOP_R_MOHM_PULSE_5CAN"),
                )
                if c_key in fw and r_key in fw},
            },
            "can_esr_ohm": profile_bank_esr,
            "bank_positions": int(fw["BANK_POSITIONS"]),
            "cans_populated": 1,
            "charge_voltage_V": 50.0,
            "voltage_max_V": float(fw["VBANK_MAX_V"]),
            "measured": {
                # Scope-fitted, not a firmware constant: see freewheel_note.
                # Kept here so regeneration does not silently drop it.
                "freewheel_tau_us": 275.0,
                "freewheel_note": "Fitted 2026-09-05 from the three scope-injected marble points (400/700/1500 us gates): tau ~275 us is the unique tail that makes all three measured-current predictions agree with the marble data (ratios 1.03/1.01/1.04). Independently corroborated: it implies freewheel R ~65 mohm, which fits under the 10 V capture's model-free TOTAL loop R of 80 mohm -- whereas the 107 mohm 'coil+leads' figure above cannot (it exceeds the whole measured loop). That 107 figure is now considered wrong at pulse conditions; it remains recorded as the 1 kHz bench measurement it was. When present, this tau overrides L/R_coil for the freewheel decay.",
                "inductance_uH": float(fw["COIL_L_UH_NOMINAL"]),
                "coil_resistance_ohm": float(fw["COIL_R_MOHM_NOMINAL"]) / 1000.0,
                "loop_resistance_ohm": float(fw["LOOP_R_MOHM_MEASURED"]) / 1000.0,
                "note": (
                    "L by 4-wire LCR at 10kHz (Q~8; the 1kHz reading is soft at "
                    "Q~1). loop_resistance is the whole discharge path and is "
                    "what sets the pulse -- confirmed twice, by on-time sweep "
                    "fit (0.164) and DC injection (0.161). Do not conflate it "
                    "with coil_resistance, which is coil+leads only."),
            },
        },
        "marble": {
            "diameter_mm": float(ball["diameter_mm"]),
            "radius_mm": float(ball["diameter_mm"]) / 2.0,
            # Prefer a weighed mass when the rig has one. Falls back to the
            # solid-steel estimate so an older rig_geometry.json still imports.
            "mass_kg": float(ball.get("mass_kg", ball["mass_kg_assumed"])),
            "mass_is_assumed": "mass_kg" not in ball,
            "mass_note": ball.get("_note", ""),
        },
        "track": {
            "flat_zone_x_mm": geom["profile"]["flat_zone_x"],
            "ramp_angle_deg": geom["profile"]["ramp_angle_deg"],
            "ramp_rise_mm": geom["profile"]["ramp_rise_mm"],
            "release_v_ideal_mps": geom["profile"]["release_v_ideal_mps"],
            "overall_length_mm": geom["overall_length_mm"],
            "ball_centre_z_mm": geom["datums"]["ball_centre_z"],
            "track_bore_mm": geom["bores"]["track_mm"],
            **({"losses": losses} if losses else {}),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vbench", type=Path, default=DEFAULT_VBENCH,
                        help="path to the omnimarble-vbench checkout")
    parser.add_argument("--check", action="store_true",
                        help="verify the stored profile matches the rig; exit 1 on drift")
    args = parser.parse_args()

    imported = build_profile(args.vbench.resolve())

    if not PROFILE_PATH.exists():
        raise SystemExit(f"{PROFILE_PATH} not found -- create it with the "
                         "legacy profile first, then re-run to fill vbench_v0")

    doc = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    stored = doc.get("profiles", {}).get("vbench_v0")

    if args.check:
        if stored == imported:
            print("OK: vbench_v0 matches the rig")
            return 0
        print("DRIFT: config/rig_profile.json disagrees with the rig.", file=sys.stderr)
        for key in sorted(set(imported) | set(stored or {})):
            if (stored or {}).get(key) != imported.get(key):
                print(f"  - {key} differs", file=sys.stderr)
        print("Re-run without --check to update.", file=sys.stderr)
        return 1

    doc.setdefault("profiles", {})["vbench_v0"] = imported
    PROFILE_PATH.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote vbench_v0 to {PROFILE_PATH.relative_to(ROOT)}")
    print(f"  pitch {imported['sensing']['pitch_mm']} mm, "
          f"L {imported['circuit']['measured']['inductance_uH']} uH, "
          f"loop R {imported['circuit']['measured']['loop_resistance_ohm']} ohm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
