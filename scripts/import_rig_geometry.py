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


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_firmware_config(path):
    """Pull module-level literal constants out of firmware/config.py via ast."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in FIRMWARE_CONSTANTS:
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
            "detect_halfwidth_mm": 9.68 / 2.0,
            "resid_warn_us": float(fw["SENSOR_RESID_WARN_US"]),
        },
        "firing": {
            "mode": "fixed_on_time",
            "trigger_station": fw["STATION_IN"],
            "required_channels": int(sensor["channels"]),
            "last_channel_to_coil_mm": float(fw["SENSOR_A_LAST_TO_COIL_MM"]),
            "on_time_us": float(fw["FIRE_DEFAULT_ON_US"]),
            "on_time_max_us": float(fw["FIRE_MAX_ON_US"]),
            "on_time_overhead_us": float(fw["FIRE_ON_US_OVERHEAD"]),
            "trigger_lead_us": float(fw["FIRE_TRIGGER_LEAD_US"]),
            "trigger_slip_us": float(fw["FIRE_TRIGGER_SLIP_US"]),
            "capture_window_ms": float(fw["CAPTURE_WINDOW_MS"]),
            "trigger_timeout_ms": float(fw["SHOT_TRIGGER_TIMEOUT_MS"]),
        },
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
            "can_capacitance_uF": float(fw["BANK_UNIT_UF"]),
            "can_esr_ohm": profile_bank_esr,
            "bank_positions": int(fw["BANK_POSITIONS"]),
            "cans_populated": 1,
            "charge_voltage_V": 50.0,
            "voltage_max_V": float(fw["VBANK_MAX_V"]),
            "measured": {
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
            "mass_kg": float(ball["mass_kg_assumed"]),
            "mass_is_assumed": True,
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
