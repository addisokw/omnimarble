"""Regression guard for the committed Kit trajectory fixture.

The 300V run in results/trajectories/ is the only artifact proving the real
Kit/PhysX/PINN path reproduces across machines. Nothing guarded it -- the README
just said "rerun the autorun command and diff" -- so any change to the circuit,
geometry or firing logic could silently invalidate it.

Two things are checked, and the split matters:

  * The PINN/circuit chain is deterministic and must match EXACTLY. Current,
    capacitor voltage, Bz and the thermal/flag columns are pure functions of
    time and the marble's position, so a real cross-machine rerun reproduces
    them bit-for-bit.
  * PhysX itself is not bit-reproducible across machines. Positions drift by
    ~0.07mm and velocities by ~0.2% between an RTX 5090 and an RTX 5080 run of
    the identical scene. Those columns get a tolerance, not a checksum.

Comparison is by column-name intersection so that adding columns to
TRAJ_COLUMNS later does not false-fail this test.

Also runnable directly to diff a fresh run against the fixture:

    uv run python tests/test_trajectory_regression.py --compare-to <fresh.csv>
"""

import csv
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = (ROOT / "results" / "trajectories"
           / "kit_launch_300V_470uF_20260702_174018.csv")

# Pure functions of (t, position) -> reproduce bit-for-bit on any machine.
DETERMINISTIC_COLUMNS = (
    "t_s", "current_A", "V_cap_V", "Bz_T", "wire_temp_C", "triggered", "pulse_cut",
)

# PhysX solver output -- reproducible only to a tolerance across machines.
# Bounds are ~3x the drift actually observed between a 5090 and a 5080 run.
SOLVER_TOLERANCES_MM = {
    "x_mm": 0.2, "y_mm": 0.2, "z_mm": 0.2, "z_along_mm": 0.2, "r_mm": 0.2,
}
SOLVER_TOLERANCES_MM_S = {
    "vx_mm_s": 8.0, "vy_mm_s": 8.0, "vz_mm_s": 8.0, "vel_axial_mm_s": 8.0,
}

# The gate-measured results are the run's headline claim: 4.5x boost.
EXPECTED_META = {
    "charge_voltage_V": "300.0",
    "capacitance_uF": "470.0",
    "num_turns": "30",
    "chi_eff": "3.0",
    "inductance_uH": "12.4061",
    "R_total_ohm": "0.11021",
    "rlc_regime": "underdamped",
    "pinn_checkpoint": "pinn_best.pt",
    "pinn_step": "250000",
    "pinn_derived_b": "True",
    "approach_velocity_mm_s": "208.33",
    "exit_velocity_mm_s": "937.5",
    "boost_ratio": "4.5",
    "gate_vel_in_1_t_s": "0.194",
    "gate_vel_in_2_t_s": "0.29",
    "gate_entry_t_s": "0.386",
    "gate_cutoff_t_s": "0.418",
    "gate_vel_out_1_t_s": "0.482",
    "gate_vel_out_2_t_s": "0.546",
}

EXPECTED_ROW_COUNT = 528

# Digest of DETERMINISTIC_COLUMNS as committed. Regenerate ONLY after a rerun
# you have verified by hand:
#   uv run python -c "import sys; sys.path.insert(0,'tests'); \
#     from test_trajectory_regression import *; \
#     print(_digest(load_trajectory(FIXTURE)[2], DETERMINISTIC_COLUMNS))"
_PINNED_DIGEST = "1b1236de45d36b27cd71bc169667d4f13dbde62c3b3abed3a84eafa40a670a82"


def load_trajectory(path):
    """Parse a trajectory CSV into (meta, fieldnames, rows).

    Values stay as strings: the digest hashes the file's own text so it cannot
    drift with float repr, and callers convert only what they compare.
    """
    meta, data_lines = {}, []
    with open(path, newline="", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                key, _, value = line[1:].strip().partition("=")
                meta[key.strip()] = value.strip()
            else:
                data_lines.append(line)
    reader = csv.DictReader(data_lines)
    return meta, list(reader.fieldnames or []), list(reader)


def _digest(rows, columns):
    """Stable digest over the given columns, hashing the CSV's own text.

    Hashing text rather than parsed floats means this is exact and has no
    tolerance to tune -- a single changed digit trips it.
    """
    h = hashlib.sha256()
    for row in rows:
        h.update("|".join(row[c] for c in columns).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def shared_columns(ref_fields, new_fields, wanted):
    """Columns present in BOTH files, so added columns don't false-fail."""
    return [c for c in wanted if c in ref_fields and c in new_fields]


@pytest.fixture(scope="module")
def fixture_run():
    if not FIXTURE.exists():
        pytest.skip(f"trajectory fixture not present: {FIXTURE}")
    return load_trajectory(FIXTURE)


def test_fixture_metadata(fixture_run):
    """The run's parameters and its headline 4.5x boost claim."""
    meta, _, _ = fixture_run
    for key, expected in EXPECTED_META.items():
        assert meta.get(key) == expected, f"metadata drift on {key}"


def test_fixture_shape(fixture_run):
    meta, fields, rows = fixture_run
    assert len(rows) == EXPECTED_ROW_COUNT
    for column in DETERMINISTIC_COLUMNS:
        assert column in fields, f"fixture is missing {column}"


def test_boost_ratio_is_self_consistent(fixture_run):
    """Guard the claim itself, not just the recorded string."""
    meta, _, _ = fixture_run
    approach = float(meta["approach_velocity_mm_s"])
    exit_v = float(meta["exit_velocity_mm_s"])
    assert exit_v / approach == pytest.approx(float(meta["boost_ratio"]), rel=1e-3)


def test_deterministic_digest_is_stable(fixture_run):
    """Pin the deterministic columns against accidental edits to the fixture."""
    _, _, rows = fixture_run
    digest = _digest(rows, DETERMINISTIC_COLUMNS)
    # Pinned on first run; see module docstring for how to regenerate.
    assert digest == _PINNED_DIGEST, (
        "the fixture's deterministic columns changed. If this was an intended "
        "re-capture, update _PINNED_DIGEST; otherwise the fixture was corrupted."
    )


def compare_runs(ref_path, new_path):
    """Diff a fresh run against a reference. Returns a list of failure strings.

    Used by the __main__ block and intended for the post-change verification
    step: the deterministic columns must match exactly, the solver columns only
    within tolerance.
    """
    ref_meta, ref_fields, ref_rows = load_trajectory(ref_path)
    new_meta, new_fields, new_rows = load_trajectory(new_path)
    problems = []

    if len(ref_rows) != len(new_rows):
        problems.append(f"row count {len(ref_rows)} -> {len(new_rows)}")

    for key in ("approach_velocity_mm_s", "exit_velocity_mm_s", "boost_ratio"):
        if ref_meta.get(key) != new_meta.get(key):
            problems.append(
                f"gate-measured {key}: {ref_meta.get(key)} -> {new_meta.get(key)}")

    n = min(len(ref_rows), len(new_rows))
    exact = shared_columns(ref_fields, new_fields, DETERMINISTIC_COLUMNS)
    for column in exact:
        bad = [i for i in range(n) if ref_rows[i][column] != new_rows[i][column]]
        if bad:
            i = bad[0]
            problems.append(
                f"{column}: {len(bad)} rows differ; first at row {i} "
                f"({ref_rows[i][column]} -> {new_rows[i][column]})")

    tolerances = {**SOLVER_TOLERANCES_MM, **SOLVER_TOLERANCES_MM_S}
    for column in shared_columns(ref_fields, new_fields, tolerances):
        worst, worst_i = 0.0, -1
        for i in range(n):
            delta = abs(float(ref_rows[i][column]) - float(new_rows[i][column]))
            if delta > worst:
                worst, worst_i = delta, i
        if worst > tolerances[column]:
            problems.append(
                f"{column}: max drift {worst:.4g} exceeds {tolerances[column]} "
                f"(row {worst_i})")

    return problems


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-to", metavar="CSV", required=True,
                        help="fresh trajectory CSV to diff against the fixture")
    parser.add_argument("--reference", metavar="CSV", default=str(FIXTURE))
    args = parser.parse_args()

    failures = compare_runs(args.reference, args.compare_to)
    if failures:
        print(f"REGRESSION vs {Path(args.reference).name}:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print(f"OK: {Path(args.compare_to).name} matches "
          f"{Path(args.reference).name} within tolerance")
