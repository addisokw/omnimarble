"""Measured-current injection (simulate_rig_shot --current-csv).

The bench proved no linear RLC describes the real discharge across voltages
(I_peak scales sub-linearly with V0, effective R rises with pulse speed), so
the sim can bypass its circuit and integrate force over a scope-measured
I(t). These tests pin the loader's alignment and interpolation, and that the
injected waveform actually reaches the force integral.
"""
import csv
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from simulate_rig_shot import load_measured_current  # noqa: E402


def _write_capture(path, gate_on_s=1.0e-3, i_pk=100.0, rise_s=300e-6,
                   fall_s=400e-6, dt=2e-6, noise=None):
    """Synthetic scope CSV: quiet, triangular pulse, quiet."""
    rows = []
    t = -2e-3
    while t < 4e-3:
        rel = t - gate_on_s
        if rel < 0 or rel > rise_s + fall_s:
            i = 0.0
        elif rel <= rise_s:
            i = i_pk * rel / rise_s
        else:
            i = i_pk * (1 - (rel - rise_s) / fall_s)
        if noise:
            i += noise * math.sin(t * 1e5)
        rows.append((t, i, 49.0))
        t += dt
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "i_A", "v_bank_V"])
        w.writerows(rows)


def test_alignment_finds_gate_on(tmp_path):
    p = tmp_path / "cap.csv"
    _write_capture(p, gate_on_s=1.0e-3)
    fn, peak = load_measured_current(p)
    assert peak == pytest.approx(100.0, abs=1.0)
    # t_rel = 0 must sit at (or just before) the current rise: near-zero
    # current at 0, clearly rising shortly after.
    assert fn(0.0) < 8.0
    # The rise detector needs a few samples above threshold, so t_rel=0 sits
    # up to ~20 us into a SLOW synthetic ramp (real captures rise 100x faster
    # and the lag is ~2 us). Tolerance covers that inherent offset.
    assert fn(100e-6) == pytest.approx(100.0 * 100e-6 / 300e-6, rel=0.25)
    # peak lands at the rise time relative to gate-on
    assert fn(300e-6) == pytest.approx(100.0, rel=0.05)


def test_interpolation_and_clipping(tmp_path):
    p = tmp_path / "cap.csv"
    _write_capture(p, noise=1.5)
    fn, _ = load_measured_current(p)
    # outside the record: zero, never an extrapolation
    assert fn(-1.0) == 0.0
    assert fn(10.0) == 0.0
    # interpolation never returns negative current (diode-clamped loop)
    assert all(fn(t * 1e-6) >= 0.0 for t in range(0, 900, 7))


def test_injected_current_is_used_by_the_shot(tmp_path):
    """A stronger injected waveform must produce a larger dv than a weaker
    one, with the circuit model held identical -- proving the injection is
    live in the force path and not silently ignored."""
    import json
    from rig_profile import load_profile
    from simulate_rig_shot import simulate_shot
    from warp_bfield_solver import WarpBFieldSolver

    profile = load_profile(ROOT, "vbench_v0")
    params = json.loads((ROOT / "config" / "coil_params.json").read_text())
    params["num_turns"] = profile.coil["num_turns"]
    params["length_mm"] = profile.coil["length_mm"]
    solver = WarpBFieldSolver(params, chi_eff=3.0)
    dvs = []
    for scale in (0.5, 1.0):
        p = tmp_path / ("cap_%s.csv" % scale)
        _write_capture(p, i_pk=280.0 * scale)
        fn, peak = load_measured_current(p)
        shot = simulate_shot(profile, solver, 3, 49.0, 0.247,
                             on_time_us=700, fire_offset_mm=-13.78,
                             current_fn=fn, current_peak=peak)
        assert not shot.get("aborted")
        dvs.append(shot["dv_true_mps"])
    assert dvs[1] > dvs[0] * 2.0, (
        "doubling the injected current must far more than double dv "
        "(force ~ I^2); got %r" % dvs)
