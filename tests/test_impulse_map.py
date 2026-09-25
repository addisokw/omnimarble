"""Impulse map: build, store, look up, mirror and rescale.

Built on the stub solver over a tiny grid so the suite stays fast; the map
schema and the lookup rules are what is under test, not the field model.
"""

import json
from pathlib import Path

import pytest

from impulse_map import (
    ImpulseMap,
    SCHEMA,
    build_map,
    check_rescale,
    load_map,
    save_map,
)
from rig_profile import load_profile
from solver_stubs import GradientSolver

ROOT = Path(__file__).resolve().parent.parent
X_GRID = [-25.0, -15.0, -10.0]
V_IN = [0.5, 0.8]


@pytest.fixture(scope="module")
def profile():
    return load_profile(ROOT, "vbench_v0")


@pytest.fixture(scope="module")
def map50(profile):
    return build_map(profile, GradientSolver(), cans=1, gate_us=200.0,
                     current_fn=None, current_peak=None, x_grid_mm=X_GRID,
                     v_in_list=V_IN, dt=4e-5, max_time_s=4.0, voltage=50.0,
                     finite_size=False)


@pytest.fixture(scope="module")
def map30(profile):
    return build_map(profile, GradientSolver(), cans=1, gate_us=200.0,
                     current_fn=None, current_peak=None, x_grid_mm=X_GRID,
                     v_in_list=V_IN, dt=4e-5, max_time_s=4.0, voltage=30.0,
                     finite_size=False)


def _synthetic(fn, xs=(-2.0, -1.0, 0.0, 1.0, 2.0), vs=(0.2, 0.8), v_bank=50.0):
    return ImpulseMap({
        "schema": SCHEMA, "cans": 4, "gate_us": 700.0, "v_bank_V": v_bank,
        "x_mm": list(xs), "v_in_mps": list(vs),
        "dv_mps": [[fn(x, v) for x in xs] for v in vs],
        "voltage_scaling": {"exponent": 2.0, "basis_V": v_bank},
    })


def test_schema_and_round_trip(map50, tmp_path):
    for key in ("schema", "cans", "gate_us", "capture", "v_bank_V", "i_peak_A",
                "profile", "tag", "x_mm", "v_in_mps", "dv_mps",
                "voltage_scaling", "known_bias", "generated"):
        assert key in map50, key
    assert map50["schema"] == SCHEMA
    assert len(map50["dv_mps"]) == len(V_IN)
    assert all(len(row) == len(X_GRID) for row in map50["dv_mps"])
    assert map50["i_peak_A"] > 0
    path = save_map(map50, tmp_path / "m.json")
    loaded = load_map(path)
    assert loaded.x == X_GRID and loaded.v == V_IN
    assert loaded.dv == map50["dv_mps"]
    assert json.loads(path.read_text())["voltage_scaling"]["exponent"] == 2.0


def test_dv_is_positive_and_peaks_inside_the_coil_face(map50):
    imap = ImpulseMap(map50)
    for v in V_IN:
        assert all(imap.dv_at(x, v) > 0 for x in X_GRID)
    # the stub's field peaks at -15 mm; the map must see more impulse there
    # than at the face
    assert imap.dv_at(-15.0, 0.5) > imap.dv_at(-25.0, 0.5)


def test_weak_v_in_dependence(map50):
    """A fixed on-time pulse does not reward a slower ball (test_rig_shot)."""
    imap = ImpulseMap(map50)
    for x in X_GRID:
        assert imap.v_in_spread(x) < 0.15


def test_bilinear_interpolation_is_exact_on_a_linear_field():
    imap = _synthetic(lambda x, v: 0.1 + 0.01 * x + 0.05 * v)
    for x, v in ((-1.5, 0.35), (0.25, 0.8), (1.9, 0.2), (0.0, 0.5)):
        dv, clamped = imap.lookup(x, v)
        assert not clamped
        assert dv == pytest.approx(0.1 + 0.01 * x + 0.05 * v, abs=1e-12)


def test_lookup_outside_the_grid_clamps_and_flags():
    imap = _synthetic(lambda x, v: 0.1 + 0.01 * x + 0.05 * v)
    dv, clamped = imap.lookup(5.0, 0.5)
    assert clamped
    assert dv == pytest.approx(imap.dv_at(2.0, 0.5))
    dv, clamped = imap.lookup(0.0, 1.5)
    assert clamped
    assert dv == pytest.approx(imap.dv_at(0.0, 0.8))
    assert not imap.lookup(2.0, 0.8)[1], "the grid edge itself is inside"


def test_mirrored_is_the_reflected_lookup():
    imap = _synthetic(lambda x, v: 0.1 + 0.03 * x)
    for x in (-1.5, 0.0, 0.7, 2.0):
        assert imap.mirrored(x, 0.5) == pytest.approx(imap.dv_at(-x, 0.5))
    assert imap.mirrored(1.5, 0.5) < imap.mirrored(-1.5, 0.5)


def test_rescale_factor_is_voltage_squared():
    imap = _synthetic(lambda x, v: 0.1, v_bank=50.0)
    assert imap.rescale_factor(40.0) == pytest.approx(0.64)
    assert imap.rescale_factor(50.0) == pytest.approx(1.0)


def test_v_squared_rescale_reproduces_a_map_built_at_another_voltage(map50, map30):
    """The one direct test of the rescale rule (bench: 3-can 30/49 V pair).

    The stub circuit is linear so I scales with V and force with V^2; the
    diode tail (constant Vf) is the only non-linearity, worth a few percent.
    """
    a, b = ImpulseMap(map50), ImpulseMap(map30)
    r = check_rescale(a, b)
    assert r["n"] == len(X_GRID) * len(V_IN)
    assert r["factor"] == pytest.approx(0.36)
    assert r["mean_ratio"] == pytest.approx(1.0, abs=0.08), r
    assert r["rms_dev"] < 0.08, r


def test_map_rejects_wrong_schema_and_shape():
    with pytest.raises(ValueError):
        ImpulseMap({"schema": "other"})
    with pytest.raises(ValueError):
        ImpulseMap({"schema": SCHEMA, "cans": 4, "gate_us": 700, "v_bank_V": 50,
                    "x_mm": [0, 1], "v_in_mps": [0.5], "dv_mps": [[1.0]]})
