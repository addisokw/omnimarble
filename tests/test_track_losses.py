"""Track-loss model and its fitter.

Synthetic-parameter recovery goes THROUGH the model's own forward functions
(integrate_flat + station_read), so the station-fit bias on a decelerating
return pass is part of what the fitter has to see through -- exactly as it
will on the bench rows.
"""

from pathlib import Path

import pytest

from fit_track_losses import (
    ReturnPassModel,
    fit_b_side,
    fit_excursion,
    fit_flat,
    load_baselines,
    pair_cycles,
    pairs_to_edge,
)
from rig_profile import load_profile
from sustain_model import (
    BSideExcess,
    Excursion,
    FLAT_ZONE_X_MM,
    FlatLoss,
    TrackLosses,
    integrate_flat,
    station_read,
)

ROOT = Path(__file__).resolve().parent.parent
HALF = 5.2
DT = 2e-3


@pytest.fixture(scope="module")
def specs():
    profile = load_profile(ROOT, "vbench_v0")
    s = profile.station_specs()
    for name, spec in s.items():
        spec["name"] = name
    return s


def test_round_trip(tmp_path):
    losses = TrackLosses(FlatLoss(0.02, 0.8), BSideExcess(1.1, 0.3),
                         Excursion(0.85, 0.01), Excursion(0.9, 0.005, c0=-0.01, c1=-0.05),
                         meta={"n": 3})
    path = tmp_path / "losses.json"
    losses.save(path)
    back = TrackLosses.load(path)
    assert back.to_dict() == losses.to_dict()


def test_flat_closed_form_matches_its_inverse():
    flat = FlatLoss(0.05, 0.8)
    v_out = flat.v_after_distance(0.5, 0.204)
    assert 0.0 < v_out < 0.5
    assert flat.v_before_distance(v_out, 0.204) == pytest.approx(0.5, rel=1e-9)
    assert FlatLoss(0.0, 0.0).v_after_distance(0.5, 1.0) == 0.5


def test_flat_loss_integrates_to_the_closed_form(specs):
    flat = FlatLoss(0.05, 0.8)
    losses = TrackLosses(flat=flat)
    channels = {n: s["channel_x_mm"] for n, s in specs.items()}
    seg = integrate_flat(losses, -FLAT_ZONE_X_MM, 0.5, +1, channels, HALF, dt=1e-3)
    assert seg.reason == "x_stop"
    expect = flat.v_after_distance(0.5, 2 * FLAT_ZONE_X_MM / 1000.0)
    assert seg.v_mps == pytest.approx(expect, rel=2e-3)


def test_fit_flat_recovers_synthetic_parameters():
    truth = FlatLoss(0.05, 0.8)
    pts = [(v, truth.v_after_distance(v, 0.204) - v) for v in (0.2, 0.3, 0.8)]
    flat, meta = fit_flat(pts)
    assert flat.a0 == pytest.approx(0.05, abs=2e-3)
    assert flat.k == pytest.approx(0.8, rel=0.02)
    assert meta["n"] == 3 and meta["rms_mps"] < 1e-4


def test_fit_flat_on_the_measured_roll_baselines():
    pts = load_baselines(ROOT / "data" / "roll_baselines.csv")
    assert len(pts) == 3
    flat, meta = fit_flat(pts)
    assert flat.a0 >= 0.0
    assert 0.4 < flat.k < 1.5, flat.to_dict()
    assert meta["rms_mps"] < 0.01


def test_fit_excursion_recovers_and_parks_below_threshold():
    truth = Excursion(0.85, 0.01)
    pairs = [(v, truth.v_back(v)) for v in (0.3, 0.45, 0.6, 0.9)]
    exc, meta = fit_excursion(pairs)
    assert exc.alpha == pytest.approx(0.85, rel=1e-6)
    assert exc.beta == pytest.approx(0.01, abs=1e-8)
    assert meta["n"] == 4 and meta["rms_mps"] < 1e-9
    # parking threshold: alpha v^2 <= beta returns None
    assert Excursion(0.8, 0.02).v_back(0.1) is None
    assert Excursion(0.8, 0.02).v_back(0.5) == pytest.approx((0.8 * 0.25 - 0.02) ** 0.5)
    assert Excursion(1.0, 0.0, form="linear", c0=-0.1, c1=0.0).v_back(0.05) is None


def test_pair_cycles_only_pairs_consecutive_kicks():
    rows = [
        {"run_id": "r1", "leg": "fwd", "kick_idx": 1, "v_in": 0.5, "v_out": 0.7},
        {"run_id": "r1", "leg": "ret", "kick_idx": 2, "v_in": 0.55, "v_out": 0.6},
        {"run_id": "r1", "leg": "fwd", "kick_idx": 3, "v_in": 0.45, "v_out": 0.65},
        {"run_id": "r1", "leg": "ret", "kick_idx": 5, "v_in": 0.5, "v_out": 0.55},
        {"run_id": "r2", "leg": "fwd", "kick_idx": 1, "v_in": 0.4, "v_out": 0.6},
    ]
    pairs = pair_cycles(rows)
    assert pairs["far"] == [(0.7, 0.55)]
    assert pairs["entry"] == [(0.6, 0.45)]


def test_return_fit_reads_high_on_a_decelerating_pass(specs):
    """The bias the twin must inherit: fit > local > true speed at the last channel."""
    losses = TrackLosses(flat=FlatLoss(0.0, 0.8), b_side=BSideExcess(0.5, 0.3))
    channels = {n: s["channel_x_mm"] for n, s in specs.items()}
    seg = integrate_flat(losses, FLAT_ZONE_X_MM, 0.5, -1, channels, HALF, dt=1e-3,
                         stop_after_station="B")
    read = station_read(seg.trips["B"], specs["B"], -1, HALF)
    assert read.complete
    v_true_last = seg.v_at(read.t_last)
    assert read.v_fit > read.v_local > v_true_last
    assert read.v_fit / v_true_last > 1.05


@pytest.fixture(scope="module")
def synthetic_return_rows(specs):
    # g must leave a 0.4 m/s ball enough speed to finish station B: the
    # excess + flat loss over the 108 mm to B's inner channel is ~0.09 m^2/s^2
    truth = TrackLosses(flat=FlatLoss(0.02, 0.8), b_side=BSideExcess(0.3, 0.4))
    model = ReturnPassModel(specs, HALF, dt=DT)
    rows = []
    for ve in (0.4, 0.5, 0.6, 0.75, 0.9):
        rd = model.read_B(truth, ve)
        assert rd is not None and rd.complete
        rows.append({"leg": "ret", "v_in": rd.v_fit, "v_local": rd.v_local})
    rB, rA = model.read_B_and_A(truth, 0.8)
    assert rA is not None
    return truth, rows, (rB.v_fit, rA.v_fit)


def test_fit_b_side_recovers_synthetic_parameters(specs, synthetic_return_rows):
    truth, rows, nokick = synthetic_return_rows
    b, meta = fit_b_side(rows, truth.flat, nokick=nokick, weight=3.0, specs=specs,
                         halfwidth_mm=HALF, dt=DT)
    # g and k trade off against each other over a narrow speed range; the
    # total excess at a mid speed is what the data pins.
    for v in (0.4, 0.6, 0.85):
        assert b.g + b.k * v * v == pytest.approx(0.3 + 0.4 * v * v, rel=0.1)
    assert meta["n_rows"] == 5
    assert meta["rms_v_local_mps"] < 0.01
    assert abs(meta["nokick_resid_mps"]) < 0.01


def test_no_kick_constraint_alone_fixes_the_excess(specs, synthetic_return_rows):
    truth, _, nokick = synthetic_return_rows
    b, meta = fit_b_side([], truth.flat, nokick=nokick, weight=3.0, specs=specs,
                         halfwidth_mm=HALF, dt=DT)
    model = ReturnPassModel(specs, HALF, dt=DT)
    fitted = TrackLosses(flat=truth.flat, b_side=b)
    ve = model.v_edge_for_B_fit(fitted, nokick[0])
    _, rA = model.read_B_and_A(fitted, ve)
    assert rA.v_fit == pytest.approx(nokick[1], abs=0.01)


def test_pairs_to_edge_round_trips_the_far_excursion(specs):
    """Station pairs -> edge pairs must reproduce the alpha/beta the sim applies."""
    truth = TrackLosses(flat=FlatLoss(0.02, 0.8), b_side=BSideExcess(0.3, 0.4),
                        far=Excursion(0.85, 0.01))
    model = ReturnPassModel(specs, HALF, dt=DT)
    d_B = (FLAT_ZONE_X_MM - abs(specs["B"]["channel_x_mm"][2])) / 1000.0
    pairs = []
    for v_B in (0.55, 0.65, 0.8, 1.0):
        # forward pass through B at ~v_B, out to the edge, back through B
        v_edge_out = truth.flat.v_after_distance(v_B, d_B)
        v_edge_back = truth.far.v_back(v_edge_out)
        rd = model.read_B(truth, v_edge_back)
        assert rd is not None and rd.complete, v_B
        pairs.append((v_B, rd.v_fit))
    edge = pairs_to_edge(pairs, "far", truth, specs, HALF, DT)
    exc, meta = fit_excursion(edge)
    assert exc.alpha == pytest.approx(0.85, rel=0.03)
    assert exc.beta == pytest.approx(0.01, abs=0.005)
