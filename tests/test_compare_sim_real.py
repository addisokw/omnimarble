"""Tests for the sim-vs-real comparison guards.

These exist because every one of them protects against a way of getting a
plausible-looking plot out of data that cannot support it:

  * a stale 22.14 -> 11.0 sensor pitch makes every dv wrong by 2x, and nothing
    downstream would notice;
  * a station that gave 3 of 5 channels is a different measurement from a clean
    one, and the raw dv cannot tell you which you had;
  * pairing on can count alone compares shots taken at different release
    heights, i.e. different experiments;
  * fitting a scale factor between measured and simulated dv is circular --
    dv is linear in pitch, so the fit can always succeed.
"""

import pytest

from compare_sim_real import (
    REQUIRED_CHANNELS,
    check_pitch,
    fit_pitch_scale,
    pair_shots,
    quality_filter,
)


def _row(**overrides):
    row = {
        "cans": "1", "C_uF": "1909", "v_in_mps": "1.006", "v_out_mps": "1.019",
        "dv_mps": "0.0125", "n_ch_in": "5", "n_ch_out": "5",
        "resid_in_us": "120", "resid_out_us": "95", "sensor_pitch_mm": "22.14",
    }
    row.update({k: str(v) for k, v in overrides.items()})
    return row


# -- pitch --------------------------------------------------------------------

def test_current_pitch_is_accepted():
    assert check_pitch([_row()], "real") == {22.14}


def test_stale_pitch_is_a_hard_failure():
    """11.0 halves every velocity; comparing it silently would be worse than
    crashing, so this must raise rather than warn."""
    with pytest.raises(SystemExit, match="2.0127"):
        check_pitch([_row(sensor_pitch_mm=11.0)], "real")


def test_stale_pitch_is_caught_even_among_good_rows():
    with pytest.raises(SystemExit):
        check_pitch([_row(), _row(sensor_pitch_mm=11.0), _row()], "real")


def test_missing_pitch_column_is_tolerated():
    """Older logs predate the column; warn rather than refuse."""
    row = _row()
    del row["sensor_pitch_mm"]
    assert check_pitch([row], "real") == set()


# -- quality ------------------------------------------------------------------

def test_incomplete_capture_is_dropped():
    kept = quality_filter([_row(), _row(n_ch_out=3)], "real")
    assert len(kept) == 1
    assert kept[0]["n_ch_out"] == str(REQUIRED_CHANNELS)


def test_large_residual_is_dropped():
    """A big residual means the straight-line fit is not describing the motion."""
    kept = quality_filter([_row(), _row(resid_in_us=26546)], "real")
    assert len(kept) == 1


def test_shot_without_dv_is_dropped():
    kept = quality_filter([_row(), _row(dv_mps="")], "real")
    assert len(kept) == 1


def test_clean_shots_all_survive():
    rows = [_row(), _row(cans=2), _row(cans=3)]
    assert len(quality_filter(rows, "real")) == 3


# -- pairing ------------------------------------------------------------------

def test_pairs_on_nearest_v_in_within_a_can_bucket():
    """The rig varies release height deliberately, so v_in must drive pairing."""
    real = [_row(cans=1, v_in_mps=0.80)]
    sim = [_row(cans=1, v_in_mps=1.006, dv_mps=0.0125),
           _row(cans=1, v_in_mps=0.81, dv_mps=0.0140)]
    (_, matched), = pair_shots(real, sim, "nearest-v-in")
    assert matched["v_in_mps"] == "0.81"


def test_pairing_never_crosses_can_counts():
    real = [_row(cans=3, v_in_mps=1.0)]
    sim = [_row(cans=1, v_in_mps=1.0)]
    assert pair_shots(real, sim, "nearest-v-in") == []


def test_cans_mode_ignores_v_in():
    real = [_row(cans=1, v_in_mps=0.5)]
    sim = [_row(cans=1, v_in_mps=1.006, dv_mps=0.0125)]
    assert len(pair_shots(real, sim, "cans")) == 1


# -- scale fit ----------------------------------------------------------------

def test_fitted_scale_recovers_a_known_offset():
    """A measured dv 15% below sim must fit k = 1/0.85."""
    pairs = []
    for dv_sim in (0.0125, 0.0176, 0.0221):
        pairs.append((_row(dv_mps=round(dv_sim * 0.85, 6)),
                      _row(dv_mps=dv_sim)))
    assert fit_pitch_scale(pairs) == pytest.approx(1 / 0.85, rel=1e-6)


def test_perfect_agreement_fits_unity():
    pairs = [(_row(dv_mps=0.0125), _row(dv_mps=0.0125))]
    assert fit_pitch_scale(pairs) == pytest.approx(1.0, rel=1e-12)


def test_stale_firmware_shows_up_as_a_factor_of_two():
    """The diagnostic value of --fit-pitch: k near 2 means stale firmware.

    A rig still running SENSOR_PITCH_MM = 11.0 reports half the true velocity,
    so its dv is half and the fit lands near 22.14/11.0.
    """
    pairs = [(_row(dv_mps=0.0125 * 11.0 / 22.14), _row(dv_mps=0.0125))]
    assert fit_pitch_scale(pairs) == pytest.approx(22.14 / 11.0, rel=1e-6)
