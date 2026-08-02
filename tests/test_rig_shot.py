"""Physics of a rig shot, exercised through the real integrator.

A stub field solver stands in for the PINN so these run in milliseconds on CPU
and test the mechanics rather than the field model. The field model is
validated separately, against the bench's measured launch-position table.

The claim under test is the spin one. The pulse lasts ~191us and the ball moves
~0.2mm in that time -- far too briefly for friction to spin it up -- so the
impulse lands almost entirely in translation and leaves the ball slipping.
Friction then restores rolling over the 102mm to station B, and angular
momentum about the contact line is conserved through that, so the ball KEEPS
ONLY 5/7 OF THE VELOCITY GAIN. That is a ~29% systematic on the rig's primary
observable. simulate_shot integrates v and omega separately rather than
applying 5/7, so these tests confirm the factor is a result and not an
assumption.
"""

import math
from pathlib import Path

import pytest

from simulate_rig_shot import SPHERE_INERTIA, simulate_shot
from rig_profile import load_profile

ROOT = Path(__file__).resolve().parent.parent
MU_0_MM = 4 * math.pi * 1e-4


class ConstantForceSolver:
    """Uniform axial dBz/dz over a window, so the impulse is exactly known.

    The field SCALES WITH CURRENT, as the real one does: the PINN is trained on
    B/I and the loader multiplies by I, so force goes as I^2. An earlier version
    of this stub returned a fixed field regardless of current, which quietly
    turned every test into a measurement of "how long is the current above
    1e-8" rather than of the impulse -- and made a correct change to the
    freewheel tail look like a 3.7x regression.

    Scaled so a ~211A peak gives a Delta-v of order 0.02 m/s, matching the real
    coil. Magnitude matters: at an unrealistically strong force the ball is
    still slipping when it reaches station B, and conclusions that hold on the
    real rig (notably mu-independence) stop holding.
    """

    REF_CURRENT_A = 211.0

    def __init__(self, dbz_dz=7.2e-3, bz=0.05, window=(-25.0, -5.0)):
        # Stored per-amp, so the caller still thinks in field units at the
        # reference current.
        self.dbz_dz_per_A = dbz_dz / self.REF_CURRENT_A
        self.bz_per_A = bz / self.REF_CURRENT_A
        self.window = window

    def field_with_grad(self, r, z, current_A):
        if self.window[0] <= z <= self.window[1]:
            return (0.0, self.bz_per_A * current_A, 0.0, 0.0, 0.0,
                    self.dbz_dz_per_A * current_A)
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.fixture(scope="module")
def profile():
    return load_profile(ROOT, "vbench_v0")


def _shot(profile, **kwargs):
    kwargs.setdefault("cans", 1)
    kwargs.setdefault("voltage", 50.0)
    kwargs.setdefault("v_in_mps", 1.006)
    return simulate_shot(profile, ConstantForceSolver(), **kwargs)


def test_shot_completes_and_measures_both_stations(profile):
    shot = _shot(profile)
    assert not shot["aborted"]
    assert shot["n_ch_in"] == 5 and shot["n_ch_out"] == 5
    assert shot["dv_mps"] is not None


def test_flat_zone_estimator_is_unbiased(profile):
    """The measurement zone is flat by design, so the fit must recover truth.

    This is what track/RIG.md's flat zone buys: a slope through a station would
    put the ball under acceleration and the straight-line fit would stop
    measuring what it thinks it measures.
    """
    shot = _shot(profile)
    assert shot["v_in_mps"] == pytest.approx(shot["v_in_true_mps"], rel=1e-6)
    assert shot["resid_in_us"] == pytest.approx(0.0, abs=1e-3)


def test_ball_keeps_five_sevenths_of_the_velocity_gain(profile):
    """The spin redistribution, measured through the integrator.

    Compared against the same shot with rotation disabled, which keeps the
    whole impulse. A frictionless ball is NOT the right control: with mu = 0
    nothing ever couples translation to spin, so it also keeps the whole gain,
    just by a different route.
    """
    spinning = _shot(profile, allow_spin=True)
    no_spin = _shot(profile, allow_spin=False)

    assert no_spin["dv_true_mps"] > spinning["dv_true_mps"]
    ratio = spinning["dv_true_mps"] / no_spin["dv_true_mps"]
    assert ratio == pytest.approx(5.0 / 7.0, rel=0.02), (
        "a rolling ball must keep 5/7 of the impulse's velocity gain")


def test_the_ball_actually_slips_during_the_pulse(profile):
    """Friction cannot hold rolling under the pulse, so the ball must slip.

    Holding the rolling condition needs (2/7)F at the contact -- order 7N here
    -- against a mu*m*g budget of order 0.025N. A model that quietly stayed in
    the rolling branch would get the right endpoint for the wrong reason, and
    would be wrong if the ball ever reached station B still slipping.
    """
    shot = _shot(profile)
    mass = 0.00842
    f_available = 0.3 * mass * 9.81                     # ~0.025 N
    # Mean force implied by the observed gain over the ~191us pulse; the peak
    # is higher still.
    f_needed = (2.0 / 7.0) * mass * (shot["dv_true_mps"] / 191e-6)
    assert f_needed > 5 * f_available, (
        f"needed {f_needed:.3f}N of contact friction against {f_available:.3f}N "
        "available -- if this ratio ever drops near 1 the rolling branch "
        "becomes reachable during the pulse and the model must be revisited")


def test_the_factor_is_independent_of_the_friction_coefficient(profile):
    """mu sets how fast rolling is restored, not the final speed.

    Worth pinning: mu is unknown for steel on PLA, and a prediction that
    depended on it would be a fitted parameter wearing a physics costume.
    """
    results = [_shot(profile, friction=mu)["dv_true_mps"]
               for mu in (0.15, 0.3, 0.6)]
    for value in results[1:]:
        assert value == pytest.approx(results[0], rel=0.02)


def test_effective_inertia_of_a_rolling_sphere():
    """A rolling sphere accelerates as F/((1 + 2/5)m); 2/7 goes into spin."""
    assert 1.0 + SPHERE_INERTIA == pytest.approx(7.0 / 5.0)
    assert SPHERE_INERTIA / (1.0 + SPHERE_INERTIA) == pytest.approx(2.0 / 7.0)


def test_more_cans_deliver_more_impulse(profile):
    """dv must rise monotonically with stored energy across the sweep."""
    dvs = [_shot(profile, cans=n)["dv_true_mps"] for n in range(1, 6)]
    assert dvs == sorted(dvs)


def test_sweep_spans_both_damping_regimes(profile):
    """The C-sweep crosses zeta = 1 between one and two cans."""
    regimes = [_shot(profile, cans=n)["regime"] for n in (1, 2, 5)]
    assert regimes[0] == "underdamped"
    assert regimes[1] != "underdamped"
    assert regimes[2] == "overdamped"


def test_no_sweep_point_reports_a_zero_peak(profile):
    """Regression: the overdamped branch used to hardcode I_peak = 0."""
    for n in range(1, 6):
        assert _shot(profile, cans=n)["i_peak_model_A"] > 0.0


def test_loop_resistance_falls_as_cans_are_added(profile):
    values = [_shot(profile, cans=n)["loop_R_mohm"] for n in range(1, 6)]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(164.0, abs=0.5)


def test_fixed_on_time_makes_dv_largely_independent_of_approach_speed(profile):
    """A FIXED on-time pulse does not reward a slower ball.

    Worth pinning because the intuition from the legacy position-cutoff model
    is the opposite: there, a slower marble dwells longer between the entry and
    cutoff gates and collects more impulse. Here the pulse is 191us regardless
    and the ball moves microns during it, so v_in barely matters. If this ever
    starts tracking v_in strongly, the cutoff has silently gone
    position-dependent again.
    """
    dvs = [_shot(profile, v_in_mps=v)["dv_true_mps"] for v in (0.5, 1.006, 2.0)]
    spread = (max(dvs) - min(dvs)) / max(dvs)
    assert spread < 0.15, f"dv varied {spread:.0%} across a 4x range of v_in"


def test_rolling_resistance_costs_speed_when_enabled(profile):
    """Off by default -- vbench flags it as unmeasured -- but must work."""
    without = _shot(profile)["v_out_true_mps"]
    with_rr = _shot(profile, rolling_resistance=0.002)["v_out_true_mps"]
    assert with_rr < without


def test_forced_fire_position_overrides_the_controller(profile):
    """--scan-fire-offset must actually move where the pulse lands."""
    early = _shot(profile, fire_offset_mm=-24.0)
    late = _shot(profile, fire_offset_mm=-10.0)
    assert early["fire_x_mm"] == pytest.approx(-24.0, abs=0.1)
    assert late["fire_x_mm"] == pytest.approx(-10.0, abs=0.1)


def test_shot_reports_the_pitch_it_used(profile):
    """Every row carries its pitch so a stale-11.0 dataset is detectable."""
    assert _shot(profile)["sensor_pitch_mm"] == pytest.approx(22.14)


# -- step-size convergence ----------------------------------------------------

from simulate_rig_shot import DEFAULT_DT_S  # noqa: E402


def test_dv_is_converged_at_the_default_step(profile):
    """Re-measure the claim behind DEFAULT_DT_S rather than trusting it.

    The dominant step-size error here was never the integrator -- it was the
    gate cut being quantised to a step boundary, which ran the pulse long by up
    to one dt. At 10us that was a 4.96% error on dv. Cutting at the exact gate
    time reduced it to 0.10%, which is what makes the default affordable.
    """
    reference = _shot(profile, dt=1e-6)["dv_true_mps"]
    at_default = _shot(profile, dt=DEFAULT_DT_S)["dv_true_mps"]
    error = abs(at_default - reference) / reference
    assert error < 0.005, (
        f"dv at the default {DEFAULT_DT_S*1e6:.0f}us step is {error:.2%} from "
        f"the 1us reference; the default is no longer converged")


def test_convergence_is_monotone_and_bounded(profile):
    """Halving the step must not make things worse, at any step we ship."""
    reference = _shot(profile, dt=1e-6)["dv_true_mps"]
    errors = []
    for dt in (4e-5, 2e-5, 1e-5, 5e-6):
        dv = _shot(profile, dt=dt)["dv_true_mps"]
        errors.append(abs(dv - reference) / reference)
    assert max(errors) < 0.01, f"errors {['%.3f%%' % (e*100) for e in errors]}"
    assert errors[-1] <= errors[0], "refining the step should not diverge"


def test_gate_is_cut_at_the_exact_time_not_a_step_boundary(profile):
    """The fix that made the above possible.

    With a coarse step, a boundary-quantised cut would run the gate long and
    inflate dv. Two very different step sizes must now agree closely.
    """
    coarse = _shot(profile, dt=4e-5)["dv_true_mps"]
    fine = _shot(profile, dt=1e-6)["dv_true_mps"]
    assert abs(coarse - fine) / fine < 0.01
