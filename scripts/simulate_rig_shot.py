"""Simulate a shot on the vbench rig, and sweep the bank.

The rig's measurement zone is dead flat by design (track/RIG.md "Profile": a
slope through a station would put the ball under acceleration and the
least-squares velocity fit would stop measuring what it thinks it measures),
and the ball runs in a bore. So a shot is a one-dimensional problem: axial
position, translational speed, and spin. That is a far better model of this rig
than bouncing a sphere off a track mesh, and it makes the primary observable --
dv = v_out - v_in -- depend only on v_in, the circuit, and the field.

    uv run python scripts/simulate_rig_shot.py --cans 1
    uv run python scripts/simulate_rig_shot.py --sweep-cans 1,2,3,4,5 --shots-out out.csv
    uv run python scripts/simulate_rig_shot.py --scan-fire-offset -60:20:2.5

SPIN IS THE INTERESTING PART. The pulse lasts ~191 us, during which the ball
moves ~0.2 mm -- far too briefly for friction to spin it up, so the impulse goes
almost entirely into translation and the ball is left slipping. Friction then
re-establishes rolling over the 102 mm to station B. Angular momentum about the
contact line is conserved through that, so a ball already rolling at v_in which
receives an impulse J ends up measured at

    v_out = v_in + (5/7)(J/m)

i.e. it KEEPS ONLY 5/7 OF THE NAIVE VELOCITY GAIN. That is a ~29% systematic on
the rig's primary observable, it falls out of the geometry rather than being
fitted, and it is independent of the friction coefficient (mu sets how quickly
rolling is restored, not the final state). The model below integrates v and
omega separately rather than hardcoding 5/7, so the factor is a result.

Rolling RESISTANCE is a different thing and is deliberately zero by default:
vbench flags it as an unmeasured ~10% systematic over the 204 mm between
stations (NEXT_SESSION.md section 8). Inventing a value would launder a guess
into a prediction -- pass --rolling-resistance to explore it.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "source" / "extensions" / "omni.marble.coaster"
                       / "omni" / "marble" / "coaster"))

from coil_sensing import (  # noqa: E402
    FiringController,
    VirtualStation,
    crossed as channel_crossed,
    interpolate_crossing_us,
)
from rig_profile import load_profile  # noqa: E402
from rlc_circuit import (  # noqa: E402
    compute_rlc_params,
    eddy_braking_force,
    rlc_current,
)

MU_0_MM = 4 * math.pi * 1e-4
STEEL_DENSITY_G_MM3 = 7.8e-3
GRAVITY_MM_S2 = 9810.0
SPHERE_INERTIA = 2.0 / 5.0          # I = (2/5) m r^2
DEFAULT_FRICTION = 0.3              # steel on PLA; only sets how fast rolling
                                    # is restored, not the final speed
SLIP_TOL_MM_S = 1e-9                # below this the contact counts as rolling

# MEASURED, not assumed. With the gate cut taken at its exact time rather
# than at a step boundary, dv is converged to 0.10% at 10us and 0.11% at
# 20us against a 1us reference -- and 12x faster than 1us. Before that fix
# the same sweep read 4.96% at 10us, because the quantised cut ran the gate
# long by up to a full step. tests/test_rig_shot.py re-measures this.
DEFAULT_DT_S = 1e-5


_CONFIG_CACHE = {}


def _config_value(key, default):
    """Read a marble constant from config/coil_params.json."""
    if not _CONFIG_CACHE:
        path = ROOT / "config" / "coil_params.json"
        if path.exists():
            _CONFIG_CACHE.update(json.loads(path.read_text(encoding="utf-8")))
        _CONFIG_CACHE.setdefault("_loaded", True)
    return _CONFIG_CACHE.get(key, default)


def freewheel_current(i_at_cut, t_since_cut, resistance_ohm, inductance_H,
                      diode_vf_V=0.0):
    """Coil current after the switch opens, freewheeling through the diode.

    TWO things the naive exp(-R_loop*t/L) gets wrong, and the tail carries a
    large share of the impulse so both matter:

    1. R is the COIL's resistance, not the discharge loop's. Once the switch
       opens the capacitor bank is out of the circuit and its ESR no longer
       participates, so the current decays more slowly than the loop figure
       implies (tau 167us vs 109us on this rig).
    2. A diode is not a resistor. Its ~0.8V forward drop adds a constant
       opposing term, so L di/dt = -(I*R + Vf) and the current reaches
       EXACTLY ZERO in finite time rather than decaying asymptotically:

           I(t) = (I0 + Vf/R) * exp(-R t / L) - Vf/R

       That matters for a tail this long: at 211A it truncates the decay at
       ~557us instead of trailing off indefinitely.

    Falls back to the pure exponential when Vf is zero.
    """
    if t_since_cut <= 0.0:
        return i_at_cut
    if inductance_H <= 0.0 or resistance_ohm <= 0.0:
        return 0.0
    decay = math.exp(-(resistance_ohm / inductance_H) * t_since_cut)
    if diode_vf_V <= 0.0:
        return max(i_at_cut * decay, 0.0)
    offset = diode_vf_V / resistance_ohm
    return max((i_at_cut + offset) * decay - offset, 0.0)


def simulate_shot(profile, solver, cans, voltage, v_in_mps, on_time_us=None,
                  fire_offset_mm=None, rolling_resistance=0.0,
                  friction=DEFAULT_FRICTION, allow_spin=True,
                  dt=DEFAULT_DT_S, max_time_s=2.0):
    """One shot down the flat zone. Returns a dict of observables."""
    bank = profile.bank(cans)
    marble = profile.marble
    r_ball = float(marble["radius_mm"])
    mass_kg = float(marble["mass_kg"])
    V_marble = (4.0 / 3.0) * math.pi * r_ball ** 3
    # From the profile, falling back to config/coil_params.json. These used to
    # be hardcoded here, so editing marble_saturation_T changed Kit and the 3-D
    # mirror and silently did nothing to this model.
    chi_eff = float(marble.get("chi_eff", _config_value("marble_chi_eff", 3.0)))
    B_sat = float(marble.get("saturation_T",
                             _config_value("marble_saturation_T", 1.8)))
    conductivity = float(marble.get(
        "conductivity_S_per_m", _config_value("marble_conductivity_S_per_m", 6e6)))

    rlc = compute_rlc_params({
        "capacitance_uF": bank["capacitance_uF"],
        "charge_voltage_V": voltage,
        "inductance_uH": bank["inductance_uH"],
        "total_resistance_ohm": bank["loop_resistance_ohm"],
    })

    specs = profile.station_specs()
    stations = {
        name: VirtualStation(name, spec["channel_x_mm"], spec["pitch_mm"],
                             rev=spec["order_rev"])
        for name, spec in specs.items()
    }
    st_in = stations[profile.sensing["station_in"]]
    st_out = stations[profile.sensing["station_out"]]
    controller = FiringController(profile.firing, st_in)
    window = profile.gate_window_us(on_time_us)

    # Start upstream of station A's first channel, with enough run-up that the
    # whole five-channel pass happens before anything else does.
    all_channels = [x for s in specs.values() for x in s["channel_x_mm"]]
    x = min(all_channels) - 30.0
    end_x = max(all_channels) + 20.0

    v = float(v_in_mps) * 1000.0          # mm/s, translational
    omega = v / r_ball                    # rad/s -- released ROLLING, not skidding
    t = 0.0
    triggered = False
    pulse_cut = False
    t_trigger = None
    t_cut = None
    I_at_cut = 0.0
    fire_x = None
    i_peak = 0.0
    prev_x = None
    slip_work_mm_s = 0.0

    # Optional override: fire when the ball reaches this x instead of using the
    # controller. Used by --scan-fire-offset to map the impulse optimum.
    forced_fire_x = fire_offset_mm

    steps = int(max_time_s / dt)
    for _ in range(steps):
        if x > end_x:
            break

        # -- sensing (interpolated within the step, never sampled at it) ----
        if prev_x is not None:
            for station in stations.values():
                for idx, ch_x in enumerate(station.channel_x_mm):
                    if station._got[idx]:
                        continue
                    if channel_crossed(prev_x, x, ch_x):
                        t_cross = interpolate_crossing_us(
                            prev_x, x, ch_x, t * 1e6, dt * 1e6)
                        if t_cross is not None:
                            station.record(idx, t_cross)

        # -- firing --------------------------------------------------------
        if not triggered:
            if forced_fire_x is not None:
                if prev_x is not None and channel_crossed(prev_x, x, forced_fire_x):
                    triggered, t_trigger, fire_x = True, t, x
            else:
                state = controller.update(t * 1e6)
                if state == FiringController.FIRED:
                    triggered, t_trigger, fire_x = True, t, x
                elif state == FiringController.ABORTED:
                    return {
                        "aborted": True,
                        "abort_reason": controller.abort_reason,
                        "cans": cans, "voltage_V": voltage,
                    }

        # -- circuit -------------------------------------------------------
        current = 0.0
        if triggered:
            t_rel = t - t_trigger
            if not pulse_cut:
                if t_rel * 1e6 >= window["gate_us"]:
                    # Cut at the EXACT gate time, not at the step boundary that
                    # happened to pass it. Quantising the cut to dt is what
                    # dominated this model's step-size error: the gate ran long
                    # by up to one step, which at dt=10us meant a 200us pulse
                    # instead of 191us and a ~5% Delta-v error. Taking the cut
                    # time and the current at it exactly makes dt=20us as good
                    # as dt=1us was.
                    t_gate = window["gate_us"] * 1e-6
                    pulse_cut = True
                    t_cut = t_trigger + t_gate
                    I_at_cut = rlc_current(t_gate, rlc)
                else:
                    current = rlc_current(t_rel, rlc)
            if pulse_cut:
                current = freewheel_current(
                    I_at_cut, t - t_cut,
                    bank["freewheel_resistance_ohm"], rlc["inductance_H"],
                    bank["diode_vf_V"])
            i_peak = max(i_peak, current)

        # -- EM force ------------------------------------------------------
        F_mN = 0.0
        if current > 1e-8:
            Br, Bz, dBr_dr, dBr_dz, dBz_dr, dBz_dz = solver.field_with_grad(
                0.0, x, current)
            # Susceptibility capped at saturation; continuous, no branch, and
            # the sign is carried by (B.grad)B.
            chi_used = min(chi_eff, B_sat / max(abs(Bz), 1e-12))
            F_mN = (chi_used * V_marble / MU_0_MM) * (Br * dBz_dr + Bz * dBz_dz)

            # Motional eddy drag. Now that it is expressed via dB/dz it is
            # step-size independent and cheap (the gradient is already to
            # hand), so include it for consistency with the other paths even
            # though it is ~0.1% here.
            F_mN += eddy_braking_force(dBz_dz, v, {
                "conductivity_S_per_m": conductivity,
                "radius_mm": r_ball,
                "volume_mm3": V_marble,
            })

        F_N = F_mN * 1e-3
        a_em = (F_N / mass_kg) * 1000.0        # mm/s^2

        # -- translation + spin, coupled by friction ------------------------
        # Contact friction is what moves energy between translation and spin,
        # and it is an order of magnitude too weak to keep up with the pulse:
        # holding the rolling condition needs (2/7)F at the contact, ~0.16 N
        # mean here, against a mu*m*g budget of ~0.025 N. So the ball slips
        # during the pulse and friction restores rolling over the next few mm.
        # Integrate both states and let that happen rather than assuming
        # either end of it -- the endpoint agrees either way, but only
        # because angular momentum about the contact makes it so.
        r_m = r_ball * 1e-3
        f_max = friction * mass_kg * (GRAVITY_MM_S2 / 1000.0)      # N
        slip = v - omega * r_ball                                   # mm/s
        rolling = abs(slip) <= SLIP_TOL_MM_S

        if rolling:
            # Static friction needed to hold the rolling condition is (2/7)F.
            f_req = SPHERE_INERTIA / (1.0 + SPHERE_INERTIA) * F_N
            rolling = abs(f_req) <= f_max

        if rolling:
            a_v = a_em / (1.0 + SPHERE_INERTIA)
            alpha = a_v / r_ball
        else:
            # Kinetic friction opposes the slip at the contact.
            direction = math.copysign(1.0, slip) if abs(slip) > SLIP_TOL_MM_S \
                else math.copysign(1.0, F_N)
            f_N = f_max * direction
            a_v = a_em - (f_N / mass_kg) * 1000.0
            alpha = f_N / (SPHERE_INERTIA * mass_kg * r_m)
            slip_work_mm_s += abs(slip) * dt

        if not allow_spin:
            # Modelling switch: a ball that cannot rotate keeps the whole
            # impulse. Only used to expose the 5/7 factor by comparison.
            a_v = a_em
            alpha = 0.0

        if rolling_resistance:
            a_v -= math.copysign(rolling_resistance * GRAVITY_MM_S2, v)

        v_new = v + a_v * dt
        omega_new = omega + alpha * dt
        # Friction must not drive the ball through the rolling condition and
        # out the other side within a step; snap to rolling instead.
        if not rolling and slip * (v_new - omega_new * r_ball) < 0:
            v_roll = (v_new + SPHERE_INERTIA * omega_new * r_ball) / (1 + SPHERE_INERTIA)
            v_new, omega_new = v_roll, v_roll / r_ball

        prev_x = x
        x += v_new * dt
        v, omega = v_new, omega_new
        t += dt

    v_in_fit = st_in.velocity_mps()
    v_out_fit = st_out.velocity_mps()
    return {
        "aborted": False,
        "cans": cans,
        "C_uF": bank["capacitance_uF"],
        "voltage_V": voltage,
        "loop_R_mohm": bank["loop_resistance_ohm"] * 1000.0,
        "L_uH": bank["inductance_uH"],
        "zeta": rlc["zeta"],
        "regime": rlc["regime"],
        "on_time_us": window["on_time_us"],
        "gate_us": window["gate_us"],
        "v_in_mps": v_in_fit,
        "v_out_mps": v_out_fit,
        "dv_mps": (v_out_fit - v_in_fit) if (v_in_fit and v_out_fit) else None,
        "v_in_true_mps": float(v_in_mps),
        "v_out_true_mps": v / 1000.0,
        "dv_true_mps": v / 1000.0 - float(v_in_mps),
        "resid_in_us": st_in.residual_us(),
        "resid_out_us": st_out.residual_us(),
        "n_ch_in": st_in.n_captured(),
        "n_ch_out": st_out.n_captured(),
        "sensor_pitch_mm": profile.sensing["pitch_mm"],
        "fire_x_mm": fire_x,
        "i_peak_model_A": i_peak,
        "stored_energy_J": rlc["stored_energy_J"],
        "marble_energy_J": 0.5 * float(profile.marble["mass_kg"])
                           * ((v / 1000.0) ** 2 - float(v_in_mps) ** 2),
    }


# Mirrors omnimarble-vbench/firmware/shotlog.py COLUMNS verbatim and in order.
# Duplicated with a citation rather than imported: vbench already accepts
# hand-sync across the repo boundary, and the schema is about to GROW (its
# NEXT_SESSION.md section 3 adds v_post_10ms, v_post_50ms, v_post_extrap,
# charge_r_ohm and psu_i_limit_ma). Everything downstream therefore matches on
# column NAME intersection, never on position or an exact set.
RIG_COLUMNS = [
    "shot_id", "t_ms", "cans", "C_uF", "v_bank_pre", "v_bank_post",
    "on_time_us", "v_in_mps", "v_out_mps", "dv_mps",
    "n_ch_in", "n_ch_out", "resid_in_us", "resid_out_us", "sensor_pitch_mm",
    "i_peak_A", "coil_L_uH", "coil_R_mohm", "loop_R_mohm", "note",
]

# Sim-only, all prefixed so a merged frame is unambiguous. The *_true and
# *_bias columns are the ones the rig CANNOT produce: they separate measurement
# error from physics error, which is what makes a sim-vs-real gap diagnosable.
SIM_COLUMNS = [
    "sim_source", "sim_rig_profile",
    "sim_v_in_true_mps", "sim_v_out_true_mps", "sim_dv_true_mps",
    "sim_v_in_bias_mps", "sim_v_out_bias_mps",
    "sim_gate_us", "sim_fire_x_mm", "sim_zeta", "sim_regime",
    "sim_i_peak_A_model", "sim_stored_energy_J", "sim_marble_energy_J",
    "sim_efficiency_pct",
]


def to_shot_row(shot, profile, shot_id=0):
    """Map a simulated shot onto the rig's own schema, plus sim-only extras."""
    v_in, v_out = shot["v_in_mps"], shot["v_out_mps"]
    stored = shot["stored_energy_J"]
    return {
        "shot_id": shot_id,
        "t_ms": "",
        "cans": shot["cans"],
        "C_uF": round(shot["C_uF"], 1),
        "v_bank_pre": shot["voltage_V"],
        "v_bank_post": "",
        "on_time_us": round(shot["on_time_us"]),
        "v_in_mps": None if v_in is None else round(v_in, 4),
        "v_out_mps": None if v_out is None else round(v_out, 4),
        "dv_mps": None if shot["dv_mps"] is None else round(shot["dv_mps"], 5),
        "n_ch_in": shot["n_ch_in"],
        "n_ch_out": shot["n_ch_out"],
        "resid_in_us": None if shot["resid_in_us"] is None
                       else round(shot["resid_in_us"], 1),
        "resid_out_us": None if shot["resid_out_us"] is None
                        else round(shot["resid_out_us"], 1),
        "sensor_pitch_mm": shot["sensor_pitch_mm"],
        # DELIBERATELY EMPTY. On the rig this is the operator's scope reading;
        # filling it with the model's own number would make a merged plot
        # compare the model against itself.
        "i_peak_A": "",
        "coil_L_uH": shot["L_uH"],
        "coil_R_mohm": round(
            profile.circuit["measured"]["coil_resistance_ohm"] * 1000, 1),
        "loop_R_mohm": round(shot["loop_R_mohm"], 1),
        "note": "simulated",
        "sim_source": "flat_bore_1d",
        "sim_rig_profile": profile.name,
        "sim_v_in_true_mps": round(shot["v_in_true_mps"], 5),
        "sim_v_out_true_mps": round(shot["v_out_true_mps"], 5),
        "sim_dv_true_mps": round(shot["dv_true_mps"], 5),
        "sim_v_in_bias_mps": None if v_in is None
                             else round(v_in - shot["v_in_true_mps"], 6),
        "sim_v_out_bias_mps": None if v_out is None
                              else round(v_out - shot["v_out_true_mps"], 6),
        "sim_gate_us": round(shot["gate_us"], 1),
        "sim_fire_x_mm": None if shot["fire_x_mm"] is None
                         else round(shot["fire_x_mm"], 3),
        "sim_zeta": round(shot["zeta"], 4),
        "sim_regime": shot["regime"],
        "sim_i_peak_A_model": round(shot["i_peak_model_A"], 1),
        "sim_stored_energy_J": round(stored, 4),
        "sim_marble_energy_J": round(shot["marble_energy_J"], 8),
        "sim_efficiency_pct": round(shot["marble_energy_J"] / stored * 100, 6)
                              if stored else 0.0,
    }


def _fmt(value, spec=".4f"):
    return "--" if value is None else format(value, spec)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rig-profile", default="vbench_v0")
    parser.add_argument("--cans", type=int, default=1)
    parser.add_argument("--sweep-cans", default=None,
                        help="comma-separated can counts, e.g. 1,2,3,4,5")
    parser.add_argument("--voltage", type=float, default=None)
    parser.add_argument("--v-in", type=float, default=None,
                        help="approach speed in m/s (default: the rig's ideal release)")
    parser.add_argument("--on-time-us", type=float, default=None)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_S,
                        help="integration step in seconds (default 1e-5, "
                             "converged to 0.1%%; see DEFAULT_DT_S)")
    parser.add_argument("--rolling-resistance", type=float, default=0.0,
                        help="rolling resistance coefficient; UNMEASURED on the "
                             "rig, so zero by default")
    parser.add_argument("--scan-fire-offset", default=None,
                        help="MIN:MAX:STEP in mm -- map dv against fire position")
    parser.add_argument("--shots-out", type=Path, default=None)
    args = parser.parse_args()

    profile = load_profile(ROOT, args.rig_profile)
    if profile.sensing_mode != "stations":
        raise SystemExit(f"profile {profile.name!r} is not a station rig")

    voltage = args.voltage if args.voltage is not None \
        else profile.circuit["charge_voltage_V"]
    v_max = profile.circuit.get("voltage_max_V")
    if v_max and voltage > v_max:
        raise SystemExit(f"{voltage}V exceeds the {v_max}V board invariant")
    v_in = args.v_in if args.v_in is not None \
        else profile.track["release_v_ideal_mps"]

    params = json.loads((ROOT / "config" / "coil_params.json").read_text())
    params["num_turns"] = profile.coil["num_turns"]
    params["length_mm"] = profile.coil["length_mm"]
    from warp_bfield_solver import WarpBFieldSolver
    solver = WarpBFieldSolver(params, chi_eff=3.0)

    print(f"Profile {profile.name}: {voltage:.1f} V, v_in {v_in:.3f} m/s, "
          f"marble {profile.marble['mass_kg']*1000:.2f} g "
          f"({'ASSUMED' if profile.marble.get('mass_is_assumed') else 'measured'})")
    if args.rolling_resistance == 0.0:
        print("Rolling resistance 0 -- vbench flags it as an UNMEASURED ~10% "
              "systematic over the 204 mm between stations.")

    rows = []
    if args.scan_fire_offset:
        lo, hi, step = (float(v) for v in args.scan_fire_offset.split(":"))
        print(f"\n{'fire_x':>8} {'dv (m/s)':>10} {'dv_true':>10} {'I_pk':>8}")
        print("-" * 40)
        best = None
        n = int(round((hi - lo) / step)) + 1
        for i in range(n):
            fx = lo + i * step
            shot = simulate_shot(profile, solver, args.cans, voltage, v_in,
                                 args.on_time_us, fire_offset_mm=fx,
                                 rolling_resistance=args.rolling_resistance,
                                 dt=args.dt)
            if shot["aborted"]:
                continue
            rows.append({**shot, "fire_offset_mm": fx})
            print(f"{fx:>8.1f} {_fmt(shot['dv_mps']):>10} "
                  f"{shot['dv_true_mps']:>10.4f} {shot['i_peak_model_A']:>8.0f}")
            if best is None or shot["dv_true_mps"] > best["dv_true_mps"]:
                best = rows[-1]
        if best:
            face_in = profile.coil["face_in_x_mm"]
            print(f"\nImpulse-optimal fire position: x = {best['fire_offset_mm']:.1f} mm "
                  f"(dv_true {best['dv_true_mps']:.4f} m/s)")
            print(f"The rig currently fires at the coil entry face, x = {face_in:.2f} mm.")
    else:
        can_list = ([int(c) for c in args.sweep_cans.split(",")]
                    if args.sweep_cans else [args.cans])
        print(f"\n{'cans':>5} {'C(uF)':>7} {'R(mohm)':>8} {'zeta':>6} {'regime':>16} "
              f"{'I_pk':>7} {'v_in':>7} {'v_out':>7} {'dv':>8} {'dv_true':>8}")
        print("-" * 92)
        for cans in can_list:
            shot = simulate_shot(profile, solver, cans, voltage, v_in,
                                 args.on_time_us,
                                 rolling_resistance=args.rolling_resistance,
                                 dt=args.dt)
            if shot["aborted"]:
                print(f"{cans:>5}  ABORTED: {shot['abort_reason']}")
                continue
            rows.append(shot)
            print(f"{cans:>5} {shot['C_uF']:>7.0f} {shot['loop_R_mohm']:>8.1f} "
                  f"{shot['zeta']:>6.3f} {shot['regime']:>16} "
                  f"{shot['i_peak_model_A']:>7.0f} "
                  f"{_fmt(shot['v_in_mps'], '.3f'):>7} "
                  f"{_fmt(shot['v_out_mps'], '.3f'):>7} "
                  f"{_fmt(shot['dv_mps'], '.4f'):>8} "
                  f"{shot['dv_true_mps']:>8.4f}")

        if rows:
            r0 = rows[0]
            eff = (r0["marble_energy_J"] / r0["stored_energy_J"] * 100
                   if r0["stored_energy_J"] else 0)
            print(f"\nEfficiency at {r0['cans']} can(s): {eff:.4f}% "
                  f"({r0['marble_energy_J']*1000:.3f} mJ into the marble from "
                  f"{r0['stored_energy_J']:.2f} J stored)")
            print("Bench measured ~0.05% -- 'a real number for the twin to "
                  "reproduce, not a fault'.")

    if args.shots_out and rows:
        args.shots_out.parent.mkdir(parents=True, exist_ok=True)
        out_rows = [to_shot_row(shot, profile, i) for i, shot in enumerate(rows)]
        with open(args.shots_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RIG_COLUMNS + SIM_COLUMNS)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\nwrote {len(out_rows)} rows to {args.shots_out} "
              f"(rig schema + sim_* columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
