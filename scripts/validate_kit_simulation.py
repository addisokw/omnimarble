"""Headless simulation mirroring the Kit extension's physics chain.

Uses RLC capacitor discharge, multi-gate IR sensors, saturation, eddy
currents, and thermal model. Loads the starter-slope track and matches
the Kit extension's marble start position from config.

Field modes (--field):
  pinn (default): PINN inference + autograd gradients + full (B.grad)B
      force — the same field/force path the Kit extension uses.
  analytical: closed-form Biot-Savart with finite-difference dBz/dz —
      a PINN-independent cross-check, NOT what Kit runs.

Collision handling here is a simple sphere-mesh bounce, not PhysX — exit
velocities are comparable to Kit but not identical.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "coil_params.json"
PLOTS_DIR = ROOT / "results" / "plots"

# Try starter-slope first, fall back to original track
TRACK_STL = ROOT / "data" / "track-starter-slope.stl"
if not TRACK_STL.exists():
    TRACK_STL = ROOT / "data" / "track.stl"

sys.path.insert(0, str(ROOT / "scripts"))
# The sensing/profile modules live with the Kit extension so that Kit and this
# headless mirror share one implementation rather than two that drift.
sys.path.insert(0, str(ROOT / "source" / "extensions" / "omni.marble.coaster"
                       / "omni" / "marble" / "coaster"))
from coil_sensing import (
    FiringController,
    VirtualStation,
    crossed as channel_crossed,
    interpolate_crossing_us,
)
from rig_profile import load_profile
from analytical_bfield import MU_0_MM, solenoid_field
from rlc_circuit import (
    compute_rlc_params,
    coupled_rlc_step_substep,
    rlc_current,
    compute_winding_geometry,
    compute_dc_resistance,
    compute_ac_resistance,
    compute_multilayer_inductance,
    eddy_braking_force,
    saturated_force,
    wire_temperature_rise,
    resistance_at_temperature,
)

MARBLE_RADIUS = 5.0  # mm
GRAVITY = np.array([0.0, 0.0, -9810.0])  # mm/s^2


def load_track_mesh():
    """Load track STL for collision detection."""
    raw = trimesh.load(str(TRACK_STL))
    if isinstance(raw, trimesh.Scene):
        meshes = list(raw.geometry.values())
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    else:
        mesh = raw
    mesh.fix_normals()
    return mesh


def check_collision(pos, vel, radius, mesh):
    """Simple sphere-mesh collision with bounce."""
    closest, distance, face_id = trimesh.proximity.closest_point(mesh, [pos])
    closest = closest[0]
    dist = distance[0]

    if dist < radius:
        normal = pos - closest
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-8:
            normal = normal / norm_len
        else:
            normal = np.array([0, 0, 1.0])

        # Push out
        penetration = radius - dist
        pos = pos + normal * penetration

        # Reflect velocity
        v_n = np.dot(vel, normal)
        if v_n < 0:
            restitution = 0.3
            friction = 0.01  # low rolling friction for steel on steel
            vel_normal = v_n * normal
            vel_tangent = vel - vel_normal
            vel = vel - (1 + restitution) * vel_normal
            vel -= friction * vel_tangent

        return True, pos, vel

    return False, pos, vel


def build_rlc_from_config(params: dict, measured: dict = None) -> dict:
    """Derive RLC parameters from config, letting measured values win.

    `measured` may carry `inductance_uH` and/or `loop_resistance_ohm` read off
    the real coil. They override the geometry-derived estimates, because the
    bench is the arbiter: the rig's coil measures 17.9 uH where the geometry
    model predicts 17.3, and its loop resistance (0.164 ohm, the whole
    discharge path) is not the same quantity as the coil's own DC resistance
    plus a guessed ESR.
    """
    geom = compute_winding_geometry(params)
    R_dc = compute_dc_resistance(geom["wire_length_mm"], geom["wire_cross_section_mm2"],
                                  params.get("ambient_temperature_C", 20.0))
    L_uH = compute_multilayer_inductance(
        params["num_turns"], geom["mean_radius_mm"],
        params["length_mm"], geom["winding_depth_mm"],
    )
    R_esr = params.get("esr_ohm", 0.01)
    R_wiring = params.get("wiring_resistance_ohm", 0.02)
    R_total_dc = R_dc + R_esr + R_wiring

    L_H = L_uH * 1e-6
    C = params.get("capacitance_uF", 470.0) * 1e-6
    omega_0 = 1.0 / math.sqrt(L_H * C)
    alpha = R_total_dc / (2 * L_H)
    zeta = alpha / omega_0
    freq_Hz = math.sqrt(abs(omega_0 ** 2 - alpha ** 2)) / (2 * math.pi) if zeta < 1 else omega_0 / (2 * math.pi)

    ac_info = compute_ac_resistance(R_dc, params["wire_diameter_mm"],
                                     geom["num_layers"], freq_Hz)
    R_total_ac = ac_info["R_ac_ohm"] + R_esr + R_wiring

    measured = measured or {}
    L_uH_used = measured.get("inductance_uH", L_uH)
    R_total_used = measured.get("loop_resistance_ohm", R_total_ac)

    rlc = compute_rlc_params({
        "capacitance_uF": params.get("capacitance_uF", 470.0),
        "charge_voltage_V": params.get("charge_voltage_V", 50.0),
        "inductance_uH": L_uH_used,
        "total_resistance_ohm": R_total_used,
    })
    rlc["_wire_mass_g"] = geom["wire_mass_g"]
    rlc["_R_dc_base"] = R_dc
    rlc["_R_esr"] = R_esr
    rlc["_R_wiring"] = R_wiring
    rlc["_L_source"] = "measured" if "inductance_uH" in measured else "derived"
    rlc["_R_source"] = "measured" if "loop_resistance_ohm" in measured else "derived"
    rlc["_L_derived_uH"] = L_uH
    rlc["_R_derived_ohm"] = R_total_ac
    return rlc


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--field", choices=["pinn", "analytical"], default="pinn",
                        help="B-field source: pinn matches the Kit extension path "
                             "(default); analytical is a PINN-independent cross-check")
    parser.add_argument("--voltage", type=float, default=None,
                        help="Override charge voltage (V), e.g. 300 for a launch-demo run")
    parser.add_argument("--rig-profile", default=None,
                        help="named profile from config/rig_profile.json "
                             "(default: that file's default_profile)")
    parser.add_argument("--cans", type=int, default=None,
                        help="populated capacitor cans, 1..5 (station profiles only). "
                             "Bank ESR parallels as 1/n, so this moves R as well as C")
    parser.add_argument("--on-time-us", type=float, default=None,
                        help="gate on-time request in us (station profiles only)")
    args = parser.parse_args()

    profile = load_profile(ROOT, args.rig_profile)

    params = json.loads(CONFIG_PATH.read_text())
    if args.voltage is not None:
        params["charge_voltage_V"] = args.voltage

    measured = None
    bank = None
    if profile.sensing_mode == "stations":
        # The rig's circuit, not config/coil_params.json. cans moves C and R
        # together because the bank's ESR parallels down as 1/n.
        bank = profile.bank(args.cans)
        params["capacitance_uF"] = bank["capacitance_uF"]
        if args.voltage is None:
            params["charge_voltage_V"] = profile.circuit["charge_voltage_V"]
        v_max = profile.circuit.get("voltage_max_V")
        if v_max and params["charge_voltage_V"] > v_max:
            raise SystemExit(
                f"charge voltage {params['charge_voltage_V']}V exceeds the "
                f"{v_max}V board invariant for profile {profile.name!r}")
        measured = {
            "inductance_uH": bank["inductance_uH"],
            "loop_resistance_ohm": bank["loop_resistance_ohm"],
        }

    rlc = build_rlc_from_config(params, measured)

    solver = None
    if args.field == "pinn":
        # Requires torch + physicsnemo + the trained checkpoint
        from warp_bfield_solver import WarpBFieldSolver
        solver = WarpBFieldSolver(params, chi_eff=3.0)

    chi_eff = 3.0
    marble_radius = float(profile.marble.get("radius_mm", MARBLE_RADIUS))
    V_marble = (4 / 3) * math.pi * marble_radius ** 3
    if profile.marble.get("mass_kg") is not None:
        # The rig's ball mass is stated rather than derived. It is ASSUMED
        # steel upstream, not weighed -- dv scales directly with it.
        marble_mass_kg = float(profile.marble["mass_kg"])
    else:
        marble_mass_kg = V_marble * 7.8e-3 * 1e-3
    marble_mass_g = marble_mass_kg * 1000

    coil_pos = np.array(params["position_mm"], dtype=float)
    coil_axis = np.array(params.get("axis", [0, 1, 0]), dtype=float)
    coil_axis = coil_axis / np.linalg.norm(coil_axis)

    B_sat = params.get("marble_saturation_T", 1.8)
    conductivity = params.get("marble_conductivity_S_per_m", 6e6)
    switch_type = params.get("switch_type", "MOSFET")

    # IR gate positions from config (match Kit extension)
    gates = params.get("gate_positions", {
        "vel_in_1": -60.0, "vel_in_2": -40.0,
        "entry": -20.0, "cutoff": 5.0,
        "vel_out_1": 60.0, "vel_out_2": 120.0,
    })

    # Station sensing (rig profiles). The gate path above is untouched so the
    # legacy profile keeps reproducing the committed 300V fixture exactly.
    stations = {}
    firing_ctl = None
    gate_window = None
    if profile.sensing_mode == "stations":
        for name, spec in profile.station_specs().items():
            stations[name] = VirtualStation(
                name, spec["channel_x_mm"], spec["pitch_mm"], rev=spec["order_rev"])
        firing_ctl = FiringController(
            profile.firing, stations[profile.sensing["station_in"]])
        gate_window = profile.gate_window_us(args.on_time_us)

    print(f"=== Headless Kit Simulation Validation (field={args.field}) ===")
    print(f"Profile: {profile.name} ({profile.sensing_mode} / {profile.firing_mode})")
    print(f"Track: {TRACK_STL.name}")
    print(f"Marble: {marble_mass_g:.2f}g, r={marble_radius}mm, "
          f"chi_eff={chi_eff}, B_sat={B_sat}T")
    print(f"RLC: {rlc['regime']}, zeta={rlc['zeta']:.4f}, I_peak={rlc['peak_current_A']:.0f}A "
          f"(L {rlc['_L_source']}, R {rlc['_R_source']})")
    print(f"Stored energy: {rlc['stored_energy_J']:.2f} J")
    print(f"Coil center: {coil_pos}")
    if bank is not None:
        print(f"Bank: {bank['cans']} can(s), {bank['capacitance_uF']:.0f}uF, "
              f"loop R {bank['loop_resistance_ohm']*1000:.0f}mOhm, "
              f"L {bank['inductance_uH']}uH")
        print(f"Gate: {gate_window['on_time_us']:.0f}us requested -> "
              f"{gate_window['gate_us']:.0f}us physical")
        for name, st in stations.items():
            print(f"Station {name}: channels {st.channel_x_mm}")
    else:
        print(f"IR Gates: {gates}")

    # Load track
    mesh = load_track_mesh()
    print(f"Track: {len(mesh.faces)} faces, bounds Y=[{mesh.bounds[0][1]:.0f}, {mesh.bounds[1][1]:.0f}]")

    # Marble start position — read from config (set by setup_launch_scene.py)
    # The marble_actor.usda has the position, but we read it from the track analysis
    verts = mesh.vertices
    # Find low-Y end of track for marble start (matches setup_launch_scene.py)
    start_y = -20.0  # matches setup_launch_scene.py
    start_mask = np.abs(verts[:, 1] - start_y) < 10
    if start_mask.sum() > 0:
        start_x = verts[start_mask, 0].mean()
        start_z_track = verts[start_mask, 2].max()
    else:
        start_x = 0.0
        start_z_track = 5.0
    pos = np.array([start_x, start_y, start_z_track + marble_radius + 2])
    vel = np.zeros(3)

    print(f"Start pos: {pos}")

    # Simulation
    dt = 0.002  # 500Hz like Kit
    total_time = 5.0
    n_steps = int(total_time / dt)

    # Multi-gate state
    gate_triggered = {name: False for name in gates}
    gate_times = {name: None for name in gates}
    triggered = False
    trigger_time = 0.0
    pulse_cut = False
    pulse_cut_time = 0.0
    I_at_cut = 0.0
    prev_z_along = None
    approach_velocity = None
    exit_velocity = None

    # Circuit state
    circuit_state = {
        "I": 0.0,
        "Q_cap": rlc["capacitance_F"] * rlc["charge_voltage_V"],
    }
    wire_temp = params.get("ambient_temperature_C", 20.0)
    prev_B = 0.0

    marble_params = {
        "chi_eff": chi_eff,
        "volume_mm3": V_marble,
        "saturation_T": B_sat,
        "conductivity_S_per_m": conductivity,
        "radius_mm": marble_radius,
    }

    times = []
    positions = []
    velocities = []
    forces_log = []
    currents_log = []
    cap_voltages_log = []
    wire_temps_log = []

    for step in range(n_steps):
        t = step * dt
        times.append(t)
        positions.append(pos.copy())
        velocities.append(vel.copy())

        # Coil-local coords
        relative = pos - coil_pos
        z_along = float(np.dot(relative, coil_axis))
        radial_vec = relative - z_along * coil_axis
        r = float(np.linalg.norm(radial_vec))

        # --- IR sensing ---
        if profile.sensing_mode == "stations":
            # Interpolate the crossing WITHIN the step. Sampling t directly
            # would quantise every timestamp to the step (2mm at 1m/s against
            # a 22.14mm pitch, ~9% per channel) and the estimator would be
            # measuring the solver rather than the marble.
            if prev_z_along is not None:
                for station in stations.values():
                    for idx, ch_x in enumerate(station.channel_x_mm):
                        if station._got[idx]:
                            continue
                        if channel_crossed(prev_z_along, z_along, ch_x):
                            t_cross_us = interpolate_crossing_us(
                                prev_z_along, z_along, ch_x, t * 1e6, dt * 1e6)
                            if t_cross_us is not None:
                                station.record(idx, t_cross_us)
        elif prev_z_along is not None:
            for gate_name, gate_pos in gates.items():
                if gate_triggered[gate_name]:
                    continue
                crossed = (prev_z_along < gate_pos and z_along >= gate_pos) or \
                          (prev_z_along > gate_pos and z_along <= gate_pos)
                if crossed:
                    gate_triggered[gate_name] = True
                    gate_times[gate_name] = t
                    print(f"  t={t:.4f}s: [IR] Gate '{gate_name}' at z_along={z_along:.1f}mm")

        # Approach velocity from vel_in pair
        t1 = gate_times.get("vel_in_1")
        t2 = gate_times.get("vel_in_2")
        if t1 is not None and t2 is not None and approach_velocity is None:
            dt_gates = t2 - t1
            if abs(dt_gates) > 1e-6:
                dist = gates["vel_in_2"] - gates["vel_in_1"]
                approach_velocity = dist / dt_gates
                print(f"  t={t:.4f}s: [IR] Approach velocity: {approach_velocity:.1f} mm/s")

        # --- Firing ---
        if firing_ctl is not None:
            if not triggered and firing_ctl.state not in (
                    FiringController.FIRED, FiringController.ABORTED):
                state = firing_ctl.update(t * 1e6)
                if state == FiringController.FIRED:
                    triggered = True
                    trigger_time = t
                    approach_velocity = firing_ctl.v_in_mps * 1000.0
                    print(f"  t={t:.4f}s: [EM] COIL FIRED at z_along={z_along:.1f}mm, "
                          f"v_in={approach_velocity:.0f}mm/s "
                          f"(slack {firing_ctl.slack_us:.0f}us)")
                    if firing_ctl.fit_suspect:
                        print(f"  WARNING: station fit residual "
                              f"{firing_ctl.residual_us:.0f}us -- the straight-line "
                              f"fit is not describing the motion, so v_in and the "
                              f"fire timing are suspect")
                elif state == FiringController.ABORTED:
                    print(f"  t={t:.4f}s: [EM] SHOT ABORTED: {firing_ctl.abort_reason}")

            # Fixed on-time cutoff: the switch is held for the PHYSICAL window,
            # which is the logged request minus the measured ~9us overhead.
            if triggered and not pulse_cut and switch_type == "MOSFET":
                if (t - trigger_time) * 1e6 >= gate_window["gate_us"]:
                    pulse_cut = True
                    pulse_cut_time = t
                    I_at_cut = abs(circuit_state["I"])
                    print(f"  t={t:.4f}s: [EM] PULSE CUT after "
                          f"{gate_window['gate_us']:.0f}us at "
                          f"z_along={z_along:.1f}mm, I={I_at_cut:.0f}A")

        # Entry gate -> fire coil
        if firing_ctl is None and not triggered and gate_triggered.get("entry"):
            triggered = True
            trigger_time = t
            v_str = f", v_approach={approach_velocity:.0f}mm/s" if approach_velocity else ""
            print(f"  t={t:.4f}s: [EM] COIL FIRED at z_along={z_along:.1f}mm{v_str}")

        # Cutoff gate -> kill pulse
        if firing_ctl is None and triggered and not pulse_cut and switch_type == "MOSFET":
            if gate_triggered.get("cutoff"):
                pulse_cut = True
                pulse_cut_time = t
                # Seed the freewheel decay from the LIVE coupled state -- see
                # the decay branch below.
                I_at_cut = abs(circuit_state["I"])
                print(f"  t={t:.4f}s: [EM] PULSE CUT at z_along={z_along:.1f}mm")

        # Exit velocity: station B's own fit, so v_out inherits exactly the
        # same estimator (and the same biases) as v_in.
        #
        # Only after a shot actually fired. An aborted shot has no trustworthy
        # v_in, so quoting a dv against it would be meaningless -- the rig
        # discards such shots rather than logging them.
        if (firing_ctl is not None and triggered and approach_velocity
                and exit_velocity is None):
            station_out = stations[profile.sensing["station_out"]]
            if station_out.complete():
                v_out_mps = station_out.velocity_mps()
                if v_out_mps:
                    exit_velocity = v_out_mps * 1000.0
                    boost = (exit_velocity / approach_velocity
                             if approach_velocity else 0)
                    dv = exit_velocity - (approach_velocity or 0.0)
                    print(f"  t={t:.4f}s: [IR] EXIT VELOCITY: {exit_velocity:.1f} mm/s "
                          f"= {v_out_mps:.3f} m/s (dv={dv/1000:.3f} m/s, "
                          f"boost {boost:.2f}x)")

        # Exit velocity from vel_out pair
        t_o1 = gate_times.get("vel_out_1")
        t_o2 = gate_times.get("vel_out_2")
        if firing_ctl is None and t_o1 is not None and t_o2 is not None and exit_velocity is None:
            dt_out = t_o2 - t_o1
            if abs(dt_out) > 1e-6:
                dist_out = gates["vel_out_2"] - gates["vel_out_1"]
                exit_velocity = dist_out / dt_out
                boost = exit_velocity / approach_velocity if approach_velocity and approach_velocity > 0 else 0
                print(f"  t={t:.4f}s: [IR] EXIT VELOCITY: {exit_velocity:.1f} mm/s "
                      f"= {exit_velocity/1000:.2f} m/s (boost: {boost:.1f}x)")

        prev_z_along = z_along

        # --- EM force ---
        em_accel = np.zeros(3)
        current = 0.0
        F_z_mN = 0.0
        V_cap = circuit_state["Q_cap"] / rlc["capacitance_F"]

        if triggered:
            if not pulse_cut:
                vel_axial = float(np.dot(vel, coil_axis))
                circuit_state = coupled_rlc_step_substep(
                    circuit_state, dt, params, rlc, z_along, vel_axial,
                )
                current = circuit_state.get("_I_rms", abs(circuit_state["I"]))
            else:
                # Switch open: freewheel through the diode, decaying L/R from
                # the current the COUPLED solver actually reached at the cut.
                # rlc_current_with_cutoff() re-derives that seed from the
                # uncoupled closed form, which throws away the back-EMF history
                # and makes the current step at the seam. Harmless while the
                # cutoff gate sits far past the end of the pulse; first-order
                # once the cut lands at the current peak, as the rig's does.
                decay = math.exp(
                    -(rlc["total_resistance_ohm"] / rlc["inductance_H"])
                    * (t - pulse_cut_time))
                current = I_at_cut * decay
                circuit_state["I"] = current

            V_cap = circuit_state["Q_cap"] / rlc["capacitance_F"]

            if abs(current) > 1e-8:
                if solver is not None:
                    # PINN path: same field + full (B.grad)B force as the
                    # Kit extension's PINNForceComputer
                    Br, Bz, dBr_dr, dBr_dz, dBz_dr, dBz_dz = solver.field_with_grad(
                        r, z_along, current,
                    )
                    # Cap the susceptibility so the magnetisation cannot
                    # exceed saturation; continuous at chi*|B| = B_sat, and the
                    # sign is already carried by (B.grad)B.
                    chi_used = min(chi_eff, B_sat / max(abs(Bz), 1e-12))
                    prefactor = chi_used * V_marble / MU_0_MM
                    F_z_mN = prefactor * (Br * dBz_dr + Bz * dBz_dz)
                else:
                    # Analytical cross-check: axial force from Bz * dBz/dz
                    # (finite difference), with saturation
                    params_now = params.copy()
                    params_now["current_A"] = current
                    _, Bz = solenoid_field(r, z_along, params_now)
                    _, Bz_p = solenoid_field(r, z_along + 0.1, params_now)
                    _, Bz_m = solenoid_field(r, z_along - 0.1, params_now)
                    dBz_dz = (Bz_p - Bz_m) / 0.2

                    F_z_mN = saturated_force(Bz, dBz_dz, marble_params)

                vel_axial = float(np.dot(vel, coil_axis))
                # From dB/dz, so the term is step-size independent. It used to
                # take a backward-differenced dB/dt and so scaled as 1/dt^2.
                F_eddy = eddy_braking_force(dBz_dz, vel_axial, marble_params)
                F_z_mN += F_eddy

                force_N = F_z_mN * 1e-3
                accel_ms2 = force_N / marble_mass_kg
                dv_mm_s2 = accel_ms2 * 1000.0
                em_accel = dv_mm_s2 * coil_axis

                R_now = resistance_at_temperature(rlc["_R_dc_base"], wire_temp)
                R_total = R_now + rlc["_R_esr"] + rlc["_R_wiring"]
                dT = wire_temperature_rise(current, R_total, dt, rlc["_wire_mass_g"])
                wire_temp += dT

        currents_log.append(current)
        forces_log.append(F_z_mN)
        cap_voltages_log.append(V_cap)
        wire_temps_log.append(wire_temp)

        # Integration
        total_accel = GRAVITY + em_accel
        vel = vel + total_accel * dt
        pos = pos + vel * dt

        # Collision
        _, pos, vel = check_collision(pos, vel, marble_radius, mesh)

        # Floor check
        if pos[2] < mesh.bounds[0][2] - 50:
            print(f"  t={t:.3f}s: Fell below floor")
            break

        # Periodic log
        if step % 250 == 0:
            speed = np.linalg.norm(vel)
            print(f"  t={t:.3f}s: Y={pos[1]:.1f} Z={pos[2]:.1f} "
                  f"v={speed:.0f}mm/s vY={vel[1]:.0f} "
                  f"I={current:.0f}A F={F_z_mN:.0f}mN")

    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)
    forces_log = np.array(forces_log)
    currents_log = np.array(currents_log)
    cap_voltages_log = np.array(cap_voltages_log)
    wire_temps_log = np.array(wire_temps_log)

    # Summary
    speed = np.linalg.norm(velocities, axis=1)
    max_speed = speed.max()
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Max speed: {max_speed:.1f} mm/s = {max_speed/1000:.3f} m/s")
    print(f"  Peak current (RMS): {currents_log.max():.1f} A")
    print(f"  Approach velocity: {approach_velocity:.1f} mm/s" if approach_velocity else "  Approach velocity: not measured")
    print(f"  Exit velocity: {exit_velocity:.1f} mm/s = {exit_velocity/1000:.2f} m/s" if exit_velocity else "  Exit velocity: not measured")
    if approach_velocity and exit_velocity:
        print(f"  Speed boost: {exit_velocity/approach_velocity:.1f}x")
    print(f"  Final wire temp: {wire_temp:.1f} C")
    print(f"  Final cap voltage: {cap_voltages_log[-1]:.1f} V (started at {rlc['charge_voltage_V']:.0f} V)")

    # Plot
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    # Y position
    axes[0, 0].plot(times, positions[:, 1], 'b-')
    for gn, gp in gates.items():
        gy = coil_pos[1] + gp
        axes[0, 0].axhline(gy, color='gray', linestyle=':', alpha=0.5)
        axes[0, 0].annotate(gn, (times[-1]*0.95, gy), fontsize=7, alpha=0.7)
    axes[0, 0].axhline(coil_pos[1], color='red', linestyle='--', alpha=0.5, label='coil center')
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Y position (mm)")
    axes[0, 0].set_title("Y (along track) vs time")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(times, positions[:, 2], 'r-')
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Z position (mm)")
    axes[0, 1].set_title("Z (height) vs time")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(times, speed, 'g-')
    axes[0, 2].set_xlabel("Time (s)")
    axes[0, 2].set_ylabel("Speed (mm/s)")
    axes[0, 2].set_title("Speed vs time")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(times, currents_log, 'm-')
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Current (A)")
    axes[1, 0].set_title(f"Coil current ({rlc['regime']})")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(times, forces_log, 'r-')
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Force (mN)")
    axes[1, 1].set_title("EM force (saturation + eddy)")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(times, cap_voltages_log, 'b-')
    axes[1, 2].set_xlabel("Time (s)")
    axes[1, 2].set_ylabel("Voltage (V)")
    axes[1, 2].set_title("Capacitor voltage")
    axes[1, 2].grid(True, alpha=0.3)

    axes[2, 0].plot(times, wire_temps_log, 'orange')
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Temperature (C)")
    axes[2, 0].set_title("Wire temperature")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(positions[:, 1], positions[:, 2], 'b-', linewidth=0.5)
    axes[2, 1].plot(positions[0, 1], positions[0, 2], 'go', markersize=8, label='start')
    axes[2, 1].set_xlabel("Y (mm)")
    axes[2, 1].set_ylabel("Z (mm)")
    axes[2, 1].set_title("YZ trajectory (side view)")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    E_stored = rlc["stored_energy_J"]
    E_dissipated = np.sum(currents_log ** 2 * rlc["total_resistance_ohm"] * dt)
    KE_final = 0.5 * marble_mass_kg * (max_speed * 1e-3) ** 2
    axes[2, 2].bar(["Stored", "Dissipated\n(resistive)", "KE (marble)"],
                   [E_stored, E_dissipated, KE_final], color=["blue", "red", "green"])
    axes[2, 2].set_ylabel("Energy (J)")
    axes[2, 2].set_title("Energy breakdown")

    plt.suptitle(f"Kit Simulation Validation — {TRACK_STL.name} "
                 f"(field={args.field}, chi={chi_eff}, {marble_mass_g:.1f}g, "
                 f"{rlc['charge_voltage_V']:.0f}V)", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kit_validation.png", dpi=150)
    plt.close()
    print(f"\nSaved: {PLOTS_DIR / 'kit_validation.png'}")


if __name__ == "__main__":
    main()
