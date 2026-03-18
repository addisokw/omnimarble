"""Headless simulation that replicates the Kit extension's exact physics.

Phase 2: Uses RLC capacitor discharge, multi-gate IR sensors, saturation,
eddy currents, and thermal model. Loads the starter-slope track and matches
the Kit extension's marble start position from config.
"""

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
from analytical_bfield import MU_0_MM, solenoid_field
from rlc_circuit import (
    compute_rlc_params,
    coupled_rlc_step_substep,
    rlc_current,
    rlc_current_with_cutoff,
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


def build_rlc_from_config(params: dict) -> dict:
    """Derive RLC parameters from config."""
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
    C = params.get("capacitance_uF", 1000.0) * 1e-6
    omega_0 = 1.0 / math.sqrt(L_H * C)
    alpha = R_total_dc / (2 * L_H)
    zeta = alpha / omega_0
    freq_Hz = math.sqrt(abs(omega_0 ** 2 - alpha ** 2)) / (2 * math.pi) if zeta < 1 else omega_0 / (2 * math.pi)

    ac_info = compute_ac_resistance(R_dc, params["wire_diameter_mm"],
                                     geom["num_layers"], freq_Hz)
    R_total_ac = ac_info["R_ac_ohm"] + R_esr + R_wiring

    rlc = compute_rlc_params({
        "capacitance_uF": params.get("capacitance_uF", 1000.0),
        "charge_voltage_V": params.get("charge_voltage_V", 400.0),
        "inductance_uH": L_uH,
        "total_resistance_ohm": R_total_ac,
    })
    rlc["_wire_mass_g"] = geom["wire_mass_g"]
    rlc["_R_dc_base"] = R_dc
    rlc["_R_esr"] = R_esr
    rlc["_R_wiring"] = R_wiring
    return rlc


def main():
    params = json.loads(CONFIG_PATH.read_text())
    rlc = build_rlc_from_config(params)

    chi_eff = 3.0
    marble_radius = MARBLE_RADIUS
    V_marble = (4 / 3) * math.pi * marble_radius ** 3
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
        "vel_out_1": 20.0, "vel_out_2": 40.0,
    })

    print(f"=== Headless Kit Simulation Validation (RLC Phase 2) ===")
    print(f"Track: {TRACK_STL.name}")
    print(f"Marble: {marble_mass_g:.2f}g, chi_eff={chi_eff}, B_sat={B_sat}T")
    print(f"RLC: {rlc['regime']}, zeta={rlc['zeta']:.4f}, I_peak={rlc['peak_current_A']:.0f}A")
    print(f"Stored energy: {rlc['stored_energy_J']:.2f} J")
    print(f"Coil center: {coil_pos}")
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

        # --- Multi-gate IR sensor detection ---
        if prev_z_along is not None:
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

        # Entry gate -> fire coil
        if not triggered and gate_triggered.get("entry"):
            triggered = True
            trigger_time = t
            v_str = f", v_approach={approach_velocity:.0f}mm/s" if approach_velocity else ""
            print(f"  t={t:.4f}s: [EM] COIL FIRED at z_along={z_along:.1f}mm{v_str}")

        # Cutoff gate -> kill pulse
        if triggered and not pulse_cut and switch_type == "MOSFET":
            if gate_triggered.get("cutoff"):
                pulse_cut = True
                pulse_cut_time = t
                print(f"  t={t:.4f}s: [EM] PULSE CUT at z_along={z_along:.1f}mm")

        # Exit velocity from vel_out pair
        t_o1 = gate_times.get("vel_out_1")
        t_o2 = gate_times.get("vel_out_2")
        if t_o1 is not None and t_o2 is not None and exit_velocity is None:
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
                t_rel = t - trigger_time
                t_cut_rel = pulse_cut_time - trigger_time
                current = rlc_current_with_cutoff(t_rel, t_cut_rel, rlc)
                circuit_state["I"] = current

            V_cap = circuit_state["Q_cap"] / rlc["capacitance_F"]

            if abs(current) > 1e-8:
                params_now = params.copy()
                params_now["current_A"] = current
                _, Bz = solenoid_field(r, z_along, params_now)
                _, Bz_p = solenoid_field(r, z_along + 0.1, params_now)
                _, Bz_m = solenoid_field(r, z_along - 0.1, params_now)
                dBz_dz = (Bz_p - Bz_m) / 0.2

                F_z_mN = saturated_force(Bz, dBz_dz, marble_params)

                B_now = abs(Bz)
                dBdt = (B_now - prev_B) / dt if dt > 0 else 0
                prev_B = B_now
                vel_axial = float(np.dot(vel, coil_axis))
                F_eddy = eddy_braking_force(dBdt, vel_axial, marble_params)
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

    plt.suptitle(f"Kit Simulation Validation — {TRACK_STL.name} (chi={chi_eff}, {marble_mass_g:.1f}g)", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kit_validation.png", dpi=150)
    plt.close()
    print(f"\nSaved: {PLOTS_DIR / 'kit_validation.png'}")


if __name__ == "__main__":
    main()
