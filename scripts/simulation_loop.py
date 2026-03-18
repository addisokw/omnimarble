"""Full simulation orchestration loop.

Phase 2: RLC capacitor discharge + coupled electromechanical ODE +
Warp B-field solver (analytical fallback) + saturation/eddy/thermal effects.

Pipeline: USD scene load -> per-frame RLC step -> B-field solve -> force -> integration -> USD update.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "coil_params.json"
SCENE_PATH = ROOT / "usd" / "marble_coaster_scene.usda"
RESULTS_DIR = ROOT / "results"
TRAJ_DIR = RESULTS_DIR / "trajectories"
PLOTS_DIR = RESULTS_DIR / "plots"

sys.path.insert(0, str(ROOT / "scripts"))

from pxr import Gf, Sdf, Usd, UsdGeom

from analytical_bfield import ferromagnetic_force, solenoid_field
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
    MU_0_MM,
)
from run_physics_test import (
    DT,
    GRAVITY,
    MARBLE_MASS,
    MARBLE_RADIUS,
    RESTITUTION,
    check_collision,
    load_scene_params,
    load_track_mesh,
    save_trajectory,
)
from warp_bfield_solver import WarpBFieldSolver

# Try to import GPU components
try:
    import torch
    from train_pinn import BFieldPINN
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import warp as wp
    HAS_WARP = True
except ImportError:
    HAS_WARP = False


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

    rlc_input = {
        "capacitance_uF": params.get("capacitance_uF", 1000.0),
        "charge_voltage_V": params.get("charge_voltage_V", 400.0),
        "inductance_uH": L_uH,
        "total_resistance_ohm": R_total_ac,
    }
    rlc = compute_rlc_params(rlc_input)

    # Also store wire info for thermal model
    rlc["_wire_mass_g"] = geom["wire_mass_g"]
    rlc["_R_dc_base"] = R_dc
    rlc["_R_esr"] = R_esr
    rlc["_R_wiring"] = R_wiring

    return rlc


class SimulationConfig:
    """Configuration for the full simulation."""
    dt: float = 0.0005
    total_time: float = 5.0
    use_pinn: bool = False
    use_warp: bool = False
    use_coupled_ode: bool = True  # Use coupled electromechanical ODE
    pulse_trigger_offset: float = 15.0  # mm above coil to trigger
    output_usd: bool = True
    fps: int = 60


def run_simulation(config: SimulationConfig):
    """Run the full simulation loop with RLC discharge and full physics."""
    coil_params = json.loads(CONFIG_PATH.read_text())
    rlc = build_rlc_from_config(coil_params)

    print(f"  RLC: {rlc['regime']}, zeta={rlc['zeta']:.4f}")
    print(f"  Peak current: {rlc['peak_current_A']:.1f}A at t={rlc['time_to_peak_s']*1e6:.1f}us")
    print(f"  Stored energy: {rlc['stored_energy_J']:.2f} J")

    # B-field solver
    solver = WarpBFieldSolver(coil_params, chi_eff=3.0)

    print("Loading track mesh...")
    mesh = load_track_mesh()
    print(f"  Track: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    initial_pos, _ = load_scene_params()
    print(f"  Marble start: {initial_pos}")

    # Coil geometry
    coil_pos = np.array(coil_params["position_mm"])
    coil_axis = np.array(coil_params.get("axis", [0, 1, 0]), dtype=float)
    coil_axis = coil_axis / np.linalg.norm(coil_axis)
    switch_type = coil_params.get("switch_type", "MOSFET")

    # Simulation state
    n_steps = int(config.total_time / config.dt)
    pos = initial_pos.copy()
    vel = np.zeros(3)

    # RLC circuit state (for coupled ODE)
    circuit_state = {
        "I": 0.0,
        "Q_cap": rlc["capacitance_F"] * rlc["charge_voltage_V"],
    }
    wire_temp = coil_params.get("ambient_temperature_C", 20.0)
    prev_B = 0.0

    # Recording
    record_interval = max(1, int(1.0 / (config.fps * config.dt)))
    times_rec = []
    positions_rec = []
    velocities_rec = []
    currents_rec = []
    cap_voltages_rec = []
    wire_temps_rec = []
    forces_rec = []

    # Trigger state
    triggered = False
    trigger_time = 0.0
    pulse_cut = False
    pulse_cut_time = 0.0
    coil_z = coil_params["position_mm"][2] if len(coil_params["position_mm"]) > 2 else coil_params["position_mm"][1]
    trigger_z = coil_z + config.pulse_trigger_offset

    print(f"\nSimulating {config.total_time}s at dt={config.dt*1000:.1f}ms ({n_steps} steps)...")

    for step in range(n_steps):
        t = step * config.dt

        # Record at frame rate
        if step % record_interval == 0:
            times_rec.append(t)
            positions_rec.append(pos.copy())
            velocities_rec.append(vel.copy())
            currents_rec.append(circuit_state["I"])
            cap_voltages_rec.append(circuit_state["Q_cap"] / rlc["capacitance_F"])
            wire_temps_rec.append(wire_temp)

        # Coil-local coordinates
        relative = pos - coil_pos
        z_along = np.dot(relative, coil_axis)

        # Check trigger
        if not triggered and z_along < 0:
            triggered = True
            trigger_time = t
            print(f"  Coil triggered at t={t:.4f}s, z_along={z_along:.1f}mm")

        # EM force computation
        em_force = np.zeros(3)
        current_A = 0.0

        if triggered:
            # Pulse cut when marble passes center
            if switch_type == "MOSFET" and z_along > 0 and not pulse_cut:
                pulse_cut = True
                pulse_cut_time = t
                print(f"  Pulse cut at t={t:.4f}s")

            if config.use_coupled_ode and not pulse_cut:
                # Step the coupled RLC ODE (sub-stepped for stability)
                vel_axial = float(np.dot(vel, coil_axis))
                circuit_state = coupled_rlc_step_substep(
                    circuit_state, config.dt, coil_params, rlc,
                    z_along, vel_axial,
                )
                # Use RMS current for force (F ~ I^2, so RMS is correct)
                current_A = circuit_state.get("_I_rms", abs(circuit_state["I"]))
            else:
                # Closed-form RLC current (or RL decay after cutoff)
                t_rel = t - trigger_time
                if pulse_cut:
                    t_cut_rel = pulse_cut_time - trigger_time
                    current_A = rlc_current_with_cutoff(t_rel, t_cut_rel, rlc)
                else:
                    current_A = rlc_current(t_rel, rlc)
                circuit_state["I"] = current_A

            if abs(current_A) > 1e-6:
                # B-field and force
                B_now, dBdz = solver.solve(current_A, np.array([0.0, z_along]))
                dBdt = (B_now - prev_B) / config.dt if config.dt > 0 else 0
                prev_B = B_now

                F_r, F_z = solver.get_force(current_A, relative,
                                             marble_vel=vel, dBdt=dBdt)

                # Convert to Cartesian
                em_force = F_z * coil_axis
                radial_vec = relative - z_along * coil_axis
                r = np.linalg.norm(radial_vec)
                if r > 1e-6:
                    em_force += F_r * (radial_vec / r)

                # Wire temperature update
                R_now = resistance_at_temperature(rlc["_R_dc_base"], wire_temp)
                R_total = R_now + rlc["_R_esr"] + rlc["_R_wiring"]
                dT = wire_temperature_rise(current_A, R_total, config.dt, rlc["_wire_mass_g"])
                wire_temp += dT

        if step % record_interval == 0:
            forces_rec.append(np.linalg.norm(em_force))

        # Integration: a = g + F/m
        accel = GRAVITY.copy()
        accel += em_force * 1000.0 / MARBLE_MASS  # mN -> mm/s^2

        vel = vel + accel * config.dt
        pos = pos + vel * config.dt

        # Collision
        collided, pos, vel = check_collision(pos, vel, MARBLE_RADIUS, mesh)

        # Floor check
        floor_z = mesh.bounds[0][2] - 50
        if pos[2] < floor_z:
            print(f"  Marble fell below floor at t={t:.3f}s")
            break

    # Convert to arrays
    times_arr = np.array(times_rec)
    pos_arr = np.array(positions_rec)
    vel_arr = np.array(velocities_rec)
    cur_arr = np.array(currents_rec)
    vcap_arr = np.array(cap_voltages_rec)
    temp_arr = np.array(wire_temps_rec)
    force_arr = np.array(forces_rec)

    print(f"  Completed: {len(times_arr)} frames recorded")
    print(f"  Final position: {pos}")
    print(f"  Max speed: {np.linalg.norm(vel_arr, axis=1).max():.1f} mm/s")
    print(f"  Peak current: {cur_arr.max():.1f} A")
    print(f"  Final wire temp: {wire_temp:.1f} C")

    # Save
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    save_trajectory(times_arr, pos_arr, vel_arr, filename="full_simulation.csv")

    if config.output_usd:
        write_usd_animation(times_arr, pos_arr, config.fps)

    plot_full_results(times_arr, pos_arr, vel_arr, cur_arr, vcap_arr,
                      temp_arr, force_arr, coil_params, rlc)

    return times_arr, pos_arr, vel_arr


def write_usd_animation(times, positions, fps):
    """Write marble trajectory as time-sampled USD transforms."""
    usd_path = ROOT / "usd" / "marble_trajectory.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(times) - 1)
    stage.SetTimeCodesPerSecond(fps)

    marble = UsdGeom.Xform.Define(stage, "/World/Marble")
    translate_op = marble.AddTranslateOp()

    for i, (t, pos) in enumerate(zip(times, positions)):
        translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])), i)

    stage.GetRootLayer().Save()
    print(f"  Saved USD animation: {usd_path} ({len(times)} frames)")


def plot_full_results(times, positions, velocities, currents, cap_voltages,
                      wire_temps, forces, coil_params, rlc):
    """Generate comprehensive result plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    speed = np.linalg.norm(velocities, axis=1)
    L = coil_params["length_mm"]
    coil_z = coil_params["position_mm"][2] if len(coil_params["position_mm"]) > 2 else coil_params["position_mm"][1]

    # Height vs time
    axes[0, 0].plot(times, positions[:, 2], "b-")
    axes[0, 0].axhspan(coil_z - L / 2, coil_z + L / 2, alpha=0.1, color="red", label="coil")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Z (mm)")
    axes[0, 0].set_title("Height vs time")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Speed vs time
    axes[0, 1].plot(times, speed, "r-")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Speed (mm/s)")
    axes[0, 1].set_title("Speed vs time")
    axes[0, 1].grid(True, alpha=0.3)

    # Current vs time
    axes[0, 2].plot(times, currents, "g-")
    axes[0, 2].set_xlabel("Time (s)")
    axes[0, 2].set_ylabel("Current (A)")
    axes[0, 2].set_title(f"Coil current ({rlc['regime']}, zeta={rlc['zeta']:.3f})")
    axes[0, 2].grid(True, alpha=0.3)

    # Capacitor voltage
    axes[1, 0].plot(times, cap_voltages, "m-")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("V_cap (V)")
    axes[1, 0].set_title("Capacitor voltage")
    axes[1, 0].grid(True, alpha=0.3)

    # Wire temperature
    axes[1, 1].plot(times, wire_temps, "orange")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Temperature (C)")
    axes[1, 1].set_title("Wire temperature")
    axes[1, 1].grid(True, alpha=0.3)

    # EM force magnitude
    min_len = min(len(times), len(forces))
    axes[1, 2].plot(times[:min_len], forces[:min_len], "r-")
    axes[1, 2].set_xlabel("Time (s)")
    axes[1, 2].set_ylabel("Force (mN)")
    axes[1, 2].set_title("EM force magnitude")
    axes[1, 2].grid(True, alpha=0.3)

    # XY trajectory
    axes[2, 0].plot(positions[:, 0], positions[:, 1], "b-", linewidth=0.5)
    axes[2, 0].plot(positions[0, 0], positions[0, 1], "go", markersize=8, label="start")
    axes[2, 0].plot(positions[-1, 0], positions[-1, 1], "rs", markersize=8, label="end")
    axes[2, 0].set_xlabel("X (mm)")
    axes[2, 0].set_ylabel("Y (mm)")
    axes[2, 0].set_title("XY trajectory")
    axes[2, 0].set_aspect("equal")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # XZ trajectory (side view)
    axes[2, 1].plot(positions[:, 0], positions[:, 2], "m-", linewidth=0.5)
    axes[2, 1].axhspan(coil_z - L / 2, coil_z + L / 2, alpha=0.1, color="red")
    axes[2, 1].set_xlabel("X (mm)")
    axes[2, 1].set_ylabel("Z (mm)")
    axes[2, 1].set_title("XZ trajectory (side view)")
    axes[2, 1].grid(True, alpha=0.3)

    # 3D path
    ax3d = fig.add_subplot(3, 3, 9, projection="3d")
    ax3d.plot3D(positions[:, 0], positions[:, 1], positions[:, 2], "b-", linewidth=0.3)
    ax3d.scatter(*positions[0], color="green", s=50, label="start")
    ax3d.scatter(*positions[-1], color="red", s=50, label="end")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("3D trajectory")
    ax3d.legend()
    axes[2, 2].set_visible(False)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "full_simulation.png", dpi=150)
    plt.close()
    print(f"  Saved plot: {PLOTS_DIR / 'full_simulation.png'}")


def main():
    config = SimulationConfig()

    if HAS_WARP:
        config.use_warp = True
        print("Warp available")

    print(f"\n=== Full Simulation Loop (RLC + Warp) ===")
    print(f"  Coupled ODE: {config.use_coupled_ode}")
    print(f"  dt={config.dt*1000:.1f}ms, total={config.total_time}s\n")

    run_simulation(config)


if __name__ == "__main__":
    main()
