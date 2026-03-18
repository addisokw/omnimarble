"""Integrate EM force with the physics simulation.

Phase 2: Uses RLC capacitor discharge model and Warp B-field solver
(with analytical fallback) including saturation and eddy current effects.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "coil_params.json"
PLOTS_DIR = ROOT / "results" / "plots"

from analytical_bfield import ferromagnetic_force, solenoid_field
from rlc_circuit import (
    compute_rlc_params,
    rlc_current,
    rlc_current_with_cutoff,
    compute_winding_geometry,
    compute_dc_resistance,
    compute_ac_resistance,
    compute_multilayer_inductance,
)
from warp_bfield_solver import WarpBFieldSolver
from run_physics_test import (
    DT,
    MARBLE_RADIUS,
    load_scene_params,
    load_track_mesh,
    plot_trajectory,
    save_trajectory,
    simulate,
)


def build_rlc_from_config(params: dict) -> dict:
    """Derive RLC parameters from config, same as generate_coil.py does."""
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

    # Estimate frequency for AC resistance
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
    return compute_rlc_params(rlc_input)


def make_coil_force_fn(coil_params: dict, trigger_zone_z: float = -30.0):
    """Create a force function using RLC discharge + Warp B-field solver.

    Args:
        coil_params: coil configuration
        trigger_zone_z: z-position threshold to trigger pulse (mm)
    """
    rlc = build_rlc_from_config(coil_params)
    solver = WarpBFieldSolver(coil_params, chi_eff=3.0)

    coil_pos = np.array(coil_params["position_mm"])
    coil_axis = np.array(coil_params.get("axis", [0, 1, 0]), dtype=float)
    coil_axis = coil_axis / np.linalg.norm(coil_axis)

    switch_type = coil_params.get("switch_type", "MOSFET")

    state = {"triggered": False, "trigger_time": None, "pulse_cut": False,
             "pulse_cut_time": None, "prev_B": 0.0}

    print(f"  RLC: {rlc['regime']}, I_peak={rlc['peak_current_A']:.1f}A, "
          f"zeta={rlc['zeta']:.4f}")
    print(f"  Stored energy: {rlc['stored_energy_J']:.2f} J")

    def force_fn(t: float, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Compute EM force on marble at given state."""
        # Check trigger
        if not state["triggered"]:
            if pos[2] < trigger_zone_z:
                state["triggered"] = True
                state["trigger_time"] = t
                print(f"  Coil triggered at t={t:.4f}s, marble z={pos[2]:.1f}mm")

        if not state["triggered"]:
            return np.zeros(3)

        # Transform to coil-local coords
        relative = pos - coil_pos
        z_along_axis = np.dot(relative, coil_axis)

        # Pulse cut when marble passes center (for MOSFET)
        if switch_type == "MOSFET" and z_along_axis > 0 and not state["pulse_cut"]:
            state["pulse_cut"] = True
            state["pulse_cut_time"] = t
            print(f"  Pulse cut at t={t:.4f}s, z_along={z_along_axis:.1f}mm")

        # RLC current
        t_cutoff = state["pulse_cut_time"] if state["pulse_cut"] else float('inf')
        I_t = rlc_current_with_cutoff(t - state["trigger_time"], t_cutoff - state["trigger_time"], rlc)

        if abs(I_t) < 1e-6:
            return np.zeros(3)

        # Compute B for dB/dt estimate
        B_now, _ = solver.solve(I_t, np.array([0.0, z_along_axis]))
        dt_est = DT if DT > 0 else 0.001
        dBdt = (B_now - state["prev_B"]) / dt_est
        state["prev_B"] = B_now

        # Force with saturation + eddy currents
        F_r, F_z = solver.get_force(I_t, relative, marble_vel=vel, dBdt=dBdt)

        # Convert back to Cartesian
        radial_vec = relative - z_along_axis * coil_axis
        r = np.linalg.norm(radial_vec)
        force = F_z * coil_axis
        if r > 1e-6:
            radial_dir = radial_vec / r
            force += F_r * radial_dir

        return force

    return force_fn


def main():
    print("=== EM Force Injection Simulation (RLC + Warp) ===\n")

    params = json.loads(CONFIG_PATH.read_text())
    print(f"Coil: {params['num_turns']} turns, "
          f"R_inner={params['inner_radius_mm']}mm, R_outer={params['outer_radius_mm']}mm")
    print(f"  Cap: {params.get('capacitance_uF', 'N/A')}uF at {params.get('charge_voltage_V', 'N/A')}V")

    print("\nLoading track mesh...")
    mesh = load_track_mesh()

    initial_pos, _ = load_scene_params()
    print(f"Marble start: {initial_pos}")

    # Determine trigger zone
    coil_z = params["position_mm"][2]
    trigger_z = coil_z + 15

    force_fn = make_coil_force_fn(params, trigger_zone_z=trigger_z)
    print(f"Trigger zone: z < {trigger_z:.1f} mm")

    print("\nRunning simulation with EM force...")
    times, positions, velocities = simulate(initial_pos, mesh, external_force_fn=force_fn)

    save_trajectory(times, positions, velocities, filename="em_launch.csv")
    plot_trajectory(times, positions, velocities, filename="em_launch.png")

    # Analysis plot
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    speed = np.linalg.norm(velocities, axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(times, speed, "b-", linewidth=1)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (mm/s)")
    axes[0].set_title("Marble speed with EM launch (RLC discharge)")
    axes[0].grid(True, alpha=0.3)

    # RLC current waveform
    rlc = build_rlc_from_config(params)
    t_pulse = np.linspace(0, rlc.get("effective_pulse_duration_s", 0.01) * 2, 500)
    i_pulse = [rlc_current(t, rlc) for t in t_pulse]

    axes[1].plot(t_pulse * 1e3, i_pulse, "r-", linewidth=2)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Current (A)")
    axes[1].set_title(f"RLC Discharge Current ({rlc['regime']}, zeta={rlc['zeta']:.3f})")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "em_launch_analysis.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'em_launch_analysis.png'}")


if __name__ == "__main__":
    main()
