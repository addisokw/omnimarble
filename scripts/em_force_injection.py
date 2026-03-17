"""Integrate EM force with the physics simulation.

Runs the marble simulation with coil force applied during a triggered pulse.
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
from run_physics_test import (
    DT,
    MARBLE_RADIUS,
    load_scene_params,
    load_track_mesh,
    plot_trajectory,
    save_trajectory,
    simulate,
)


def make_coil_force_fn(coil_params: dict, trigger_zone_z: float = -30.0):
    """Create a force function that models the coil EM force with LR circuit pulse.

    The coil axis is along X in world coords (per config), centered at coil position.
    The trigger fires when the marble enters the trigger zone.

    Args:
        coil_params: coil configuration
        trigger_zone_z: z-position threshold to trigger pulse (mm)
    """
    V = coil_params["supply_voltage_V"]
    R = coil_params["resistance_ohm"]
    L = coil_params["inductance_uH"] * 1e-6  # Convert to H
    pulse_width = coil_params["pulse_width_ms"] * 1e-3  # Convert to s
    coil_pos = np.array(coil_params["position_mm"])
    coil_axis = np.array(coil_params.get("axis", [1, 0, 0]), dtype=float)
    coil_axis = coil_axis / np.linalg.norm(coil_axis)

    # LR circuit time constant
    tau = L / R
    I_max = V / R

    state = {"triggered": False, "trigger_time": None}

    def current_at_time(t: float, t_trigger: float) -> float:
        """LR circuit current: rise during pulse, exponential decay after."""
        dt = t - t_trigger
        if dt < 0:
            return 0.0
        if dt < pulse_width:
            # Rising: I(t) = (V/R)(1 - e^(-Rt/L))
            return I_max * (1 - math.exp(-dt / tau))
        else:
            # Decay after pulse ends
            I_at_end = I_max * (1 - math.exp(-pulse_width / tau))
            return I_at_end * math.exp(-(dt - pulse_width) / tau)

    def force_fn(t: float, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Compute EM force on marble at given state."""
        # Check trigger
        if not state["triggered"]:
            # Trigger when marble Z drops below trigger zone
            if pos[2] < trigger_zone_z:
                state["triggered"] = True
                state["trigger_time"] = t
                print(f"  Coil triggered at t={t:.4f}s, marble z={pos[2]:.1f}mm")

        if not state["triggered"]:
            return np.zeros(3)

        # Current from LR circuit
        I_t = current_at_time(t, state["trigger_time"])
        if abs(I_t) < 1e-6:
            return np.zeros(3)

        # Transform marble position to coil-local cylindrical coords
        # Coil center is at coil_pos, axis is coil_axis
        relative = pos - coil_pos
        z_along_axis = np.dot(relative, coil_axis)
        radial_vec = relative - z_along_axis * coil_axis
        r = np.linalg.norm(radial_vec)

        # Create modified params with current current
        params_now = coil_params.copy()
        params_now["current_A"] = I_t

        # Compute force in cylindrical coords
        F_r, F_z = ferromagnetic_force(r, z_along_axis, MARBLE_RADIUS, params_now)

        # Convert back to Cartesian
        force = F_z * coil_axis  # Axial component
        if r > 1e-6:
            radial_dir = radial_vec / r
            force += F_r * radial_dir

        return force

    return force_fn


def main():
    print("=== EM Force Injection Simulation ===\n")

    params = json.loads(CONFIG_PATH.read_text())
    print(f"Coil: {params['num_turns']} turns, R_mean={(params['inner_radius_mm']+params['outer_radius_mm'])/2:.1f}mm")
    print(f"  Pulse: {params['supply_voltage_V']}V, {params['resistance_ohm']}ohm, {params['inductance_uH']}uH")
    print(f"  Max current: {params['supply_voltage_V']/params['resistance_ohm']:.1f}A")

    print("\nLoading track mesh...")
    mesh = load_track_mesh()

    initial_pos, _ = load_scene_params()
    print(f"Marble start: {initial_pos}")

    # Determine trigger zone — slightly before coil position
    coil_z = params["position_mm"][2]
    trigger_z = coil_z + 15  # Trigger when marble approaches coil from above

    force_fn = make_coil_force_fn(params, trigger_zone_z=trigger_z)
    print(f"Trigger zone: z < {trigger_z:.1f} mm")

    print("\nRunning simulation with EM force...")
    times, positions, velocities = simulate(initial_pos, mesh, external_force_fn=force_fn)

    save_trajectory(times, positions, velocities, filename="em_launch.csv")
    plot_trajectory(times, positions, velocities, filename="em_launch.png")

    # Additional force/velocity analysis plot
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    speed = np.linalg.norm(velocities, axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(times, speed, "b-", linewidth=1)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (mm/s)")
    axes[0].set_title("Marble speed with EM launch")
    axes[0].grid(True, alpha=0.3)

    # Plot force magnitude along trajectory
    force_mags = []
    params_snapshot = params.copy()
    for i in range(len(times)):
        relative = positions[i] - np.array(params["position_mm"])
        coil_axis = np.array(params.get("axis", [1, 0, 0]), dtype=float)
        coil_axis = coil_axis / np.linalg.norm(coil_axis)
        z_along = np.dot(relative, coil_axis)
        r = np.linalg.norm(relative - z_along * coil_axis)
        Fr, Fz = ferromagnetic_force(r, z_along, MARBLE_RADIUS, params_snapshot)
        force_mags.append(math.sqrt(Fr**2 + Fz**2))

    axes[1].plot(times, force_mags, "r-", linewidth=1)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("EM Force magnitude (mN)")
    axes[1].set_title("EM force magnitude (static current)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "em_launch_analysis.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'em_launch_analysis.png'}")


if __name__ == "__main__":
    main()
