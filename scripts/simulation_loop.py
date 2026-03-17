"""Full simulation orchestration loop.

Pipeline: USD scene load → per-frame PINN/analytical query → force injection → integration → USD update.
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

# Try to import GPU components
try:
    import torch
    from train_pinn import BFieldPINN
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import warp as wp
    from warp_em_kernel import HybridEMForceComputer, lr_circuit_current
    HAS_WARP = True
except ImportError:
    HAS_WARP = False


class SimulationConfig:
    """Configuration for the full simulation."""
    dt: float = 0.0005
    total_time: float = 5.0
    use_pinn: bool = False  # Use PINN instead of analytical
    use_warp: bool = False  # Use Warp GPU kernels
    pulse_trigger_offset: float = 15.0  # mm above coil to trigger
    output_usd: bool = True  # Write time-sampled USD output
    fps: int = 60  # USD output frame rate


def load_pinn_model():
    """Load PINN model if available."""
    if not HAS_TORCH:
        return None

    model_path = ROOT / "models" / "pinn_checkpoint" / "pinn_best.pt"
    if not model_path.exists():
        print("  PINN checkpoint not found, using analytical fallback")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = BFieldPINN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded PINN model (step {checkpoint['step']}) on {device}")
    return model


def analytical_force_fn(pos, coil_params, current_A):
    """Compute EM force using analytical B-field."""
    coil_pos = np.array(coil_params["position_mm"])
    coil_axis = np.array(coil_params.get("axis", [1, 0, 0]), dtype=float)
    coil_axis = coil_axis / np.linalg.norm(coil_axis)

    relative = pos - coil_pos
    z_along = np.dot(relative, coil_axis)
    radial_vec = relative - z_along * coil_axis
    r = np.linalg.norm(radial_vec)

    params_now = coil_params.copy()
    params_now["current_A"] = current_A

    F_r, F_z = ferromagnetic_force(r, z_along, MARBLE_RADIUS, params_now)

    force = F_z * coil_axis
    if r > 1e-6:
        force += F_r * (radial_vec / r)

    return force


def lr_current(t, trigger_time, coil_params):
    """LR circuit current model."""
    V = coil_params["supply_voltage_V"]
    R = coil_params["resistance_ohm"]
    L = coil_params["inductance_uH"] * 1e-6
    pulse_width = coil_params["pulse_width_ms"] * 1e-3
    tau = L / R
    I_max = V / R

    dt = t - trigger_time
    if dt < 0:
        return 0.0
    if dt < pulse_width:
        return I_max * (1 - math.exp(-dt / tau))
    else:
        I_end = I_max * (1 - math.exp(-pulse_width / tau))
        return I_end * math.exp(-(dt - pulse_width) / tau)


def run_simulation(config: SimulationConfig):
    """Run the full simulation loop."""
    coil_params = json.loads(CONFIG_PATH.read_text())

    print("Loading track mesh...")
    mesh = load_track_mesh()
    print(f"  Track: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    initial_pos, _ = load_scene_params()
    print(f"  Marble start: {initial_pos}")

    # PINN or analytical
    pinn_model = None
    em_computer = None
    if config.use_pinn:
        pinn_model = load_pinn_model()
        if pinn_model and HAS_WARP and config.use_warp:
            em_computer = HybridEMForceComputer(
                pinn_model=pinn_model,
                coil_params=coil_params,
                marble_radius=MARBLE_RADIUS,
            )
            print("  Using PINN + Warp hybrid pipeline")
        elif pinn_model:
            print("  Using PINN (CPU/GPU, no Warp)")
    if pinn_model is None:
        print("  Using analytical B-field")

    # Simulation state
    n_steps = int(config.total_time / config.dt)
    pos = initial_pos.copy()
    vel = np.zeros(3)

    # Recording
    record_interval = max(1, int(1.0 / (config.fps * config.dt)))
    times_rec = []
    positions_rec = []
    velocities_rec = []
    currents_rec = []

    # Trigger state
    triggered = False
    trigger_time = 0.0
    coil_z = coil_params["position_mm"][2]
    trigger_z = coil_z + config.pulse_trigger_offset

    print(f"\nSimulating {config.total_time}s at dt={config.dt*1000:.1f}ms ({n_steps} steps)...")

    for step in range(n_steps):
        t = step * config.dt

        # Record at frame rate
        if step % record_interval == 0:
            times_rec.append(t)
            positions_rec.append(pos.copy())
            velocities_rec.append(vel.copy())

        # Check trigger
        if not triggered and pos[2] < trigger_z:
            triggered = True
            trigger_time = t
            print(f"  Coil triggered at t={t:.4f}s, z={pos[2]:.1f}mm")

        # Compute EM force
        em_force = np.zeros(3)
        current_A = 0.0
        if triggered:
            current_A = lr_current(t, trigger_time, coil_params)
            if abs(current_A) > 1e-6:
                if em_computer is not None:
                    forces = em_computer.compute_force_pinn(
                        pos.reshape(1, 3), current_A
                    )
                    em_force = forces[0]
                else:
                    em_force = analytical_force_fn(pos, coil_params, current_A)

        if step % record_interval == 0:
            currents_rec.append(current_A)

        # Integration: a = g + F/m
        accel = GRAVITY.copy()
        accel += em_force * 1000.0 / MARBLE_MASS

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

    print(f"  Completed: {len(times_arr)} frames recorded")
    print(f"  Final position: {pos}")
    print(f"  Max speed: {np.linalg.norm(vel_arr, axis=1).max():.1f} mm/s")

    # Save trajectory
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    save_trajectory(times_arr, pos_arr, vel_arr, filename="full_simulation.csv")

    # Write USD time samples
    if config.output_usd:
        write_usd_animation(times_arr, pos_arr, config.fps)

    # Plot results
    plot_full_results(times_arr, pos_arr, vel_arr, cur_arr, coil_params)

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
        frame = i  # Each recorded frame maps to one USD time code
        translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])), frame)

    stage.GetRootLayer().Save()
    print(f"  Saved USD animation: {usd_path} ({len(times)} frames)")


def plot_full_results(times, positions, velocities, currents, coil_params):
    """Generate comprehensive result plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    speed = np.linalg.norm(velocities, axis=1)
    L = coil_params["length_mm"]
    coil_z = coil_params["position_mm"][2]

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
    axes[0, 2].set_title("Coil current vs time")
    axes[0, 2].grid(True, alpha=0.3)

    # XY trajectory
    axes[1, 0].plot(positions[:, 0], positions[:, 1], "b-", linewidth=0.5)
    axes[1, 0].plot(positions[0, 0], positions[0, 1], "go", markersize=8, label="start")
    axes[1, 0].plot(positions[-1, 0], positions[-1, 1], "rs", markersize=8, label="end")
    axes[1, 0].set_xlabel("X (mm)")
    axes[1, 0].set_ylabel("Y (mm)")
    axes[1, 0].set_title("XY trajectory")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # XZ trajectory (side view)
    axes[1, 1].plot(positions[:, 0], positions[:, 2], "m-", linewidth=0.5)
    axes[1, 1].axhspan(coil_z - L / 2, coil_z + L / 2, alpha=0.1, color="red")
    axes[1, 1].set_xlabel("X (mm)")
    axes[1, 1].set_ylabel("Z (mm)")
    axes[1, 1].set_title("XZ trajectory (side view)")
    axes[1, 1].grid(True, alpha=0.3)

    # 3D path visualization
    ax3d = fig.add_subplot(2, 3, 6, projection="3d")
    ax3d.plot3D(positions[:, 0], positions[:, 1], positions[:, 2], "b-", linewidth=0.3)
    ax3d.scatter(*positions[0], color="green", s=50, label="start")
    ax3d.scatter(*positions[-1], color="red", s=50, label="end")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("3D trajectory")
    ax3d.legend()
    axes[1, 2].set_visible(False)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "full_simulation.png", dpi=150)
    plt.close()
    print(f"  Saved plot: {PLOTS_DIR / 'full_simulation.png'}")


def main():
    config = SimulationConfig()

    # Auto-detect capabilities
    if HAS_TORCH:
        model_exists = (ROOT / "models" / "pinn_checkpoint" / "pinn_best.pt").exists()
        if model_exists:
            config.use_pinn = True
            print("PINN model found — will use for EM force computation")
    if HAS_WARP:
        config.use_warp = True
        print("Warp available — will use GPU kernels")

    print(f"\n=== Full Simulation Loop ===")
    print(f"  PINN: {config.use_pinn}, Warp: {config.use_warp}")
    print(f"  dt={config.dt*1000:.1f}ms, total={config.total_time}s\n")

    run_simulation(config)


if __name__ == "__main__":
    main()
