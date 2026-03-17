"""Basic physics test — gravity-only marble drop with Euler integration.

Fallback standalone simulation that doesn't require Isaac Sim or PhysX runtime.
Uses ray-casting against track mesh for collision detection.
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = ROOT / "usd" / "marble_coaster_scene.usda"
TRACK_STL = ROOT / "data" / "track.stl"
RESULTS_DIR = ROOT / "results"
TRAJ_DIR = RESULTS_DIR / "trajectories"
PLOTS_DIR = RESULTS_DIR / "plots"

# Simulation parameters
DT = 0.0005  # 0.5 ms timestep
TOTAL_TIME = 5.0  # seconds
GRAVITY = np.array([0, 0, -9810.0])  # mm/s²
MARBLE_RADIUS = 5.0  # mm
MARBLE_MASS = (4 / 3) * math.pi * MARBLE_RADIUS**3 * 7.8e-3  # grams
RESTITUTION = 0.4
FRICTION_COEFF = 0.35


def load_scene_params() -> tuple[np.ndarray, float]:
    """Load marble initial position from USD scene."""
    if SCENE_PATH.exists():
        stage = Usd.Stage.Open(str(SCENE_PATH))
        marble_prim = stage.GetPrimAtPath("/World/Marble")
        if marble_prim:
            xform = UsdGeom.Xformable(marble_prim)
            xform_ops = xform.GetOrderedXformOps()
            for op in xform_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    pos = op.Get()
                    return np.array([pos[0], pos[1], pos[2]]), MARBLE_RADIUS

    return np.array([0, 0, 50.0]), MARBLE_RADIUS


def load_track_mesh() -> trimesh.Trimesh:
    """Load track mesh for collision detection."""
    raw = trimesh.load(str(TRACK_STL))
    if isinstance(raw, trimesh.Scene):
        meshes = list(raw.geometry.values())
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    else:
        mesh = raw
    return mesh


def check_collision(
    pos: np.ndarray, vel: np.ndarray, radius: float, mesh: trimesh.Trimesh
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Check if marble collides with track mesh using ray casting.

    Cast rays downward and in velocity direction to find nearby surfaces.
    Returns (collided, new_pos, new_vel).
    """
    # Find closest point on mesh to marble center
    closest_points, distances, face_ids = mesh.nearest.on_surface([pos])
    dist = distances[0]
    closest = closest_points[0]

    if dist > radius * 3:
        # Far from track — no collision
        return False, pos, vel

    if dist < radius:
        # Collision detected
        # Normal from closest point to marble center
        normal = pos - closest
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            normal = np.array([0, 0, 1.0])
        else:
            normal = normal / norm_len

        # Push marble out of collision
        penetration = radius - dist
        new_pos = pos + normal * penetration

        # Reflect velocity with restitution
        v_n = np.dot(vel, normal) * normal
        v_t = vel - v_n

        if np.dot(vel, normal) < 0:
            # Moving into surface
            new_vel = v_t * (1 - FRICTION_COEFF) - v_n * RESTITUTION
        else:
            new_vel = vel

        return True, new_pos, new_vel

    return False, pos, vel


def simulate(
    initial_pos: np.ndarray,
    mesh: trimesh.Trimesh,
    external_force_fn=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Euler integration simulation.

    Args:
        initial_pos: starting position (mm)
        mesh: track mesh for collisions
        external_force_fn: optional callable(t, pos, vel) -> force_vector (mN)

    Returns:
        (times, positions, velocities) arrays
    """
    n_steps = int(TOTAL_TIME / DT)
    times = np.zeros(n_steps)
    positions = np.zeros((n_steps, 3))
    velocities = np.zeros((n_steps, 3))

    pos = initial_pos.copy()
    vel = np.zeros(3)

    for i in range(n_steps):
        times[i] = i * DT
        positions[i] = pos
        velocities[i] = vel

        # Compute forces
        # Gravity: F = m * g (mass in grams, gravity in mm/s², so F in g·mm/s² = μN)
        # Actually we just work with acceleration: a = g + F_ext/m
        accel = GRAVITY.copy()

        if external_force_fn is not None:
            # External force in mN, mass in grams
            # a = F/m → mN / g = (1e-3 N) / (1e-3 kg) = m/s² → need mm/s²: multiply by 1e3
            # Wait: mN / g = 1e-3 N / 1e-3 kg = 1 m/s² = 1000 mm/s²
            f_ext = external_force_fn(times[i], pos, vel)
            accel += f_ext * 1000.0 / MARBLE_MASS  # Convert mN to mm/s² via mass in g

        # Semi-implicit Euler
        vel = vel + accel * DT
        pos = pos + vel * DT

        # Collision detection
        collided, pos, vel = check_collision(pos, vel, MARBLE_RADIUS, mesh)

        # Floor check (safety)
        floor_z = mesh.bounds[0][2] - 50
        if pos[2] < floor_z:
            print(f"Marble fell below floor at t={times[i]:.3f}s")
            times = times[: i + 1]
            positions = positions[: i + 1]
            velocities = velocities[: i + 1]
            break

    return times, positions, velocities


def save_trajectory(times, positions, velocities, filename="gravity_drop.csv"):
    """Save trajectory to CSV."""
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    path = TRAJ_DIR / filename
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "x_mm", "y_mm", "z_mm", "vx_mm_s", "vy_mm_s", "vz_mm_s"])
        for i in range(len(times)):
            writer.writerow([
                f"{times[i]:.6f}",
                f"{positions[i, 0]:.4f}",
                f"{positions[i, 1]:.4f}",
                f"{positions[i, 2]:.4f}",
                f"{velocities[i, 0]:.4f}",
                f"{velocities[i, 1]:.4f}",
                f"{velocities[i, 2]:.4f}",
            ])
    print(f"Saved trajectory: {path}")


def plot_trajectory(times, positions, velocities, filename="gravity_drop.png"):
    """Plot marble height and speed vs time."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Height vs time
    axes[0, 0].plot(times, positions[:, 2], "b-", linewidth=1)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Z position (mm)")
    axes[0, 0].set_title("Marble height vs time")
    axes[0, 0].grid(True, alpha=0.3)

    # Speed vs time
    speed = np.linalg.norm(velocities, axis=1)
    axes[0, 1].plot(times, speed, "r-", linewidth=1)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Speed (mm/s)")
    axes[0, 1].set_title("Marble speed vs time")
    axes[0, 1].grid(True, alpha=0.3)

    # XY trajectory
    axes[1, 0].plot(positions[:, 0], positions[:, 1], "g-", linewidth=0.5)
    axes[1, 0].plot(positions[0, 0], positions[0, 1], "go", markersize=8, label="start")
    axes[1, 0].set_xlabel("X (mm)")
    axes[1, 0].set_ylabel("Y (mm)")
    axes[1, 0].set_title("XY trajectory")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 3D trajectory (XZ view)
    axes[1, 1].plot(positions[:, 0], positions[:, 2], "m-", linewidth=0.5)
    axes[1, 1].plot(positions[0, 0], positions[0, 2], "mo", markersize=8, label="start")
    axes[1, 1].set_xlabel("X (mm)")
    axes[1, 1].set_ylabel("Z (mm)")
    axes[1, 1].set_title("XZ trajectory (side view)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot: {path}")


def main():
    print("Loading track mesh...")
    mesh = load_track_mesh()
    print(f"  Track: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"  Bounds: {mesh.bounds[0]} to {mesh.bounds[1]}")

    print("Loading scene parameters...")
    initial_pos, radius = load_scene_params()
    print(f"  Marble start: {initial_pos}")
    print(f"  Marble radius: {radius} mm, mass: {MARBLE_MASS:.2f} g")

    print(f"\nRunning gravity-only simulation ({TOTAL_TIME}s, dt={DT*1000:.1f}ms)...")
    times, positions, velocities = simulate(initial_pos, mesh)
    print(f"  Simulated {len(times)} steps")
    print(f"  Final position: {positions[-1]}")
    print(f"  Max speed: {np.linalg.norm(velocities, axis=1).max():.1f} mm/s")

    save_trajectory(times, positions, velocities)
    plot_trajectory(times, positions, velocities)


if __name__ == "__main__":
    main()
