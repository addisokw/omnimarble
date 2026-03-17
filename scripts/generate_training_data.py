"""Generate training data for the PINN by sampling the analytical B-field.

Samples across spatial domain and parameter space with extra density near coil.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "coil_params.json"
DATA_DIR = ROOT / "data"

sys.path.insert(0, str(ROOT / "scripts"))
from analytical_bfield import solenoid_field


def generate_spatial_points(n_dense: int = 30000, n_sparse: int = 20000) -> np.ndarray:
    """Generate sample points with higher density near the coil region.

    Returns array of shape (N, 2) with columns [r, z] in mm.
    """
    rng = np.random.default_rng(42)

    # Dense samples near coil: r ∈ [0, 30], z ∈ [-40, 40]
    r_dense = rng.uniform(0, 30, n_dense)
    z_dense = rng.uniform(-40, 40, n_dense)

    # Sparse samples covering far field: r ∈ [0, 100], z ∈ [-150, 150]
    r_sparse = rng.uniform(0, 100, n_sparse)
    z_sparse = rng.uniform(-150, 150, n_sparse)

    r = np.concatenate([r_dense, r_sparse])
    z = np.concatenate([z_dense, z_sparse])

    return np.column_stack([r, z])


def generate_parameter_samples(n_configs: int = 50, rng=None) -> list[dict]:
    """Generate random coil parameter configurations for training."""
    if rng is None:
        rng = np.random.default_rng(123)

    configs = []
    for _ in range(n_configs):
        config = {
            "current_A": float(rng.uniform(0.5, 20.0)),
            "num_turns": int(rng.integers(10, 61)),
            "inner_radius_mm": float(rng.uniform(6.0, 20.0)),
            "length_mm": float(rng.uniform(15.0, 60.0)),
        }
        # Outer radius = inner + some winding thickness
        config["outer_radius_mm"] = config["inner_radius_mm"] + float(rng.uniform(2.0, 8.0))
        configs.append(config)

    return configs


def compute_fields(points: np.ndarray, coil_params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compute B_r and B_z for all points with given coil parameters.

    Returns (B_r, B_z) arrays of shape (N,).
    """
    N = len(points)
    Br = np.zeros(N)
    Bz = np.zeros(N)

    for i in range(N):
        r, z = points[i]
        br, bz = solenoid_field(r, z, coil_params)
        Br[i] = br
        Bz[i] = bz

    return Br, Bz


def main():
    print("=== PINN Training Data Generation ===\n")

    # Generate spatial sample points
    points = generate_spatial_points()
    print(f"Generated {len(points)} spatial sample points")

    # Generate parameter configurations
    param_configs = generate_parameter_samples(n_configs=50)

    # Also include the default config
    default_params = json.loads(CONFIG_PATH.read_text())
    param_configs.insert(0, {
        "current_A": default_params["max_current_A"],
        "num_turns": default_params["num_turns"],
        "inner_radius_mm": default_params["inner_radius_mm"],
        "outer_radius_mm": default_params["outer_radius_mm"],
        "length_mm": default_params["length_mm"],
    })
    print(f"Parameter configurations: {len(param_configs)}")

    # Subsample points per config to keep total manageable
    points_per_config = 1000
    rng = np.random.default_rng(99)
    total_points = len(param_configs) * points_per_config
    print(f"Total training samples: {total_points}")

    # Storage arrays
    all_inputs = np.zeros((total_points, 6))  # r, z, I, N, R_inner, L
    all_Br = np.zeros(total_points)
    all_Bz = np.zeros(total_points)

    idx = 0
    for ci, config in enumerate(param_configs):
        if ci % 10 == 0:
            print(f"  Processing config {ci+1}/{len(param_configs)}...")

        # Random subset of points
        subset_idx = rng.choice(len(points), size=points_per_config, replace=False)
        subset = points[subset_idx]

        Br, Bz = compute_fields(subset, config)

        for i in range(points_per_config):
            all_inputs[idx] = [
                subset[i, 0],  # r
                subset[i, 1],  # z
                config["current_A"],
                config["num_turns"],
                (config["inner_radius_mm"] + config["outer_radius_mm"]) / 2,  # R_mean
                config["length_mm"],
            ]
            all_Br[idx] = Br[i]
            all_Bz[idx] = Bz[i]
            idx += 1

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "pinn_inputs.npy", all_inputs[:idx])
    np.save(DATA_DIR / "pinn_Br.npy", all_Br[:idx])
    np.save(DATA_DIR / "pinn_Bz.npy", all_Bz[:idx])

    print(f"\nSaved {idx} training samples to {DATA_DIR}:")
    print(f"  pinn_inputs.npy: shape {all_inputs[:idx].shape} (r, z, I, N, R, L)")
    print(f"  pinn_Br.npy: shape {all_Br[:idx].shape}")
    print(f"  pinn_Bz.npy: shape {all_Bz[:idx].shape}")

    # Stats
    print(f"\nField statistics:")
    print(f"  B_r: [{all_Br[:idx].min():.2e}, {all_Br[:idx].max():.2e}] T")
    print(f"  B_z: [{all_Bz[:idx].min():.2e}, {all_Bz[:idx].max():.2e}] T")


if __name__ == "__main__":
    main()
