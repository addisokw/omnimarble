"""Generate training data for the field-accuracy PINN baseline.

This is the v4 recipe: boundary-enriched sampling with design-space grid
corners, but without gradient targets or failure-mined configs. Produces
a model optimized for strict pointwise field accuracy (L1 max < 20%).

Use generate_training_data_designspace.py for the design-exploration model.
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
from analytical_bfield import solenoid_field_batch


def generate_spatial_points(
    n_dense: int = 30000,
    n_boundary: int = 20000,
    n_axis: int = 5000,
    n_sparse: int = 20000,
) -> np.ndarray:
    """Generate sample points with extra density at coil boundaries and on-axis."""
    rng = np.random.default_rng(42)

    r_dense = rng.uniform(0, 30, n_dense)
    z_dense = rng.uniform(-40, 40, n_dense)

    z_ends_pos = rng.normal(15, 5, n_boundary // 4)
    z_ends_neg = rng.normal(-15, 5, n_boundary // 4)
    r_ends = rng.uniform(0, 30, n_boundary // 2)
    r_winding = np.abs(rng.normal(15, 3, n_boundary // 2))
    z_winding = rng.uniform(-40, 40, n_boundary // 2)

    r_boundary = np.concatenate([r_ends, r_winding])
    z_boundary = np.concatenate([z_ends_pos, z_ends_neg, z_winding])

    r_axis = rng.uniform(0, 1.0, n_axis)
    z_axis = rng.uniform(-80, 80, n_axis)

    r_sparse = rng.uniform(0, 100, n_sparse)
    z_sparse = rng.uniform(-150, 150, n_sparse)

    r = np.concatenate([r_dense, r_boundary, r_axis, r_sparse])
    z = np.concatenate([z_dense, z_boundary, z_axis, z_sparse])

    return np.column_stack([r, z])


def generate_boundary_points_for_config(coil_params, n, rng):
    """Config-adaptive boundary points near this coil's geometry."""
    R_mean = (coil_params["inner_radius_mm"] + coil_params["outer_radius_mm"]) / 2
    L = coil_params["length_mm"]
    half_L = L / 2
    pts = []

    n_ends = n // 3
    z_end_pos = rng.normal(half_L, L * 0.1, n_ends // 2)
    z_end_neg = rng.normal(-half_L, L * 0.1, n_ends // 2)
    r_end = rng.uniform(0, 2 * R_mean, n_ends)
    pts.append(np.column_stack([r_end, np.concatenate([z_end_pos, z_end_neg])]))

    n_wind = n // 3
    r_wind = np.abs(rng.normal(R_mean, R_mean * 0.15, n_wind))
    z_wind = rng.uniform(-2 * half_L, 2 * half_L, n_wind)
    pts.append(np.column_stack([r_wind, z_wind]))

    n_ax = n - n_ends - n_wind
    r_ax = rng.uniform(0, 0.5, n_ax)
    z_ax = rng.uniform(-2 * half_L, 2 * half_L, n_ax)
    pts.append(np.column_stack([r_ax, z_ax]))

    return np.vstack(pts)


def generate_parameter_samples(n_configs=500, rng=None):
    """Grid corners + random configs (v4 recipe, no failure mining)."""
    if rng is None:
        rng = np.random.default_rng(123)

    configs = []

    Ns = [10, 30, 50, 80]
    R_means = [8.0, 12.0, 15.0, 20.0]
    Ls = [15.0, 30.0, 45.0, 60.0]
    for N_val in Ns:
        for R_val in R_means:
            for L_val in Ls:
                configs.append({
                    "current_A": 100.0,
                    "num_turns": N_val,
                    "inner_radius_mm": R_val - 3.0,
                    "outer_radius_mm": R_val + 3.0,
                    "length_mm": L_val,
                })
    for I in [10.0, 500.0, 1000.0]:
        for N_val in Ns:
            for R_val in R_means:
                for L_val in Ls:
                    configs.append({
                        "current_A": I,
                        "num_turns": N_val,
                        "inner_radius_mm": R_val - 3.0,
                        "outer_radius_mm": R_val + 3.0,
                        "length_mm": L_val,
                    })

    n_random = max(0, n_configs - len(configs))
    for _ in range(n_random):
        log_I = rng.uniform(np.log(0.5), np.log(4000.0))
        config = {
            "current_A": float(np.exp(log_I)),
            "num_turns": int(rng.integers(10, 81)),
            "inner_radius_mm": float(rng.uniform(6.0, 20.0)),
            "length_mm": float(rng.uniform(15.0, 60.0)),
        }
        config["outer_radius_mm"] = config["inner_radius_mm"] + float(rng.uniform(2.0, 8.0))
        configs.append(config)

    return configs


def main():
    print("=== PINN Training Data Generation (field-accuracy baseline) ===\n")

    points = generate_spatial_points()
    print(f"Generated {len(points)} spatial sample points")

    param_configs = generate_parameter_samples(n_configs=500)

    default_params = json.loads(CONFIG_PATH.read_text())
    for I in [1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 320.0, 500.0, 1000.0, 2000.0, 3000.0]:
        param_configs.append({
            "current_A": I,
            "num_turns": default_params["num_turns"],
            "inner_radius_mm": default_params["inner_radius_mm"],
            "outer_radius_mm": default_params["outer_radius_mm"],
            "length_mm": default_params["length_mm"],
        })
    print(f"Parameter configurations: {len(param_configs)}")

    rng = np.random.default_rng(99)
    general_per_config = 1500
    boundary_per_config = 500
    points_per_config = general_per_config + boundary_per_config
    total_points = len(param_configs) * points_per_config
    print(f"Total training samples: {total_points}")

    all_inputs = np.zeros((total_points, 6), dtype=np.float32)
    all_Br = np.zeros(total_points, dtype=np.float32)
    all_Bz = np.zeros(total_points, dtype=np.float32)

    idx = 0
    for ci, config in enumerate(param_configs):
        if ci % 50 == 0:
            print(f"  Processing config {ci+1}/{len(param_configs)}...")

        R_mean = (config["inner_radius_mm"] + config["outer_radius_mm"]) / 2

        subset_idx = rng.choice(len(points), size=general_per_config, replace=False)
        general_pts = points[subset_idx]
        bnd_pts = generate_boundary_points_for_config(config, boundary_per_config, rng)
        all_pts = np.vstack([general_pts, bnd_pts])

        Br, Bz = solenoid_field_batch(all_pts[:, 0], all_pts[:, 1], config)

        for i in range(points_per_config):
            all_inputs[idx] = [
                all_pts[i, 0], all_pts[i, 1],
                config["current_A"], config["num_turns"],
                R_mean, config["length_mm"],
            ]
            all_Br[idx] = Br[i]
            all_Bz[idx] = Bz[i]
            idx += 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "pinn_inputs.npy", all_inputs[:idx])
    np.save(DATA_DIR / "pinn_Br.npy", all_Br[:idx])
    np.save(DATA_DIR / "pinn_Bz.npy", all_Bz[:idx])
    # Save empty gradient files so train_pinn.py doesn't crash
    np.save(DATA_DIR / "pinn_dBz_dz.npy", np.zeros(idx, dtype=np.float32))
    np.save(DATA_DIR / "pinn_dBz_dr.npy", np.zeros(idx, dtype=np.float32))

    print(f"\nSaved {idx} training samples to {DATA_DIR}")
    print(f"  B_r: [{all_Br[:idx].min():.2e}, {all_Br[:idx].max():.2e}] T")
    print(f"  B_z: [{all_Bz[:idx].min():.2e}, {all_Bz[:idx].max():.2e}] T")


if __name__ == "__main__":
    main()
