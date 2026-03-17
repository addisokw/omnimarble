"""Evaluate PINN against analytical solution.

Generates error maps, line plots, and statistics.
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "pinn_checkpoint"
PLOTS_DIR = ROOT / "results" / "plots"
CONFIG_PATH = ROOT / "config" / "coil_params.json"

sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed")
    raise SystemExit(1)

from analytical_bfield import solenoid_field
from train_pinn import BFieldPINN


def load_model(device):
    """Load trained PINN model."""
    checkpoint = torch.load(MODEL_DIR / "pinn_best.pt", map_location=device, weights_only=True)
    model = BFieldPINN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from step {checkpoint['step']}, loss={checkpoint['loss']:.2e}")
    return model


def evaluate_grid(model, coil_params, device):
    """Compare PINN vs analytical on a regular grid."""
    import json

    r_grid = np.linspace(0.1, 60, 80)
    z_grid = np.linspace(-80, 80, 160)
    R, Z = np.meshgrid(r_grid, z_grid)

    # Analytical
    Br_exact = np.zeros_like(R)
    Bz_exact = np.zeros_like(R)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            br, bz = solenoid_field(R[i, j], Z[i, j], coil_params)
            Br_exact[i, j] = br
            Bz_exact[i, j] = bz

    # PINN
    flat_r = R.flatten()
    flat_z = Z.flatten()
    N_pts = len(flat_r)

    I = coil_params.get("current_A", coil_params.get("max_current_A", 10.0))
    N_turns = coil_params["num_turns"]
    R_mean = (coil_params["inner_radius_mm"] + coil_params["outer_radius_mm"]) / 2
    L = coil_params["length_mm"]

    inputs = np.column_stack([
        flat_r, flat_z,
        np.full(N_pts, I),
        np.full(N_pts, N_turns),
        np.full(N_pts, R_mean),
        np.full(N_pts, L),
    ]).astype(np.float32)

    with torch.no_grad():
        x_t = torch.tensor(inputs, device=device)
        out = model(x_t).cpu().numpy()

    output_scale = model.output_scale.cpu().numpy()
    Br_pinn = (out[:, 1] * output_scale[1]).reshape(R.shape)
    Bz_pinn = (out[:, 2] * output_scale[2]).reshape(R.shape)

    return R, Z, Br_exact, Bz_exact, Br_pinn, Bz_pinn


def plot_comparison(R, Z, Br_exact, Bz_exact, Br_pinn, Bz_pinn):
    """Generate comparison plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    Bmag_exact = np.sqrt(Br_exact**2 + Bz_exact**2)
    Bmag_pinn = np.sqrt(Br_pinn**2 + Bz_pinn**2)

    # Relative error
    Bmag_max = Bmag_exact.max()
    rel_err_Br = np.abs(Br_pinn - Br_exact) / (np.abs(Br_exact) + 1e-10 * Bmag_max)
    rel_err_Bz = np.abs(Bz_pinn - Bz_exact) / (np.abs(Bz_exact) + 1e-10 * Bmag_max)
    rel_err_mag = np.abs(Bmag_pinn - Bmag_exact) / (Bmag_exact + 1e-10 * Bmag_max)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Row 1: Bz comparison
    vmin, vmax = Bz_exact.min(), Bz_exact.max()
    axes[0, 0].contourf(Z, R, Bz_exact, levels=30, cmap="RdBu_r")
    axes[0, 0].set_title("B_z analytical")
    axes[0, 0].set_ylabel("r (mm)")

    axes[0, 1].contourf(Z, R, Bz_pinn, levels=30, cmap="RdBu_r")
    axes[0, 1].set_title("B_z PINN")

    c2 = axes[0, 2].contourf(Z, R, rel_err_Bz * 100, levels=np.linspace(0, 20, 21), cmap="hot_r")
    plt.colorbar(c2, ax=axes[0, 2], label="Relative error (%)")
    axes[0, 2].set_title("B_z relative error")

    # Row 2: |B| and line plots
    axes[1, 0].contourf(Z, R, np.log10(Bmag_exact + 1e-10), levels=30, cmap="viridis")
    axes[1, 0].set_title("|B| analytical (log)")
    axes[1, 0].set_xlabel("z (mm)")
    axes[1, 0].set_ylabel("r (mm)")

    axes[1, 1].contourf(Z, R, np.log10(Bmag_pinn + 1e-10), levels=30, cmap="viridis")
    axes[1, 1].set_title("|B| PINN (log)")
    axes[1, 1].set_xlabel("z (mm)")

    # Line plot: on-axis comparison
    z_idx = R.shape[1] // 2  # Not exactly right — use r=0 row
    # Find row closest to r = 0.1 (our minimum r)
    r_row = 0
    z_line = Z[r_row, :]
    axes[1, 2].plot(z_line, Bz_exact[r_row, :] * 1e3, "b-", label="Analytical", linewidth=2)
    axes[1, 2].plot(z_line, Bz_pinn[r_row, :] * 1e3, "r--", label="PINN", linewidth=2)
    axes[1, 2].set_xlabel("z (mm)")
    axes[1, 2].set_ylabel("B_z (mT)")
    axes[1, 2].set_title("B_z on axis (r≈0)")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pinn_evaluation.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'pinn_evaluation.png'}")

    # Statistics
    print(f"\nError statistics:")
    print(f"  B_r relative error: mean={rel_err_Br.mean()*100:.2f}%, "
          f"median={np.median(rel_err_Br)*100:.2f}%, max={rel_err_Br.max()*100:.2f}%")
    print(f"  B_z relative error: mean={rel_err_Bz.mean()*100:.2f}%, "
          f"median={np.median(rel_err_Bz)*100:.2f}%, max={rel_err_Bz.max()*100:.2f}%")
    print(f"  |B| relative error: mean={rel_err_mag.mean()*100:.2f}%, "
          f"median={np.median(rel_err_mag)*100:.2f}%, max={rel_err_mag.max()*100:.2f}%")

    under_5pct = (rel_err_mag < 0.05).mean() * 100
    print(f"  Points with |B| error < 5%: {under_5pct:.1f}%")


def benchmark_inference(model, device, n_points=10000):
    """Benchmark PINN inference speed."""
    import time

    inputs = torch.randn(n_points, 6, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    n_iters = 100
    for _ in range(n_iters):
        with torch.no_grad():
            model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    per_point = elapsed / (n_iters * n_points) * 1e6  # microseconds
    print(f"\nInference benchmark ({device}):")
    print(f"  {n_points} points × {n_iters} iters in {elapsed:.3f}s")
    print(f"  {per_point:.2f} us/point")


def main():
    import json

    print("=== PINN Evaluation ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    params = json.loads(CONFIG_PATH.read_text())
    params["current_A"] = params["max_current_A"]

    print("\nEvaluating on grid...")
    R, Z, Br_exact, Bz_exact, Br_pinn, Bz_pinn = evaluate_grid(model, params, device)
    plot_comparison(R, Z, Br_exact, Bz_exact, Br_pinn, Bz_pinn)

    benchmark_inference(model, device)


if __name__ == "__main__":
    main()
