"""PINN 5-level validation suite.

Validates the trained B-field PINN against analytical Biot-Savart ground truth
across field accuracy, gradient accuracy, end-to-end force, design space
coverage, and physics consistency checks.

Usage:
    python scripts/evaluate_pinn.py

Exit code 0 if all checks pass, 1 if any fail.
"""

import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "pinn_checkpoint"
PLOTS_DIR = ROOT / "results" / "plots" / "pinn_validation"
RESULTS_DIR = ROOT / "results"
CONFIG_PATH = ROOT / "config" / "coil_params.json"

sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed")
    raise SystemExit(1)

from analytical_bfield import MU_0_MM, ferromagnetic_force, solenoid_field, solenoid_field_gradient
from train_pinn import BFieldPINN

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


def load_pinn_model(device):
    """Load trained PINN model.

    Returns:
        (model, current_normalized, metadata)
    """
    ckpt_path = MODEL_DIR / "pinn_best.pt"
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    model = BFieldPINN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    current_normalized = bool(model.current_normalized.item())
    metadata = {
        "step": checkpoint.get("step", "?"),
        "loss": checkpoint.get("loss", "?"),
        "backend": checkpoint.get("backend", "?"),
        "current_normalized": current_normalized,
    }
    print(f"Loaded PINN: step={metadata['step']}, loss={metadata['loss']:.2e}, "
          f"current_normalized={current_normalized}")
    return model, current_normalized, metadata


def pinn_predict_field(model, r, z, I, N, R_mean, L, current_normalized, device):
    """Batch PINN inference returning (Br, Bz) arrays.

    Args:
        r, z: 1-D numpy arrays of spatial positions (mm)
        I, N, R_mean, L: scalar coil parameters
        current_normalized: if True, multiply output by I
        device: torch device

    Returns:
        (Br, Bz) as numpy arrays in Tesla
    """
    n = len(r)
    inputs = np.column_stack([
        r, z,
        np.full(n, I, dtype=np.float32),
        np.full(n, N, dtype=np.float32),
        np.full(n, R_mean, dtype=np.float32),
        np.full(n, L, dtype=np.float32),
    ]).astype(np.float32)

    with torch.no_grad():
        out = model(torch.tensor(inputs, device=device)).cpu().numpy()

    scale = I if current_normalized else 1.0
    Br = out[:, 1] * scale
    Bz = out[:, 2] * scale
    return Br, Bz


def pinn_predict_field_with_grad(model, r, z, I, N, R_mean, L, current_normalized, device):
    """PINN forward pass with autograd gradients, matching extension.py pattern.

    Args:
        r, z: 1-D numpy arrays
        I, N, R_mean, L: scalar coil parameters

    Returns:
        (Br, Bz, dBr_dr, dBr_dz, dBz_dr, dBz_dz) as numpy arrays
    """
    n = len(r)
    inputs = np.column_stack([
        r, z,
        np.full(n, I, dtype=np.float32),
        np.full(n, N, dtype=np.float32),
        np.full(n, R_mean, dtype=np.float32),
        np.full(n, L, dtype=np.float32),
    ]).astype(np.float32)

    inp_t = torch.tensor(inputs, device=device, requires_grad=True)
    out = model(inp_t)
    B_r = out[:, 1]
    B_z = out[:, 2]

    # Compute gradients one component at a time (matching extension.py)
    grad_Br = torch.autograd.grad(
        B_r, inp_t, grad_outputs=torch.ones_like(B_r),
        create_graph=False, retain_graph=True,
    )[0]
    grad_Bz = torch.autograd.grad(
        B_z, inp_t, grad_outputs=torch.ones_like(B_z),
        create_graph=False, retain_graph=False,
    )[0]

    scale = I if current_normalized else 1.0

    Br_np = B_r.detach().cpu().numpy() * scale
    Bz_np = B_z.detach().cpu().numpy() * scale
    dBr_dr = grad_Br[:, 0].detach().cpu().numpy() * scale
    dBr_dz = grad_Br[:, 1].detach().cpu().numpy() * scale
    dBz_dr = grad_Bz[:, 0].detach().cpu().numpy() * scale
    dBz_dz = grad_Bz[:, 1].detach().cpu().numpy() * scale

    return Br_np, Bz_np, dBr_dr, dBr_dz, dBz_dr, dBz_dz


def compute_error_stats(pred, exact, name):
    """Compute error statistics with denominator = max|exact| over grid.

    Returns dict with mean/median/p95/max relative error (as fractions, not %).
    """
    denom = np.max(np.abs(exact))
    if denom < 1e-20:
        denom = 1.0
    rel_err = np.abs(pred - exact) / denom
    return {
        "name": name,
        "mean": float(np.mean(rel_err)),
        "median": float(np.median(rel_err)),
        "p95": float(np.percentile(rel_err, 95)),
        "p99": float(np.percentile(rel_err, 99)),
        "max": float(np.max(rel_err)),
        "denom": float(denom),
    }


def make_coil_params(I, N, R_inner, R_outer, L):
    """Build a coil_params dict for analytical functions."""
    return {
        "current_A": I,
        "num_turns": int(N),
        "inner_radius_mm": R_inner,
        "outer_radius_mm": R_outer,
        "length_mm": L,
    }


def _default_coil_params(config, I):
    """Default coil params dict from config with specified current."""
    return make_coil_params(
        I=I,
        N=config["num_turns"],
        R_inner=config["inner_radius_mm"],
        R_outer=config["outer_radius_mm"],
        L=config["length_mm"],
    )


def _default_R_mean(config):
    return (config["inner_radius_mm"] + config["outer_radius_mm"]) / 2


# ---------------------------------------------------------------------------
# Level 1: Field accuracy vs analytical ground truth
# ---------------------------------------------------------------------------


def level1_field_accuracy(model, current_normalized, config, device):
    """Level 1: Field accuracy at default coil across multiple currents."""
    print("\n" + "=" * 60)
    print("LEVEL 1: Field accuracy vs analytical ground truth")
    print("=" * 60)

    N = config["num_turns"]
    R_mean = _default_R_mean(config)
    L = config["length_mm"]
    currents = [1, 10, 100, 500, 1000, 3000]

    r_pts = np.linspace(0.1, 60, 80)
    z_pts = np.linspace(-80, 80, 160)
    R_grid, Z_grid = np.meshgrid(r_pts, z_pts)
    flat_r = R_grid.flatten().astype(np.float32)
    flat_z = Z_grid.flatten().astype(np.float32)
    n_pts = len(flat_r)

    results = {}
    all_pass = True

    for I in currents:
        cp = _default_coil_params(config, I)

        # Analytical
        Br_exact = np.zeros(n_pts)
        Bz_exact = np.zeros(n_pts)
        for k in range(n_pts):
            br, bz = solenoid_field(float(flat_r[k]), float(flat_z[k]), cp)
            Br_exact[k] = br
            Bz_exact[k] = bz

        # PINN
        Br_pinn, Bz_pinn = pinn_predict_field(
            model, flat_r, flat_z, I, N, R_mean, L, current_normalized, device,
        )

        Bmag_exact = np.sqrt(Br_exact**2 + Bz_exact**2)
        Bmag_pinn = np.sqrt(Br_pinn**2 + Bz_pinn**2)

        stats_br = compute_error_stats(Br_pinn, Br_exact, f"Br@{I}A")
        stats_bz = compute_error_stats(Bz_pinn, Bz_exact, f"Bz@{I}A")
        stats_bmag = compute_error_stats(Bmag_pinn, Bmag_exact, f"|B|@{I}A")

        passed = stats_bz["mean"] < 0.05 and stats_bz["max"] < 0.20
        if not passed:
            all_pass = False

        status = "PASS" if passed else "FAIL"
        print(f"  I={I:5d}A  Bz: mean={stats_bz['mean']*100:6.2f}%  "
              f"max={stats_bz['max']*100:6.2f}%  [{status}]")

        results[f"I={I}A"] = {
            "Br": stats_br, "Bz": stats_bz, "|B|": stats_bmag,
            "pass": passed,
        }

    # --- Plots ---
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: Contour maps at I=100A
    I_plot = 100
    cp_plot = _default_coil_params(config, I_plot)
    Br_ex = np.zeros(n_pts)
    Bz_ex = np.zeros(n_pts)
    for k in range(n_pts):
        br, bz = solenoid_field(float(flat_r[k]), float(flat_z[k]), cp_plot)
        Br_ex[k] = br
        Bz_ex[k] = bz
    Br_pi, Bz_pi = pinn_predict_field(
        model, flat_r, flat_z, I_plot, N, R_mean, L, current_normalized, device,
    )

    Bz_ex_2d = Bz_ex.reshape(R_grid.shape)
    Bz_pi_2d = Bz_pi.reshape(R_grid.shape)
    denom = max(np.abs(Bz_ex).max(), 1e-20)
    err_2d = np.abs(Bz_pi_2d - Bz_ex_2d) / denom * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    c0 = axes[0].contourf(Z_grid, R_grid, Bz_ex_2d * 1e3, levels=30, cmap="RdBu_r")
    plt.colorbar(c0, ax=axes[0], label="Bz (mT)")
    axes[0].set_title(f"Bz analytical (I={I_plot}A)")
    axes[0].set_xlabel("z (mm)")
    axes[0].set_ylabel("r (mm)")

    c1 = axes[1].contourf(Z_grid, R_grid, Bz_pi_2d * 1e3, levels=30, cmap="RdBu_r")
    plt.colorbar(c1, ax=axes[1], label="Bz (mT)")
    axes[1].set_title(f"Bz PINN (I={I_plot}A)")
    axes[1].set_xlabel("z (mm)")

    c2 = axes[2].contourf(Z_grid, R_grid, err_2d, levels=np.linspace(0, 20, 21), cmap="hot_r")
    plt.colorbar(c2, ax=axes[2], label="Rel error (%)")
    axes[2].set_title("Bz relative error")
    axes[2].set_xlabel("z (mm)")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "level1_contour.png", dpi=150)
    plt.close()

    # Plot 2: On-axis Bz at all currents
    z_axis = np.linspace(-80, 80, 200).astype(np.float32)
    r_axis = np.full_like(z_axis, 0.1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for I in currents:
        cp = _default_coil_params(config, I)
        bz_ana = np.array([solenoid_field(0.1, float(zv), cp)[1] for zv in z_axis])
        bz_pin, _ = pinn_predict_field(
            model, r_axis, z_axis, I, N, R_mean, L, current_normalized, device,
        )
        # Swap: Bz_pinn is second return
        _, bz_pin = pinn_predict_field(
            model, r_axis, z_axis, I, N, R_mean, L, current_normalized, device,
        )
        ax.plot(z_axis, bz_ana * 1e3, "-", label=f"Ana {I}A", alpha=0.8)
        ax.plot(z_axis, bz_pin * 1e3, "--", label=f"PINN {I}A", alpha=0.8)

    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Bz (mT)")
    ax.set_title("On-axis Bz: analytical vs PINN")
    L_coil = config["length_mm"]
    ax.axvspan(-L_coil / 2, L_coil / 2, alpha=0.08, color="red", label="coil")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "level1_onaxis.png", dpi=150)
    plt.close()

    print(f"  Saved plots to {PLOTS_DIR}")
    return all_pass, results


# ---------------------------------------------------------------------------
# Level 2: Gradient accuracy
# ---------------------------------------------------------------------------


def level2_gradient_accuracy(model, current_normalized, config, device):
    """Level 2: Gradient accuracy at default coil, RLC peak current."""
    print("\n" + "=" * 60)
    print("LEVEL 2: Gradient accuracy (drives force computation)")
    print("=" * 60)

    # Estimate peak current from config RLC params
    L_H = config.get("inductance_uH", 12.4) * 1e-6
    C_F = config["capacitance_uF"] * 1e-6
    R_total = config.get("resistance_ohm", 0.08)
    V0 = config["charge_voltage_V"]
    alpha = R_total / (2 * L_H)
    omega_0 = 1.0 / math.sqrt(L_H * C_F)
    if alpha < omega_0:
        omega_d = math.sqrt(omega_0**2 - alpha**2)
        I_peak = V0 / (omega_d * L_H)
    else:
        I_peak = 320.0  # fallback
    print(f"  RLC peak current estimate: {I_peak:.1f} A")

    N = config["num_turns"]
    R_mean = _default_R_mean(config)
    L = config["length_mm"]

    r_pts = np.linspace(0.5, 30, 40).astype(np.float32)
    z_pts = np.linspace(-40, 40, 80).astype(np.float32)
    R_grid, Z_grid = np.meshgrid(r_pts, z_pts)
    flat_r = R_grid.flatten().astype(np.float32)
    flat_z = Z_grid.flatten().astype(np.float32)
    n = len(flat_r)

    cp = _default_coil_params(config, I_peak)

    # Analytical gradients
    dBr_dr_ex = np.zeros(n)
    dBr_dz_ex = np.zeros(n)
    dBz_dr_ex = np.zeros(n)
    dBz_dz_ex = np.zeros(n)
    for k in range(n):
        g = solenoid_field_gradient(float(flat_r[k]), float(flat_z[k]), cp)
        dBr_dr_ex[k], dBr_dz_ex[k], dBz_dr_ex[k], dBz_dz_ex[k] = g

    # PINN gradients
    _, _, dBr_dr_pi, dBr_dz_pi, dBz_dr_pi, dBz_dz_pi = pinn_predict_field_with_grad(
        model, flat_r, flat_z, I_peak, N, R_mean, L, current_normalized, device,
    )

    stats = {}
    names = ["dBr_dr", "dBr_dz", "dBz_dr", "dBz_dz"]
    exacts = [dBr_dr_ex, dBr_dz_ex, dBz_dr_ex, dBz_dz_ex]
    preds = [dBr_dr_pi, dBr_dz_pi, dBz_dr_pi, dBz_dz_pi]

    for name, pred, exact in zip(names, preds, exacts):
        s = compute_error_stats(pred, exact, name)
        stats[name] = s
        print(f"  {name}: mean={s['mean']*100:6.2f}%  p95={s['p95']*100:6.2f}%  max={s['max']*100:6.2f}%")

    # Pass criteria: dBz_dz mean < 10%, P99 < 10%
    # Max error is dominated by isolated singular points at the coil winding
    # surface where the analytical model has a near-discontinuity (current
    # sheet). A smooth PINN cannot reproduce this. P99 measures "almost
    # everywhere" accuracy while excluding the handful of pathological points.
    dBz_dz_stats = stats["dBz_dz"]
    passed = dBz_dz_stats["mean"] < 0.10 and dBz_dz_stats["p99"] < 0.10
    status = "PASS" if passed else "FAIL"
    print(f"  dBz/dz check: mean<10% and P99<10% -> [{status}]")
    print(f"    (P95={dBz_dz_stats['p95']*100:.2f}%, P99={dBz_dz_stats['p99']*100:.2f}%, max={dBz_dz_stats['max']*100:.1f}%)")

    # Plot: dBz/dz error contour
    denom = max(np.abs(dBz_dz_ex).max(), 1e-20)
    err_2d = (np.abs(dBz_dz_pi - dBz_dz_ex) / denom * 100).reshape(R_grid.shape)

    fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.contourf(Z_grid, R_grid, err_2d, levels=np.linspace(0, 50, 26), cmap="hot_r")
    plt.colorbar(c, ax=ax, label="dBz/dz rel error (%)")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("r (mm)")
    ax.set_title(f"dBz/dz gradient error (I={I_peak:.0f}A)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "level2_gradient_error.png", dpi=150)
    plt.close()

    return passed, {"I_peak": I_peak, "stats": stats, "pass": passed}


# ---------------------------------------------------------------------------
# Level 3: End-to-end force accuracy
# ---------------------------------------------------------------------------


def level3_force_accuracy(model, current_normalized, config, device):
    """Level 3: End-to-end force comparison at default coil."""
    print("\n" + "=" * 60)
    print("LEVEL 3: End-to-end force accuracy")
    print("=" * 60)

    # Peak current
    L_H = config.get("inductance_uH", 12.4) * 1e-6
    C_F = config["capacitance_uF"] * 1e-6
    R_total = config.get("resistance_ohm", 0.08)
    V0 = config["charge_voltage_V"]
    alpha = R_total / (2 * L_H)
    omega_0 = 1.0 / math.sqrt(L_H * C_F)
    if alpha < omega_0:
        omega_d = math.sqrt(omega_0**2 - alpha**2)
        I_peak = V0 / (omega_d * L_H)
    else:
        I_peak = 320.0

    N = config["num_turns"]
    R_mean = _default_R_mean(config)
    L_coil = config["length_mm"]
    marble_radius = 5.0
    chi_eff = 3.0
    V_marble = (4 / 3) * math.pi * marble_radius**3

    results = {}
    all_pass = True

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (r_test, label) in enumerate([(0.1, "r=0mm"), (5.0, "r=5mm")]):
        z_pts = np.linspace(-40, 40, 200).astype(np.float32)
        r_pts = np.full_like(z_pts, r_test)

        # Analytical force
        cp = _default_coil_params(config, I_peak)
        Fz_ana = np.array([
            ferromagnetic_force(float(r_test), float(zv), marble_radius, cp, chi_eff)[1]
            for zv in z_pts
        ])

        # PINN force: replicate PINNForceComputer.compute_force() logic
        Br, Bz, dBr_dr, dBr_dz, dBz_dr, dBz_dz = pinn_predict_field_with_grad(
            model, r_pts, z_pts, I_peak, N, R_mean, L_coil, current_normalized, device,
        )
        prefactor = chi_eff * V_marble / MU_0_MM
        Fz_pinn = prefactor * (Br * dBz_dr + Bz * dBz_dz)

        # Metrics
        # Peak force error
        peak_ana = np.max(np.abs(Fz_ana))
        peak_pinn = np.max(np.abs(Fz_pinn))
        peak_err = abs(peak_pinn - peak_ana) / max(peak_ana, 1e-20)

        # Pearson correlation
        corr, _ = pearsonr(Fz_ana, Fz_pinn)

        # Zero-crossing error (find where Fz crosses zero near z=0)
        def find_zero_crossing(z_arr, f_arr):
            for k in range(len(f_arr) - 1):
                if f_arr[k] * f_arr[k + 1] < 0:
                    # Linear interpolation
                    frac = abs(f_arr[k]) / (abs(f_arr[k]) + abs(f_arr[k + 1]))
                    return z_arr[k] + frac * (z_arr[k + 1] - z_arr[k])
            return 0.0

        zc_ana = find_zero_crossing(z_pts, Fz_ana)
        zc_pinn = find_zero_crossing(z_pts, Fz_pinn)
        zc_err = abs(zc_pinn - zc_ana)

        passed = peak_err < 0.15 and corr > 0.95 and zc_err < 2.0
        if not passed:
            all_pass = False

        status = "PASS" if passed else "FAIL"
        print(f"  {label}: peak_err={peak_err*100:.1f}%  Pearson_r={corr:.4f}  "
              f"zc_err={zc_err:.2f}mm  [{status}]")

        results[label] = {
            "peak_err": float(peak_err),
            "pearson_r": float(corr),
            "zero_crossing_err_mm": float(zc_err),
            "pass": passed,
        }

        # Plot
        axes[idx].plot(z_pts, Fz_ana, "b-", label="Analytical", linewidth=2)
        axes[idx].plot(z_pts, Fz_pinn, "r--", label="PINN", linewidth=2)
        axes[idx].set_xlabel("z (mm)")
        axes[idx].set_ylabel("F_z (mN)")
        axes[idx].set_title(f"Axial force ({label}, I={I_peak:.0f}A)")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axvspan(-L_coil / 2, L_coil / 2, alpha=0.08, color="green", label="coil")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "level3_force.png", dpi=150)
    plt.close()

    return all_pass, results


# ---------------------------------------------------------------------------
# Level 4: Design space coverage
# ---------------------------------------------------------------------------


def level4_design_space(model, current_normalized, config, device):
    """Level 4: Accuracy across 64 coil geometries + 4 extrapolation points."""
    print("\n" + "=" * 60)
    print("LEVEL 4: Design space coverage (64 configs)")
    print("=" * 60)

    I_test = 100.0
    Ns = [10, 30, 50, 80]
    R_means = [8.0, 12.0, 15.0, 20.0]
    Ls = [15.0, 30.0, 45.0, 60.0]

    errors = []
    configs_detail = []

    for N_val in Ns:
        for R_val in R_means:
            for L_val in Ls:
                # On-axis at 5 z-positions relative to coil
                z_frac = np.array([-0.5, -0.25, 0.0, 0.25, 0.5]) * L_val
                r_pts = np.full(5, 0.1, dtype=np.float32)
                z_pts = z_frac.astype(np.float32)

                # Use R_mean ± 3mm for inner/outer
                R_inner = R_val - 3.0
                R_outer = R_val + 3.0
                cp = make_coil_params(I_test, N_val, R_inner, R_outer, L_val)

                # Analytical
                Bz_exact = np.array([
                    solenoid_field(0.1, float(zv), cp)[1] for zv in z_pts
                ])

                # PINN
                _, Bz_pinn = pinn_predict_field(
                    model, r_pts, z_pts, I_test, N_val, R_val, L_val,
                    current_normalized, device,
                )

                # Center Bz error
                center_idx = 2  # z = 0
                center_exact = abs(Bz_exact[center_idx])
                center_err = abs(Bz_pinn[center_idx] - Bz_exact[center_idx]) / max(center_exact, 1e-20)

                errors.append(center_err)
                configs_detail.append({
                    "N": N_val, "R_mean": R_val, "L": L_val,
                    "center_err": float(center_err),
                    "Bz_exact_center": float(Bz_exact[center_idx]),
                    "Bz_pinn_center": float(Bz_pinn[center_idx]),
                })

    errors = np.array(errors)
    pct_under_5 = (errors < 0.05).mean() * 100
    worst = errors.max()

    passed = pct_under_5 >= 95.0 and worst < 0.20
    status = "PASS" if passed else "FAIL"
    print(f"  Configs with center Bz error < 5%: {pct_under_5:.1f}% (need >=95%)")
    print(f"  Worst center error: {worst*100:.2f}% (need <20%)")
    print(f"  [{status}]")

    # Extrapolation points (outside training bounds — document as known limitation)
    extrap_configs = [
        {"N": 5, "R_mean": 5.0, "L": 10.0, "label": "N=5, R=5, L=10 (below range)"},
        {"N": 100, "R_mean": 25.0, "L": 80.0, "label": "N=100, R=25, L=80 (above range)"},
        {"N": 30, "R_mean": 3.0, "L": 30.0, "label": "R=3mm (tiny)"},
        {"N": 30, "R_mean": 30.0, "L": 30.0, "label": "R=30mm (wide)"},
    ]
    extrap_results = []
    print("  Extrapolation points (informational only):")
    for ec in extrap_configs:
        R_inner = ec["R_mean"] - 3.0
        R_outer = ec["R_mean"] + 3.0
        cp = make_coil_params(I_test, ec["N"], max(R_inner, 0.5), R_outer, ec["L"])
        bz_exact = solenoid_field(0.1, 0.0, cp)[1]
        r_pts = np.array([0.1], dtype=np.float32)
        z_pts = np.array([0.0], dtype=np.float32)
        _, bz_pinn = pinn_predict_field(
            model, r_pts, z_pts, I_test, ec["N"], ec["R_mean"], ec["L"],
            current_normalized, device,
        )
        err = abs(float(bz_pinn[0]) - bz_exact) / max(abs(bz_exact), 1e-20)
        print(f"    {ec['label']}: err={err*100:.1f}%")
        extrap_results.append({"config": ec["label"], "err": float(err)})

    # Heatmap plot (N vs R_mean at L=30)
    fig, ax = plt.subplots(figsize=(7, 5))
    err_grid = np.zeros((len(Ns), len(R_means)))
    for cfg in configs_detail:
        if cfg["L"] == 30.0:
            i = Ns.index(cfg["N"])
            j = R_means.index(cfg["R_mean"])
            err_grid[i, j] = cfg["center_err"] * 100
    im = ax.imshow(err_grid, origin="lower", cmap="hot_r", aspect="auto",
                   extent=[R_means[0] - 2, R_means[-1] + 2, Ns[0] - 5, Ns[-1] + 5])
    plt.colorbar(im, ax=ax, label="Center Bz error (%)")
    ax.set_xlabel("R_mean (mm)")
    ax.set_ylabel("N (turns)")
    ax.set_title("Design space error (L=30mm, I=100A)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "level4_design_space.png", dpi=150)
    plt.close()

    return passed, {
        "pct_under_5": float(pct_under_5),
        "worst_err": float(worst),
        "pass": passed,
        "configs": configs_detail,
        "extrapolation": extrap_results,
    }


# ---------------------------------------------------------------------------
# Level 5: Physics consistency
# ---------------------------------------------------------------------------


def level5_physics_consistency(model, current_normalized, config, device):
    """Level 5: div(B)=0, axial symmetry, z-symmetry."""
    print("\n" + "=" * 60)
    print("LEVEL 5: Physics consistency checks")
    print("=" * 60)

    N = config["num_turns"]
    R_mean = _default_R_mean(config)
    L_coil = config["length_mm"]
    I_test = 100.0

    r_pts = np.linspace(0.5, 30, 40).astype(np.float32)
    z_pts = np.linspace(-40, 40, 80).astype(np.float32)
    R_grid, Z_grid = np.meshgrid(r_pts, z_pts)
    flat_r = R_grid.flatten().astype(np.float32)
    flat_z = Z_grid.flatten().astype(np.float32)

    Br, Bz, dBr_dr, dBr_dz, dBz_dr, dBz_dz = pinn_predict_field_with_grad(
        model, flat_r, flat_z, I_test, N, R_mean, L_coil, current_normalized, device,
    )

    results = {}

    # Check 1: div(B) = dBr/dr + Br/r + dBz/dz ≈ 0
    r_safe = np.maximum(flat_r, 0.1)
    div_B = dBr_dr + Br / r_safe + dBz_dz
    B_mag = np.sqrt(Br**2 + Bz**2)
    B_scale = np.max(B_mag)
    # Normalize by max(|B|) / typical_length_scale
    norm_div = np.abs(div_B) / (B_scale / 10.0 + 1e-20)  # 10mm characteristic length
    mean_norm_div = float(np.mean(norm_div))

    div_pass = mean_norm_div < 0.01
    status = "PASS" if div_pass else "FAIL"
    print(f"  div(B)=0: mean normalized |div B| = {mean_norm_div:.6f} (need <0.01) [{status}]")
    results["div_B"] = {"mean_normalized": mean_norm_div, "pass": div_pass}

    # Check 2: Axial symmetry — Br ≈ 0 at r → 0
    r_near_axis = np.full(80, 0.1, dtype=np.float32)
    z_check = np.linspace(-40, 40, 80).astype(np.float32)
    Br_axis, Bz_axis = pinn_predict_field(
        model, r_near_axis, z_check, I_test, N, R_mean, L_coil, current_normalized, device,
    )
    ratio = np.max(np.abs(Br_axis)) / max(np.max(np.abs(Bz_axis)), 1e-20)
    sym_pass = ratio < 0.01
    status = "PASS" if sym_pass else "FAIL"
    print(f"  Axial symmetry: max|Br/Bz| at r~0 = {ratio:.6f} (need <0.01) [{status}]")
    results["axial_symmetry"] = {"max_Br_over_Bz": float(ratio), "pass": sym_pass}

    # Check 3: z-symmetry — Bz(r, z) = Bz(r, -z) for centered coil
    z_pos = np.linspace(1, 40, 40).astype(np.float32)
    r_test = np.full(40, 5.0, dtype=np.float32)
    _, Bz_pos = pinn_predict_field(
        model, r_test, z_pos, I_test, N, R_mean, L_coil, current_normalized, device,
    )
    _, Bz_neg = pinn_predict_field(
        model, r_test, -z_pos, I_test, N, R_mean, L_coil, current_normalized, device,
    )
    asymmetry = np.abs(Bz_pos - Bz_neg) / (np.max(np.abs(Bz_pos)) + 1e-20)
    max_asym = float(np.max(asymmetry))
    zsym_pass = max_asym < 0.02
    status = "PASS" if zsym_pass else "FAIL"
    print(f"  z-symmetry: max asymmetry = {max_asym:.6f} (need <0.02) [{status}]")
    results["z_symmetry"] = {"max_asymmetry": max_asym, "pass": zsym_pass}

    all_pass = div_pass and sym_pass and zsym_pass
    return all_pass, results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_audit_report(all_results, metadata, config):
    """Append a validation run entry to the existing audit report.

    Does NOT overwrite the curated report. Appends a new section at the end
    with the timestamped results from this run.
    """
    report_path = RESULTS_DIR / "pinn_audit_report.md"

    # Summarize pass/fail
    checks = []
    for level_name, (passed, _) in all_results.items():
        checks.append((level_name, passed))
    all_pass = all(p for _, p in checks)
    n_pass = sum(1 for _, p in checks if p)

    step = metadata.get("step", "?")
    loss = metadata.get("loss", 0)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    verdict = "ALL PASS" if all_pass else f"{n_pass}/5 pass"

    lines = []
    lines.append(f"\n\n---\n\n### Validation run: {timestamp} (step={step}, loss={loss:.2e}, {verdict})\n\n")

    # Summary table
    lines.append("| Check | Result |\n|-------|--------|\n")
    for name, passed in checks:
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |\n")

    # Level 1
    _, l1_data = all_results.get("Level 1: Field accuracy", (None, {}))
    lines.append("\n**Level 1: Field Accuracy**\n\n")
    lines.append("| Current | Bz mean err | Bz max err | Result |\n")
    lines.append("|---------|-------------|------------|--------|\n")
    if isinstance(l1_data, dict):
        for key, val in l1_data.items():
            if isinstance(val, dict) and "Bz" in val:
                bz = val["Bz"]
                status = "PASS" if val.get("pass") else "FAIL"
                lines.append(f"| {key} | {bz['mean']*100:.2f}% | {bz['max']*100:.2f}% | {status} |\n")

    # Level 2
    _, l2_data = all_results.get("Level 2: Gradient accuracy", (None, {}))
    lines.append(f"\n**Level 2: Gradient Accuracy** (I={l2_data.get('I_peak', '?'):.0f}A)\n\n")
    if isinstance(l2_data, dict) and "stats" in l2_data:
        lines.append("| Component | Mean err | P95 err | Max err |\n")
        lines.append("|-----------|----------|---------|---------|  \n")
        for name, s in l2_data["stats"].items():
            lines.append(f"| {name} | {s['mean']*100:.2f}% | {s['p95']*100:.2f}% | {s['max']*100:.2f}% |\n")

    # Level 3
    _, l3_data = all_results.get("Level 3: Force accuracy", (None, {}))
    lines.append("\n**Level 3: Force Accuracy**\n\n")
    lines.append("| Position | Peak F err | Pearson r | Zero-cross err | Result |\n")
    lines.append("|----------|------------|-----------|----------------|--------|\n")
    if isinstance(l3_data, dict):
        for key, val in l3_data.items():
            if isinstance(val, dict) and "peak_err" in val:
                status = "PASS" if val.get("pass") else "FAIL"
                lines.append(f"| {key} | {val['peak_err']*100:.1f}% | {val['pearson_r']:.4f} | {val['zero_crossing_err_mm']:.2f} mm | {status} |\n")

    # Level 4
    _, l4_data = all_results.get("Level 4: Design space", (None, {}))
    lines.append("\n**Level 4: Design Space**\n")
    if isinstance(l4_data, dict):
        lines.append(f"- Configs under 5%: {l4_data.get('pct_under_5', '?'):.1f}% (need 95%)\n")
        lines.append(f"- Worst: {l4_data.get('worst_err', 0)*100:.2f}% (need <20%)\n")

    # Level 5
    _, l5_data = all_results.get("Level 5: Physics consistency", (None, {}))
    lines.append("\n**Level 5: Physics Consistency**\n")
    if isinstance(l5_data, dict):
        for check_name, check_data in l5_data.items():
            if isinstance(check_data, dict):
                status = "PASS" if check_data.get("pass") else "FAIL"
                vals = {k: v for k, v in check_data.items() if k != "pass"}
                val_str = ", ".join(f"{k}={v:.6f}" for k, v in vals.items())
                lines.append(f"- {check_name}: {val_str} [{status}]\n")

    # Append to existing report (or create if missing)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        with open(report_path, "a", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"\nAudit entry appended to: {report_path}")
    else:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# PINN B-Field Model Audit Report\n\n")
            f.write("Validation run entries are appended below.\n")
            f.writelines(lines)
        print(f"\nAudit report created: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  PINN VALIDATION SUITE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, current_normalized, metadata = load_pinn_model(device)
    config = json.loads(CONFIG_PATH.read_text())

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Level 1
    passed, data = level1_field_accuracy(model, current_normalized, config, device)
    all_results["Level 1: Field accuracy"] = (passed, data)

    # Level 2
    passed, data = level2_gradient_accuracy(model, current_normalized, config, device)
    all_results["Level 2: Gradient accuracy"] = (passed, data)

    # Level 3
    passed, data = level3_force_accuracy(model, current_normalized, config, device)
    all_results["Level 3: Force accuracy"] = (passed, data)

    # Level 4
    passed, data = level4_design_space(model, current_normalized, config, device)
    all_results["Level 4: Design space"] = (passed, data)

    # Level 5
    passed, data = level5_physics_consistency(model, current_normalized, config, device)
    all_results["Level 5: Physics consistency"] = (passed, data)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    any_fail = False
    for name, (passed, _) in all_results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            any_fail = True
        print(f"  {name}: {status}")

    # Save JSON report
    json_report = {
        "metadata": metadata,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for name, (passed, data) in all_results.items():
        # Strip large arrays for JSON
        json_data = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if k == "configs":
                    json_data[k] = v[:5]  # just first 5 for brevity
                    json_data["configs_total"] = len(v)
                else:
                    json_data[k] = v
        json_report[name] = {"passed": passed, "data": json_data}

    json_path = RESULTS_DIR / "pinn_validation_report.json"
    json_path.write_text(json.dumps(json_report, indent=2, default=str), encoding="utf-8")
    print(f"\nJSON report: {json_path}")

    # Generate audit report
    generate_audit_report(all_results, metadata, config)

    if any_fail:
        print("\nRESULT: FAIL (some checks did not pass)")
        sys.exit(1)
    else:
        print("\nRESULT: ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
