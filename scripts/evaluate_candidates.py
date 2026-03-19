"""Evaluate multiple PINN checkpoint candidates for Pareto comparison.

Runs a compact version of the 5-level validation on each checkpoint
and outputs a comparison table.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "pinn_checkpoint"
CANDIDATES_DIR = MODEL_DIR / "candidates"
CONFIG_PATH = ROOT / "config" / "coil_params.json"

sys.path.insert(0, str(ROOT / "scripts"))

import torch
from scipy.stats import pearsonr

from analytical_bfield import MU_0_MM, ferromagnetic_force, solenoid_field, solenoid_field_gradient
from evaluate_pinn import (
    compute_error_stats,
    load_pinn_model,
    make_coil_params,
    pinn_predict_field,
    pinn_predict_field_with_grad,
)
from train_pinn import BFieldPINN


def evaluate_checkpoint(ckpt_path, config, device):
    """Run compact validation on a single checkpoint. Returns metrics dict."""
    # Load
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = BFieldPINN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    cn = bool(model.current_normalized.item())
    step = checkpoint.get("step", "?")

    N = config["num_turns"]
    R_mean = (config["inner_radius_mm"] + config["outer_radius_mm"]) / 2
    L_coil = config["length_mm"]

    results = {"step": step, "ckpt": ckpt_path.name}

    # --- Level 1: Field accuracy at I=100A (representative) ---
    r_pts = np.linspace(0.1, 60, 80).astype(np.float32)
    z_pts = np.linspace(-80, 80, 160).astype(np.float32)
    R_grid, Z_grid = np.meshgrid(r_pts, z_pts)
    flat_r = R_grid.flatten().astype(np.float32)
    flat_z = Z_grid.flatten().astype(np.float32)

    I_test = 100.0
    cp = {"current_A": I_test, "num_turns": N,
          "inner_radius_mm": config["inner_radius_mm"],
          "outer_radius_mm": config["outer_radius_mm"],
          "length_mm": L_coil}

    Bz_exact = np.array([solenoid_field(float(r), float(z), cp)[1]
                          for r, z in zip(flat_r, flat_z)])
    _, Bz_pinn = pinn_predict_field(model, flat_r, flat_z, I_test, N, R_mean, L_coil, cn, device)

    s = compute_error_stats(Bz_pinn, Bz_exact, "Bz")
    results["L1_mean"] = s["mean"]
    results["L1_p95"] = s["p95"]
    results["L1_p99"] = s["p99"]
    results["L1_max"] = s["max"]

    # --- Level 2: Gradient accuracy at RLC peak ---
    L_H = config.get("inductance_uH", 12.4) * 1e-6
    C_F = config["capacitance_uF"] * 1e-6
    R_total = config.get("resistance_ohm", 0.08)
    V0 = config["charge_voltage_V"]
    alpha = R_total / (2 * L_H)
    omega_0 = 1.0 / math.sqrt(L_H * C_F)
    omega_d = math.sqrt(omega_0**2 - alpha**2) if alpha < omega_0 else omega_0
    I_peak = V0 / (omega_d * L_H) if alpha < omega_0 else 320.0

    r2 = np.linspace(0.5, 30, 40).astype(np.float32)
    z2 = np.linspace(-40, 40, 80).astype(np.float32)
    R2, Z2 = np.meshgrid(r2, z2)
    fr2 = R2.flatten().astype(np.float32)
    fz2 = Z2.flatten().astype(np.float32)

    cp_peak = {"current_A": I_peak, "num_turns": N,
               "inner_radius_mm": config["inner_radius_mm"],
               "outer_radius_mm": config["outer_radius_mm"],
               "length_mm": L_coil}

    dBz_dz_exact = np.array([solenoid_field_gradient(float(r), float(z), cp_peak)[3]
                              for r, z in zip(fr2, fz2)])
    _, _, _, _, _, dBz_dz_pinn = pinn_predict_field_with_grad(
        model, fr2, fz2, I_peak, N, R_mean, L_coil, cn, device)

    sg = compute_error_stats(dBz_dz_pinn, dBz_dz_exact, "dBz_dz")
    results["L2_mean"] = sg["mean"]
    results["L2_p95"] = sg["p95"]
    results["L2_p99"] = sg["p99"]
    results["L2_max"] = sg["max"]

    # --- Level 3: Force accuracy at r=5mm ---
    z3 = np.linspace(-40, 40, 200).astype(np.float32)
    r3 = np.full_like(z3, 5.0)
    marble_radius = 5.0
    chi_eff = 3.0
    V_marble = (4 / 3) * math.pi * marble_radius**3

    Fz_ana = np.array([ferromagnetic_force(5.0, float(zv), marble_radius, cp_peak, chi_eff)[1]
                        for zv in z3])
    Br, Bz, dBr_dr, dBr_dz, dBz_dr, dBz_dz = pinn_predict_field_with_grad(
        model, r3, z3, I_peak, N, R_mean, L_coil, cn, device)
    prefactor = chi_eff * V_marble / MU_0_MM
    Fz_pinn = prefactor * (Br * dBz_dr + Bz * dBz_dz)

    peak_ana = np.max(np.abs(Fz_ana))
    peak_pinn = np.max(np.abs(Fz_pinn))
    results["L3_peak_err"] = abs(peak_pinn - peak_ana) / max(peak_ana, 1e-20)
    results["L3_pearson"] = float(pearsonr(Fz_ana, Fz_pinn)[0])

    # --- Level 4: Design space (64 configs) ---
    Ns = [10, 30, 50, 80]
    R_means = [8.0, 12.0, 15.0, 20.0]
    Ls = [15.0, 30.0, 45.0, 60.0]
    errs = []
    for Nv in Ns:
        for Rv in R_means:
            for Lv in Ls:
                cp4 = make_coil_params(100.0, Nv, Rv - 3.0, Rv + 3.0, Lv)
                bz_ex = solenoid_field(0.1, 0.0, cp4)[1]
                _, bz_pi = pinn_predict_field(
                    model, np.array([0.1], dtype=np.float32),
                    np.array([0.0], dtype=np.float32),
                    100.0, Nv, Rv, Lv, cn, device)
                errs.append(abs(float(bz_pi[0]) - bz_ex) / max(abs(bz_ex), 1e-20))
    errs = np.array(errs)
    results["L4_pct_under5"] = float((errs < 0.05).mean() * 100)
    results["L4_worst"] = float(errs.max())

    # --- Level 5: div(B) ---
    Br5, Bz5, dBr_dr5, _, _, dBz_dz5 = pinn_predict_field_with_grad(
        model, fr2, fz2, 100.0, N, R_mean, L_coil, cn, device)
    r_safe = np.maximum(fr2, 0.1)
    div_B = dBr_dr5 + Br5 / r_safe + dBz_dz5
    B_mag = np.sqrt(Br5**2 + Bz5**2)
    B_scale = np.max(B_mag)
    norm_div = np.abs(div_B) / (B_scale / 10.0 + 1e-20)
    results["L5_div"] = float(np.mean(norm_div))

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = json.loads(CONFIG_PATH.read_text())

    # Collect all candidates
    candidates = sorted(CANDIDATES_DIR.glob("pinn_best_step*.pt"),
                         key=lambda p: int(p.stem.split("step")[1]))
    # Also include the promoted best
    candidates.append(MODEL_DIR / "pinn_best.pt")

    print(f"Evaluating {len(candidates)} checkpoints...\n")

    all_results = []
    for ckpt in candidates:
        print(f"  {ckpt.name}...", end=" ", flush=True)
        r = evaluate_checkpoint(ckpt, config, device)
        all_results.append(r)
        print(f"L1={r['L1_mean']*100:.2f}%/{r['L1_max']*100:.1f}%  "
              f"L4={r['L4_pct_under5']:.0f}%  L5div={r['L5_div']:.4f}")

    # Print comparison table
    print("\n" + "=" * 120)
    print(f"{'Checkpoint':<28} | {'L1 mean':>7} {'L1 p99':>7} {'L1 max':>7} | "
          f"{'L2 p99':>7} | {'L3 pk%':>6} {'L3 r':>6} | "
          f"{'L4 %<5':>6} {'L4 wrst':>7} | {'L5 div':>7}")
    print("-" * 120)
    for r in all_results:
        l4_status = "PASS" if r["L4_pct_under5"] >= 95 else "FAIL"
        print(f"{r['ckpt']:<28} | "
              f"{r['L1_mean']*100:6.2f}% {r['L1_p99']*100:6.2f}% {r['L1_max']*100:6.1f}% | "
              f"{r['L2_p99']*100:6.2f}% | "
              f"{r['L3_peak_err']*100:5.1f}% {r['L3_pearson']:.4f} | "
              f"{r['L4_pct_under5']:5.1f}% {r['L4_worst']*100:6.2f}% | "
              f"{r['L5_div']:.5f}")
    print("=" * 120)

    # Save
    out_path = ROOT / "results" / "checkpoint_comparison.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
