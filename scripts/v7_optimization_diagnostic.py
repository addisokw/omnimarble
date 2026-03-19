"""V7 PINN optimization diagnostic: ranking stability + trajectory error analysis.

Tests whether v7's field errors affect design optimization by comparing
candidate rankings between v7 PINN and analytical reference.

Phase 1: Ranking stability over (geometry, drive) candidate pairs
Phase 2: Trajectory-overlaid error heatmaps for disagreement cases

A candidate is a (coil geometry, drive settings) pair scored by exit velocity
from a simplified 1D marble launch simulation.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, kendalltau

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "coil_params.json"
PLOTS_DIR = ROOT / "results" / "plots" / "v7_diagnostic"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(ROOT / "scripts"))

import torch

from analytical_bfield import (
    MU_0_MM,
    solenoid_field,
    solenoid_field_batch,
    solenoid_field_gradient,
)
from evaluate_pinn import load_pinn_model, pinn_predict_field, pinn_predict_field_with_grad
from train_pinn import BFieldPINN


# ---------------------------------------------------------------------------
# 1D marble launch simulation
# ---------------------------------------------------------------------------

def rlc_current(t, V0, C_uF, L_uH, R_ohm):
    """Compute underdamped RLC current at time t (with flyback diode)."""
    C = C_uF * 1e-6
    L = L_uH * 1e-6
    R = R_ohm

    alpha = R / (2 * L)
    omega_0_sq = 1.0 / (L * C)

    if alpha**2 >= omega_0_sq:
        # Overdamped / critically damped — skip
        return 0.0

    omega_d = math.sqrt(omega_0_sq - alpha**2)
    I = (V0 / (omega_d * L)) * math.exp(-alpha * t) * math.sin(omega_d * t)
    return max(I, 0.0)  # flyback diode


def rlc_peak_current(V0, C_uF, L_uH, R_ohm):
    """Compute peak current for underdamped RLC."""
    C = C_uF * 1e-6
    L = L_uH * 1e-6
    R = R_ohm
    alpha = R / (2 * L)
    omega_0_sq = 1.0 / (L * C)
    if alpha**2 >= omega_0_sq:
        return 0.0
    omega_d = math.sqrt(omega_0_sq - alpha**2)
    return V0 / (omega_d * L)


def simulate_1d_launch(coil_params, drive_params, field_fn, dt=1e-5, t_max=0.005):
    """Simplified 1D marble launch simulation along coil axis (r=0).

    Args:
        coil_params: dict with coil geometry
        drive_params: dict with V0, C_uF, L_uH, R_ohm, cutoff_z
        field_fn: callable(r, z, I, coil_params) -> (Bz, dBz_dz)
        dt: timestep (s)
        t_max: max simulation time (s)

    Returns:
        dict with trajectory arrays and final exit velocity
    """
    marble_radius = 5.0  # mm
    marble_mass_kg = (4 / 3) * math.pi * marble_radius**3 * 7.8e-3 * 1e-3
    chi_eff = 3.0
    V_marble = (4 / 3) * math.pi * marble_radius**3
    prefactor = chi_eff * V_marble / MU_0_MM  # mN per (T * T/mm)

    V0 = drive_params["V0"]
    C_uF = drive_params["C_uF"]
    L_uH = drive_params["L_uH"]
    R_ohm = drive_params["R_ohm"]
    cutoff_z = drive_params.get("cutoff_z", 5.0)  # mm past center

    # Initial conditions: marble at entry, moving forward
    z = -20.0  # mm, entry position
    v = 200.0  # mm/s, approach velocity
    r = 0.1    # mm, slightly off-axis to avoid r=0 singularity

    t_arr, z_arr, v_arr, F_arr, I_arr = [], [], [], [], []
    pulse_active = True

    for step in range(int(t_max / dt)):
        t = step * dt

        # Current
        if pulse_active:
            I = rlc_current(t, V0, C_uF, L_uH, R_ohm)
            if z > cutoff_z:
                pulse_active = False
                I = 0.0
        else:
            I = 0.0

        if abs(I) < 0.01:
            if t > 0.001:  # allow initial ramp
                # Record and continue coasting
                t_arr.append(t)
                z_arr.append(z)
                v_arr.append(v)
                F_arr.append(0.0)
                I_arr.append(0.0)
                # v is mm/s, dt is s, so dz = v * dt is in mm
                z += v * dt
                if z > 80:
                    break
                continue

        # Field and gradient from the provided function
        Bz, dBz_dz = field_fn(r, z, I, coil_params)

        # Axial force: F_z = prefactor * Bz * dBz/dz (mN)
        F_z = prefactor * Bz * dBz_dz

        # Acceleration: F(mN) -> F(N) = F*1e-3, a = F/m (m/s^2), convert to mm/s^2
        a = (F_z * 1e-3 / marble_mass_kg) * 1e3  # mm/s^2

        t_arr.append(t)
        z_arr.append(z)
        v_arr.append(v)
        F_arr.append(F_z)
        I_arr.append(I)

        # Euler integration
        v += a * dt
        z += v * dt

        if z > 80:
            break

    return {
        "t": np.array(t_arr),
        "z": np.array(z_arr),
        "v": np.array(v_arr),
        "F": np.array(F_arr),
        "I": np.array(I_arr),
        "exit_v": v,
        "peak_F": max(F_arr) if F_arr else 0.0,
    }


# ---------------------------------------------------------------------------
# Field functions (analytical and PINN)
# ---------------------------------------------------------------------------

def analytical_field_fn(r, z, I, coil_params):
    """Return (Bz, dBz_dz) from analytical model."""
    cp = dict(coil_params)
    cp["current_A"] = I
    _, Bz = solenoid_field(r, z, cp)
    _, _, _, dBz_dz = solenoid_field_gradient(r, z, cp)
    return Bz, dBz_dz


def make_pinn_field_fn(model, current_normalized, device):
    """Return a field function using the PINN model."""
    def pinn_field_fn(r, z, I, coil_params):
        N = coil_params["num_turns"]
        R_mean = (coil_params["inner_radius_mm"] + coil_params["outer_radius_mm"]) / 2
        L = coil_params["length_mm"]
        r_arr = np.array([r], dtype=np.float32)
        z_arr = np.array([z], dtype=np.float32)
        _, Bz_arr, _, _, _, dBz_dz_arr = pinn_predict_field_with_grad(
            model, r_arr, z_arr, I, N, R_mean, L, current_normalized, device,
        )
        return float(Bz_arr[0]), float(dBz_dz_arr[0])
    return pinn_field_fn


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(n=60):
    """Generate (geometry, drive) candidate pairs spanning the design space.

    Mix of structured grid (small, to anchor known regimes) and random
    candidates (majority, to exercise the full parameter space including
    varied capacitance and resistance regimes).
    """
    rng = np.random.default_rng(777)

    candidates = []

    # Small structured grid: 18 candidates anchoring key regimes
    for N_val in [15, 30, 50]:
        for R_val in [10, 15]:
            for V0 in [50, 200, 400]:
                candidates.append({
                    "id": len(candidates),
                    "N": N_val,
                    "R_mean": R_val,
                    "L": 30.0,
                    "V0": V0,
                    "C_uF": 470,
                })

    # Random candidates: varied geometry, voltage, capacitance
    while len(candidates) < n:
        candidates.append({
            "id": len(candidates),
            "N": int(rng.choice([10, 15, 20, 30, 40, 50, 60, 80])),
            "R_mean": float(rng.uniform(8, 20)),
            "L": float(rng.uniform(15, 60)),
            "V0": float(rng.uniform(30, 400)),
            "C_uF": float(rng.choice([100, 220, 470, 1000])),
        })

    return candidates[:n]


def candidate_to_params(cand):
    """Convert candidate dict to (coil_params, drive_params)."""
    R_mean = cand["R_mean"]
    coil = {
        "num_turns": int(cand["N"]),
        "inner_radius_mm": R_mean - 3.0,
        "outer_radius_mm": R_mean + 3.0,
        "length_mm": cand["L"],
    }
    # Estimate inductance via Wheeler's formula
    a_in = R_mean / 25.4
    l_in = cand["L"] / 25.4
    c_in = 6.0 / 25.4  # ~6mm winding depth
    denom = 6 * a_in + 9 * l_in + 10 * c_in
    L_uH = 0.8 * a_in**2 * cand["N"]**2 / denom if denom > 0 else 10.0

    # Resistance estimate
    wire_length = cand["N"] * 2 * math.pi * R_mean  # mm
    wire_cross = math.pi * 0.4**2  # 0.8mm wire
    R_dc = 1.72e-5 * wire_length / wire_cross
    R_ohm = R_dc + 0.03  # + ESR + wiring

    drive = {
        "V0": cand["V0"],
        "C_uF": cand["C_uF"],
        "L_uH": L_uH,
        "R_ohm": R_ohm,
        "cutoff_z": 5.0,
    }
    return coil, drive


# ---------------------------------------------------------------------------
# Phase 1: Ranking stability
# ---------------------------------------------------------------------------

def phase1_ranking(model, current_normalized, device, n_candidates=54):
    """Compare candidate rankings between analytical and v7 PINN."""
    print("\n" + "=" * 60)
    print("PHASE 1: Ranking stability over candidate designs")
    print("=" * 60)

    candidates = generate_candidates(n_candidates)
    pinn_fn = make_pinn_field_fn(model, current_normalized, device)

    results = []
    for i, cand in enumerate(candidates):
        coil, drive = candidate_to_params(cand)

        # Check if underdamped (skip overdamped)
        I_peak = rlc_peak_current(drive["V0"], drive["C_uF"], drive["L_uH"], drive["R_ohm"])
        if I_peak < 1.0:
            continue

        if (i + 1) % 10 == 0:
            print(f"  Evaluating candidate {i+1}/{len(candidates)}...")

        sim_ref = simulate_1d_launch(coil, drive, analytical_field_fn)
        sim_v7 = simulate_1d_launch(coil, drive, pinn_fn)

        results.append({
            **cand,
            "I_peak": I_peak,
            "L_uH": drive["L_uH"],
            "R_ohm": drive["R_ohm"],
            "ref_exit_v": sim_ref["exit_v"],
            "v7_exit_v": sim_v7["exit_v"],
            "ref_peak_F": sim_ref["peak_F"],
            "v7_peak_F": sim_v7["peak_F"],
        })

    # Compute rankings
    ref_scores = [r["ref_exit_v"] for r in results]
    v7_scores = [r["v7_exit_v"] for r in results]

    ref_ranks = np.argsort(np.argsort(-np.array(ref_scores))) + 1
    v7_ranks = np.argsort(np.argsort(-np.array(v7_scores))) + 1

    for i, r in enumerate(results):
        r["ref_rank"] = int(ref_ranks[i])
        r["v7_rank"] = int(v7_ranks[i])
        r["rank_diff"] = int(v7_ranks[i] - ref_ranks[i])

    # Statistics
    spearman_r, spearman_p = spearmanr(ref_scores, v7_scores)
    kendall_t, kendall_p = kendalltau(ref_scores, v7_scores)

    # Top-k overlap
    n_total = len(results)
    for k in [5, 10]:
        ref_topk = set(i for i, r in enumerate(results) if r["ref_rank"] <= k)
        v7_topk = set(i for i, r in enumerate(results) if r["v7_rank"] <= k)
        overlap = len(ref_topk & v7_topk)
        print(f"  Top-{k} overlap: {overlap}/{k}")

    print(f"  Spearman rank correlation: {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"  Kendall tau: {kendall_t:.4f} (p={kendall_p:.2e})")

    # Print table of worst disagreements
    sorted_by_diff = sorted(results, key=lambda r: abs(r["rank_diff"]), reverse=True)
    print(f"\n  {'ID':>3} {'N':>3} {'R':>4} {'L':>4} {'V0':>4} {'C':>5} | "
          f"{'Ref v':>8} {'V7 v':>8} | {'Ref#':>4} {'V7#':>4} {'Diff':>5}")
    print("  " + "-" * 75)
    for r in sorted_by_diff[:10]:
        print(f"  {r['id']:3d} {r['N']:3d} {r['R_mean']:4.0f} {r['L']:4.0f} "
              f"{r['V0']:4.0f} {r['C_uF']:5.0f} | "
              f"{r['ref_exit_v']:8.1f} {r['v7_exit_v']:8.1f} | "
              f"{r['ref_rank']:4d} {r['v7_rank']:4d} {r['rank_diff']:+5d}")

    # Plot: ranking scatter
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Exit velocity scatter
    axes[0].scatter(ref_scores, v7_scores, alpha=0.6, s=30)
    lims = [min(min(ref_scores), min(v7_scores)), max(max(ref_scores), max(v7_scores))]
    axes[0].plot(lims, lims, "k--", alpha=0.3, label="y=x")
    axes[0].set_xlabel("Reference exit velocity (mm/s)")
    axes[0].set_ylabel("V7 PINN exit velocity (mm/s)")
    axes[0].set_title(f"Exit velocity: Spearman r={spearman_r:.3f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Rank scatter
    axes[1].scatter(ref_ranks, v7_ranks, alpha=0.6, s=30)
    axes[1].plot([1, n_total], [1, n_total], "k--", alpha=0.3)
    axes[1].set_xlabel("Reference rank")
    axes[1].set_ylabel("V7 rank")
    axes[1].set_title(f"Rank comparison (Kendall tau={kendall_t:.3f})")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "phase1_ranking.png", dpi=150)
    plt.close()
    print(f"\n  Saved: {PLOTS_DIR / 'phase1_ranking.png'}")

    return results, {"spearman_r": spearman_r, "kendall_tau": kendall_t}


# ---------------------------------------------------------------------------
# Phase 2: Trajectory-overlaid error heatmaps
# ---------------------------------------------------------------------------

def phase2_heatmaps(model, current_normalized, device, ranking_results):
    """Plot error heatmaps with trajectory overlay for key candidates."""
    print("\n" + "=" * 60)
    print("PHASE 2: Trajectory-overlaid error heatmaps")
    print("=" * 60)

    # Pick 3 candidates: best, mid-pack, worst disagreement
    sorted_by_ref = sorted(ranking_results, key=lambda r: r["ref_rank"])
    sorted_by_diff = sorted(ranking_results, key=lambda r: abs(r["rank_diff"]), reverse=True)

    cases = [
        ("best_candidate", sorted_by_ref[0]),
        ("mid_candidate", sorted_by_ref[len(sorted_by_ref) // 2]),
        ("worst_disagreement", sorted_by_diff[0]),
    ]

    pinn_fn = make_pinn_field_fn(model, current_normalized, device)

    for label, cand in cases:
        print(f"\n  {label}: id={cand['id']} N={cand['N']} R={cand['R_mean']} "
              f"L={cand['L']} V0={cand['V0']}  rank_diff={cand['rank_diff']:+d}")

        coil, drive = candidate_to_params(cand)

        # Simulate trajectories
        traj_ref = simulate_1d_launch(coil, drive, analytical_field_fn)
        traj_v7 = simulate_1d_launch(coil, drive, pinn_fn)

        # Error heatmap grid
        N = coil["num_turns"]
        R_mean = (coil["inner_radius_mm"] + coil["outer_radius_mm"]) / 2
        L_coil = coil["length_mm"]

        # Use representative current (peak)
        I_peak = rlc_peak_current(drive["V0"], drive["C_uF"], drive["L_uH"], drive["R_ohm"])

        r_grid = np.linspace(0.1, 40, 60).astype(np.float32)
        z_grid = np.linspace(-40, 40, 80).astype(np.float32)
        R_g, Z_g = np.meshgrid(r_grid, z_grid)
        fr = R_g.flatten().astype(np.float32)
        fz = Z_g.flatten().astype(np.float32)

        # Analytical field
        cp = dict(coil)
        cp["current_A"] = I_peak
        Bz_ref = np.array([solenoid_field(float(r), float(z), cp)[1]
                           for r, z in zip(fr, fz)])

        # PINN field
        _, Bz_v7 = pinn_predict_field(
            model, fr, fz, I_peak, N, R_mean, L_coil, current_normalized, device)

        # Relative error
        denom = max(np.abs(Bz_ref).max(), 1e-20)
        rel_err = (np.abs(Bz_v7 - Bz_ref) / denom * 100).reshape(R_g.shape)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        c = ax.contourf(Z_g, R_g, rel_err,
                        levels=[0, 1, 2, 5, 10, 20, 50, 100],
                        cmap="YlOrRd")
        plt.colorbar(c, ax=ax, label="Bz relative error (%)")

        # Overlay trajectory
        ax.plot(traj_ref["z"], np.full_like(traj_ref["z"], 0.1),
                "b-", linewidth=2, label="Reference path")
        ax.plot(traj_v7["z"], np.full_like(traj_v7["z"], 0.1),
                "c--", linewidth=2, label="V7 path")

        # Mark coil region
        ax.axvspan(-L_coil / 2, L_coil / 2, alpha=0.1, color="green")
        ax.axhline(R_mean, color="gray", linestyle=":", alpha=0.5, label=f"R_mean={R_mean}")

        ax.set_xlabel("z (mm)")
        ax.set_ylabel("r (mm)")
        ax.set_title(f"{label}: N={cand['N']} R={cand['R_mean']} L={cand['L']} "
                     f"V={cand['V0']}V  (rank diff={cand['rank_diff']:+d})")
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"phase2_{label}.png", dpi=150)
        plt.close()
        print(f"    Saved: {PLOTS_DIR / f'phase2_{label}.png'}")

        # Force comparison along trajectory
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(traj_ref["z"], traj_ref["F"], "b-", label="Reference", linewidth=2)
        axes[0].plot(traj_v7["z"], traj_v7["F"], "r--", label="V7", linewidth=2)
        axes[0].set_xlabel("z (mm)")
        axes[0].set_ylabel("F_z (mN)")
        axes[0].set_title("Force along trajectory")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Cumulative impulse
        if len(traj_ref["t"]) > 1:
            dt_ref = np.diff(traj_ref["t"])
            impulse_ref = np.cumsum(np.abs(traj_ref["F"][:-1]) * dt_ref)
            axes[1].plot(traj_ref["z"][:-1], impulse_ref, "b-", label="Reference")
        if len(traj_v7["t"]) > 1:
            dt_v7 = np.diff(traj_v7["t"])
            impulse_v7 = np.cumsum(np.abs(traj_v7["F"][:-1]) * dt_v7)
            axes[1].plot(traj_v7["z"][:-1], impulse_v7, "r--", label="V7")
        axes[1].set_xlabel("z (mm)")
        axes[1].set_ylabel("Cumulative |impulse| (mN*s)")
        axes[1].set_title("Cumulative impulse")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"phase2_{label}_force.png", dpi=150)
        plt.close()
        print(f"    Saved: {PLOTS_DIR / f'phase2_{label}_force.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  V7 PINN OPTIMIZATION DIAGNOSTIC")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cn, meta = load_pinn_model(device)
    print(f"Loaded: step={meta['step']}, current_normalized={cn}")

    # Phase 1
    ranking_results, stats = phase1_ranking(model, cn, device, n_candidates=54)

    # Phase 2
    phase2_heatmaps(model, cn, device, ranking_results)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "stats": stats,
        "candidates": ranking_results,
    }
    out_path = RESULTS_DIR / "v7_optimization_diagnostic.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults saved: {out_path}")

    # Verdict
    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)
    sr = stats["spearman_r"]
    kt = stats["kendall_tau"]
    if sr > 0.95 and kt > 0.90:
        print(f"  Spearman r={sr:.3f}, Kendall tau={kt:.3f}")
        print("  -> V7 is SAFE for design optimization (strong ranking preservation)")
    elif sr > 0.85:
        print(f"  Spearman r={sr:.3f}, Kendall tau={kt:.3f}")
        print("  -> V7 is USABLE for optimization with finalist validation against reference")
    else:
        print(f"  Spearman r={sr:.3f}, Kendall tau={kt:.3f}")
        print("  -> V7 ranking diverges from reference — use with caution")


if __name__ == "__main__":
    main()
