"""Multi-objective coil design optimizer using the validated v7 PINN surrogate.

Searches a 6D design space (N, inner_radius, L, wire_gauge, V0, C_uF) using
Latin Hypercube Sampling. Balances launch velocity against voltage danger,
capacitor cost, construction difficulty, and thermal safety.

Usage:
    uv run python scripts/optimize_coil_design.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from coil_optimizer_core import (
    W_BUILD,
    W_COST,
    W_DANGER,
    W_THERMAL,
    W_VEL,
    MARBLE_RADIUS_MM,
    UserConstraints,
    analytical_boost,
    build_difficulty,
    capacitor_cost,
    load_pinn_designspace,
    plot_pareto,
    run_optimization,
    thermal_penalty,
    voltage_danger,
)
from rlc_circuit import compute_rlc_params

try:
    import torch
except ImportError:
    print("ERROR: PyTorch required")
    raise SystemExit(1)


# ============================================================
# Output formatting
# ============================================================

def print_top_table(scored, n=10):
    """Print top N designs as a table."""
    print(f"\n{'='*120}")
    print(f"  TOP {n} COIL DESIGNS — SCREENING PASS (by composite score)")
    print(f"{'='*120}")
    header = (f"{'Rank':>4} {'Score':>6} {'Boost':>8} {'N':>3} {'R_in':>5} {'L':>5} "
              f"{'AWG':>3} {'V0':>5} {'C_uF':>6} {'Lyrs':>4} {'I_pk':>6} "
              f"{'Regime':>11} {'dT':>5} {'Pen':>5}")
    print(header)
    print("-" * 120)
    for i, r in enumerate(scored[:n]):
        print(f"{i+1:4d} {r['score']:6.3f} {r['boost_ms']:7.3f}m "
              f"{r['N']:3d} {r['inner_radius_mm']:5.1f} {r['length_mm']:5.1f} "
              f"{r['wire_gauge_awg']:3d} {r['V0']:5.0f} {r['C_uF']:6.0f} "
              f"{r['num_layers']:4d} {r['peak_current_A']:6.0f} "
              f"{r['regime']:>11s} {r['delta_T_C']:5.1f} {r['combined_penalty']:5.3f}")


def print_detailed_specs(result, rank):
    """Print detailed specs for a single design."""
    print(f"\n{'-'*60}")
    print(f"  DESIGN #{rank} -- Score: {result['score']:.4f}")
    print(f"{'-'*60}")
    print(f"  Turns:           {result['N']}")
    print(f"  Inner radius:    {result['inner_radius_mm']:.1f} mm")
    print(f"  Outer radius:    {result['outer_radius_mm']:.1f} mm")
    print(f"  Mean radius:     {result['R_mean_mm']:.1f} mm")
    print(f"  Coil length:     {result['length_mm']:.1f} mm")
    print(f"  Wire gauge:      {result['wire_gauge_awg']} AWG ({result['wire_diameter_mm']:.3f} mm)")
    print(f"  Layers:          {result['num_layers']}")
    print(f"  Wire length:     {result['wire_length_mm']/1000:.2f} m")
    print(f"  Wire mass:       {result['wire_mass_g']:.1f} g")
    print()
    print(f"  Voltage:         {result['V0']:.0f} V")
    print(f"  Capacitance:     {result['C_uF']:.0f} uF")
    print(f"  Stored energy:   {result['stored_energy_J']:.2f} J")
    print(f"  DC resistance:   {result['R_dc_ohm']:.4f} ohm")
    print(f"  Inductance:      {result['L_uH']:.2f} uH")
    print(f"  Regime:          {result['regime']}")
    print(f"  Peak current:    {result['peak_current_A']:.0f} A")
    print(f"  Pulse duration:  {result['pulse_duration_s']*1e3:.2f} ms")
    print()
    print(f"  Entry velocity:  {result['entry_velocity_ms']:.3f} m/s")
    print(f"  Exit velocity:   {result['exit_velocity_ms']:.3f} m/s")
    print(f"  Boost:           {result['boost_ms']:.3f} m/s")
    print(f"  Temp rise:       {result['delta_T_C']:.1f} C")
    print()
    print(f"  Penalties:")
    print(f"    Voltage:       {result['penalty_voltage']:.3f}")
    print(f"    Cost:          {result['penalty_cost']:.3f}")
    print(f"    Build:         {result['penalty_build']:.3f}")
    print(f"    Thermal:       {result['penalty_thermal']:.3f}")
    print(f"    Combined:      {result['combined_penalty']:.3f}")


def print_recommendation(best):
    """Print the recommended best practical design."""
    print(f"\n{'='*60}")
    print(f"  RECOMMENDED BEST PRACTICAL DESIGN")
    print(f"{'='*60}")
    print()
    print(f"  {best['N']} turns of {best['wire_gauge_awg']} AWG wire")
    print(f"  Inner bore: {best['inner_radius_mm']:.1f} mm, Length: {best['length_mm']:.1f} mm")
    print(f"  Powered by {best['V0']:.0f}V / {best['C_uF']:.0f}uF ({best['stored_energy_J']:.1f}J)")
    print()

    # Safety tier
    v = best["V0"]
    if v <= 50:
        safety = "SAFE — battery/USB-C powered, no enclosure needed"
    elif v <= 120:
        safety = "HOBBY-SAFE — use caution, consider a case"
    elif v <= 400:
        safety = "REQUIRES ENCLOSURE — HV interlock recommended"
    else:
        safety = "DANGEROUS — lethal voltage, professional build only"
    print(f"  Safety:     {safety}")
    print(f"  Boost:      {best['boost_ms']:.3f} m/s ({best['entry_velocity_ms']:.2f} -> {best['exit_velocity_ms']:.2f} m/s)")
    print(f"  Peak I:     {best['peak_current_A']:.0f} A, Temp rise: {best['delta_T_C']:.1f}C")
    print(f"  Buildable:  {best['num_layers']} layer(s), {best['wire_length_mm']/1000:.1f}m wire")
    if "boost_coupled_ms" in best:
        print(f"  Coupled:    {best['boost_coupled_ms']:.3f} m/s (rank #{best.get('rank_coupled', '?')})")
    print()
    print("  NOTE: Screening pass uses closed-form current + cutoff + saturation")
    print("  + eddy braking. Top 50 reranked with approximate coupled ODE")
    print("  (adds back-EMF). See coupled boost for higher-fidelity estimates.")


def print_rerank_table(top_candidates, coupled_sorted):
    """Print the coupled ODE reranking comparison table."""
    print(f"\n{'='*80}")
    print(f"  COUPLED ODE RERANKING (top {len(top_candidates)} candidates)")
    print(f"{'='*80}")
    print("  Using position-dependent inductance (back-EMF) + cutoff + saturation + eddy")

    print(f"\n  {'Scrn':>4} {'Cpld':>4} {'Screening Boost':>15} {'Coupled Boost':>15} "
          f"{'Delta':>8} {'N':>3} {'V0':>5} {'C_uF':>6}")
    print("  " + "-" * 72)
    for c in top_candidates[:20]:
        orig_rank = top_candidates.index(c) + 1
        delta = c["boost_coupled_ms"] - c["boost_ms"]
        print(f"  {orig_rank:4d} {c['rank_coupled']:4d} "
              f"{c['boost_ms']:14.4f}m {c['boost_coupled_ms']:14.4f}m "
              f"{delta:+7.4f} {c['N']:3d} {c['V0']:5.0f} {c['C_uF']:6.0f}")


# ============================================================
# Verification
# ============================================================

def run_verification(scored, model, device):
    """Run verification checks on results."""
    print(f"\n{'='*60}")
    print(f"  VERIFICATION")
    print(f"{'='*60}")

    # --- 1. Analytical boost validation for top 5 + 5 random ---
    print("\n--- Boost validation (PINN vs analytical) ---")
    rng = np.random.default_rng(99)
    test_indices = list(range(min(5, len(scored))))
    if len(scored) > 10:
        test_indices += rng.choice(range(5, len(scored)), size=5, replace=False).tolist()

    pinn_boosts = []
    ana_boosts = []
    for idx in test_indices:
        r = scored[idx]
        rlc_p = {
            "capacitance_uF": r["C_uF"],
            "charge_voltage_V": r["V0"],
            "inductance_uH": r["L_uH"],
            "total_resistance_ohm": r["R_total_ohm"],
        }
        _, _, boost_ana = analytical_boost(r, rlc_p)
        pinn_boosts.append(r["boost_ms"])
        ana_boosts.append(boost_ana)
        rank_label = f"#{idx+1}" if idx < 5 else f"random"
        print(f"  {rank_label:>8s}: PINN={r['boost_ms']:.4f} m/s  "
              f"Ana={boost_ana:.4f} m/s  "
              f"diff={abs(r['boost_ms'] - boost_ana):.4f}")

    # Spearman rank correlation
    from scipy.stats import spearmanr
    if len(pinn_boosts) >= 3:
        rho, _ = spearmanr(pinn_boosts, ana_boosts)
        print(f"  Spearman rank correlation: {rho:.4f}")

    # --- 2. Sanity checks on penalties ---
    print("\n--- Penalty sanity checks ---")

    # Reference design: 30 turns, 20 AWG, 50V, 470uF
    ref_penalty = (W_DANGER * voltage_danger(50) +
                   W_COST * capacitor_cost(50, 470) +
                   W_BUILD * build_difficulty(30, 15.0, 30.0, 20, 1, 12.0) +
                   W_THERMAL * thermal_penalty(10))
    print(f"  Reference design (30T/20AWG/50V/470uF) penalty: {ref_penalty:.4f}")
    assert ref_penalty < 0.1, f"Reference penalty too high: {ref_penalty}"
    print(f"    -> Near-zero as expected: PASS")

    # Maximal design: 80 turns, 400V, 4700uF — should have significantly higher penalty
    max_penalty = (W_DANGER * voltage_danger(400) +
                   W_COST * capacitor_cost(400, 4700) +
                   W_BUILD * build_difficulty(80, 15.0, 50.0, 26, 4, 8.0) +
                   W_THERMAL * thermal_penalty(250))
    print(f"  Maximal design (80T/26AWG/400V/4700uF) penalty: {max_penalty:.4f}")
    assert max_penalty > 0.3, f"Maximal penalty too low: {max_penalty}"
    assert max_penalty > ref_penalty * 5, f"Maximal not much worse than reference"
    print(f"    -> >> reference penalty as expected: PASS")

    # --- 3. Overdamped inclusion ---
    print("\n--- Overdamped inclusion check ---")
    n_overdamped = sum(1 for r in scored if r["regime"] == "overdamped")
    n_critical = sum(1 for r in scored if r["regime"] == "critically_damped")
    n_underdamped = sum(1 for r in scored if r["regime"] == "underdamped")
    print(f"  Underdamped: {n_underdamped}, Critically damped: {n_critical}, Overdamped: {n_overdamped}")
    if n_overdamped > 0:
        print(f"    -> Overdamped candidates survived: PASS")
    else:
        print(f"    -> WARNING: No overdamped candidates in results")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  MULTI-OBJECTIVE COIL DESIGN OPTIMIZER")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, device = load_pinn_designspace(device)

    # Run optimization with default constraints
    constraints = UserConstraints()
    n_samples = constraints.n_samples
    print(f"\nGenerating {n_samples} candidates via Latin Hypercube Sampling...")
    print(f"Evaluating candidates...")

    def cli_progress(current, total, phase):
        if phase == "screening":
            print(f"  {current}/{total} evaluated...")
        elif phase == "reranking" and current % 10 == 0:
            print(f"  {current}/{total} reranked...")

    result = run_optimization(constraints, model, device, progress_callback=cli_progress)

    elapsed = result.eval_time_s
    print(f"\nEvaluation complete: {result.n_valid} valid / {result.n_rejected} rejected "
          f"in {elapsed:.1f}s ({elapsed/n_samples*1000:.1f}ms per candidate)")

    if not result.scored:
        print("ERROR: No valid candidates survived filtering")
        return

    scored = result.scored
    coupled_top = result.coupled_top

    # Output
    print_top_table(scored, n=10)

    # Print reranking table
    print(f"\nReranking top {min(50, len(scored))} with coupled ODE (back-EMF)...")
    print_rerank_table(scored[:min(50, len(scored))], coupled_top)
    print(f"Coupled ODE reranking complete in {result.rerank_time_s:.1f}s")

    for i in range(min(3, len(coupled_top))):
        print_detailed_specs(coupled_top[i], i + 1)

    print_recommendation(coupled_top[0])

    # Plots
    plots_dir = ROOT / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_pareto(scored, plots_dir / "coil_optimization_pareto.png",
               recommended=coupled_top[0])
    print(f"\nPareto plot saved: {plots_dir / 'coil_optimization_pareto.png'}")

    # Verification
    run_verification(scored, model, device)

    # Save JSON results
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "coil_optimization_results.json"

    # Serialize top 50 + summary
    json_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_samples": n_samples,
        "n_valid": result.n_valid,
        "n_rejected": result.n_rejected,
        "runtime_s": elapsed,
        "weights": {
            "W_VEL": W_VEL, "W_DANGER": W_DANGER,
            "W_COST": W_COST, "W_BUILD": W_BUILD, "W_THERMAL": W_THERMAL,
        },
        "recommended": coupled_top[0],
        "top_50": coupled_top,
    }
    json_path.write_text(json.dumps(json_output, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved: {json_path}")

    print(f"\n{'='*60}")
    print("  OPTIMIZATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
