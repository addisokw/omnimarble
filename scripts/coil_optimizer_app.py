"""Gradio web UI for the coil design optimizer.

Launch with:
    uv run python scripts/coil_optimizer_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import gradio as gr
import pandas as pd

from coil_optimizer_core import (
    UserConstraints,
    load_pinn_designspace,
    plot_pareto,
    run_optimization,
)

# ============================================================
# Lazy model loading
# ============================================================

_model = None
_device = None


def _ensure_model_loaded():
    """Load the PINN model on first use. Caches for subsequent runs."""
    global _model, _device
    if _model is not None:
        return _model, _device

    import torch
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model, _device = load_pinn_designspace(_device)
    return _model, _device


# ============================================================
# Run handler
# ============================================================

def run_handler(max_voltage, max_current, thinnest_awg, max_turns,
                max_temp_rise, n_samples, target_boost_enabled, target_boost_value,
                progress=gr.Progress()):
    """Run optimization and return (recommendation_md, dataframe, figure)."""
    try:
        model, device = _ensure_model_loaded()
    except Exception as e:
        error_md = f"### Error loading model\n\n`{e}`"
        default_label = "Top 10 Reranked Designs (from screening top 50)"
        return error_md, gr.update(value=None, label=default_label), None

    target = target_boost_value if target_boost_enabled else None

    constraints = UserConstraints(
        max_voltage_V=max_voltage,
        max_current_A=max_current,
        thinnest_wire_awg=int(thinnest_awg),
        max_turns=int(max_turns),
        max_temp_rise_C=max_temp_rise,
        n_samples=int(n_samples),
        seed=42,
        target_boost_ms=target,
    )

    def progress_callback(current, total, phase):
        if phase == "screening":
            progress(current / total, desc=f"Screening: {current}/{total}")
        elif phase == "reranking":
            frac = 0.8 + 0.2 * (current / total)
            progress(frac, desc=f"Reranking: {current}/{total}")

    progress(0, desc="Starting optimization...")
    result = run_optimization(constraints, model, device,
                              progress_callback=progress_callback)

    is_target_mode = target is not None

    if not result.scored:
        no_results_md = (
            "### No valid candidates\n\n"
            "All designs were rejected by the constraints. "
            "Try relaxing the limits (higher voltage, current, or temperature rise)."
        )
        default_label = "Top 10 Reranked Designs (from screening top 50)"
        return no_results_md, gr.update(value=None, label=default_label), None

    # In target mode, check if any candidates fell within the band
    if is_target_mode:
        margin = max(target * 0.20, 0.1)
    else:
        margin = 0.0
    any_met_screening = is_target_mode and any(
        target - margin <= r["boost_ms"] <= target + margin
        for r in result.scored
    )
    any_met_coupled = is_target_mode and any(
        r.get("meets_target_coupled", False) for r in result.coupled_top
    )

    if is_target_mode and not any_met_screening:
        closest = min((r["boost_ms"] for r in result.scored),
                      key=lambda b: abs(b - target),
                      default=0)
        no_target_md = (
            f"### No designs within target band\n\n"
            f"No designs achieved a boost within "
            f"**{target:.1f} \u00b1 {margin:.2f} m/s**. "
            f"The closest was **{closest:.3f} m/s**. "
            f"Try adjusting the target or relaxing constraints."
        )
        default_label = "Top 10 Reranked Designs (from screening top 50)"
        return no_target_md, gr.update(value=None, label=default_label), None

    # Build recommendation card
    best = result.coupled_top[0]
    v = best["V0"]
    if v <= 50:
        safety = "SAFE — battery/USB-C powered"
    elif v <= 120:
        safety = "HOBBY-SAFE — use caution"
    elif v <= 400:
        safety = "REQUIRES ENCLOSURE — HV interlock recommended"
    else:
        safety = "DANGEROUS — lethal voltage"

    # Banner for target mode when coupled reranking demoted all candidates
    banner = ""
    if is_target_mode and not any_met_coupled:
        banner = (
            f"> **Note:** No designs fell within **{target:.1f} \u00b1 {margin:.2f} m/s** "
            f"after coupled ODE reranking. The following near-miss designs came closest.\n\n"
        )

    if is_target_mode and any_met_coupled:
        heading = "### Most Efficient Design"
        target_row = f"| **Target boost** | {target:.1f} m/s (met) |"
    elif is_target_mode:
        heading = "### Best Near-Miss Design"
        target_row = f"| **Target boost** | {target:.1f} m/s (not met after coupled reranking) |"
    else:
        heading = "### Recommended Design"
        target_row = ""

    target_row_line = f"\n{target_row}" if target_row else ""

    rec_md = f"""{heading}

{banner}| Parameter | Value |
|-----------|-------|
| **Turns** | {best['N']} turns of {best['wire_gauge_awg']} AWG |
| **Inner bore** | {best['inner_radius_mm']:.1f} mm |
| **Coil length** | {best['length_mm']:.1f} mm |
| **Layers** | {best['num_layers']} |
| **Voltage / Capacitance** | {best['V0']:.0f} V / {best['C_uF']:.0f} uF ({best['stored_energy_J']:.1f} J) |
| **Peak current** | {best['peak_current_A']:.0f} A |
| **Boost** | {best['boost_ms']:.3f} m/s |
| **Coupled boost** | {best.get('boost_coupled_ms', 0):.3f} m/s |
| **Temp rise** | {best['delta_T_C']:.1f} °C |
| **Safety** | {safety} |{target_row_line}

*{result.n_valid} valid designs from {result.n_samples} samples, {result.n_rejected} rejected. Screening: {result.eval_time_s:.1f}s, Reranking: {result.rerank_time_s:.1f}s*
"""

    # Build top-10 table
    top10 = result.coupled_top[:10]
    rows = []
    for i, r in enumerate(top10):
        row = {
            "Rank": i + 1,
            "Boost (m/s)": f"{r['boost_ms']:.3f}",
            "Coupled (m/s)": f"{r.get('boost_coupled_ms', 0):.3f}",
            "Penalty": f"{r['combined_penalty']:.3f}",
            "N": r["N"],
            "AWG": r["wire_gauge_awg"],
            "V0": f"{r['V0']:.0f}",
            "C (uF)": f"{r['C_uF']:.0f}",
            "I_pk (A)": f"{r['peak_current_A']:.0f}",
            "dT (°C)": f"{r['delta_T_C']:.1f}",
            "Regime": r["regime"],
        }
        if not is_target_mode:
            row["Score"] = f"{r['score']:.3f}"
        rows.append(row)
    df = pd.DataFrame(rows)

    # Dynamic table label
    if is_target_mode and any_met_coupled:
        table_label = "Top 10 Most Efficient Designs"
    elif is_target_mode:
        table_label = "Top 10 Near-Miss Designs"
    else:
        table_label = "Top 10 Reranked Designs (from screening top 50)"

    # Pareto plot
    fig = plot_pareto(result.scored, recommended=result.coupled_top[0])

    return rec_md, gr.update(value=df, label=table_label), fig


# ============================================================
# UI layout
# ============================================================

with gr.Blocks(title="Coil Design Optimizer") as app:
    gr.Markdown("# Coil Design Optimizer\n"
                "Set constraints and run the multi-objective optimizer to find "
                "the best coil design for your marble coaster stage.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Constraints")
            max_voltage = gr.Slider(
                20, 450, value=200, step=10,
                label="Max Voltage (V)",
            )
            max_current = gr.Slider(
                100, 4000, value=2000, step=100,
                label="Max Peak Current (A)",
            )
            thinnest_awg = gr.Dropdown(
                choices=["18", "20", "22", "24", "26"],
                value="22",
                label="Thinnest Allowed Wire (AWG)",
                info="Larger AWG = thinner wire. Selecting 22 excludes 24 and 26.",
            )
            max_turns = gr.Slider(
                10, 80, value=60, step=5,
                label="Max Turns",
            )
            max_temp_rise = gr.Slider(
                10, 300, value=100, step=10,
                label="Max Temp Rise (°C)",
            )
            n_samples = gr.Slider(
                500, 5000, value=2000, step=500,
                label="Number of Samples",
            )

            gr.Markdown("### Target Boost Mode")
            target_boost_enabled = gr.Checkbox(
                label="Target a specific boost",
                value=False,
                info="Find the cheapest/safest design that achieves a target boost.",
            )
            target_boost_value = gr.Slider(
                0.1, 5.0, value=1.0, step=0.1,
                label="Target Boost (m/s)",
            )

            run_btn = gr.Button("Run Optimization", variant="primary")

        with gr.Column(scale=2):
            rec_output = gr.Markdown(label="Recommended Design")
            table_output = gr.DataFrame(label="Top 10 Reranked Designs (from screening top 50)")
            plot_output = gr.Plot(label="Pareto Front")

    run_btn.click(
        fn=run_handler,
        inputs=[max_voltage, max_current, thinnest_awg, max_turns,
                max_temp_rise, n_samples, target_boost_enabled, target_boost_value],
        outputs=[rec_output, table_output, plot_output],
    )

if __name__ == "__main__":
    app.queue(default_concurrency_limit=1)
    app.launch()
