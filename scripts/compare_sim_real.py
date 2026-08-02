"""Compare simulated shots against the bench rig's shots.csv.

    uv run python scripts/compare_sim_real.py --real ../omnimarble-vbench/shots.csv \
        --sim results/sweeps/sim_shots.csv --out results/plots/sim_vs_real.png

Two panels, because they separate the two ways the twin can be wrong:

  Delta-v vs C     the FORCE model. A miss here is the PINN field, the
                   saturation model, or the fire position.
  energy / I_peak  the CIRCUIT model. A miss here is L, R, C or the on-time.

Both mark the zeta = 1 crossing, which sits inside the sweep between one and
two cans. A kink there that the sim misses is diagnostic on its own.

Three things this refuses to do quietly:

  * Pair on can count alone. The rig deliberately varies release height so
    v_in differs shot to shot ("release the ball part-way down the slope for a
    lower v_in; v_in is measured every shot so the spread costs nothing"), so
    two shots at the same can count are not the same experiment. Default
    pairing is nearest-v_in within a can bucket.
  * Accept shots that were not properly measured. A station that gave 3 of 5
    channels, or a large fit residual, is a different measurement from a clean
    one, and the raw dv alone cannot tell you which you had.
  * Accept a stale sensor pitch. Any row reading 11.0 predates the 2026-08-01
    correction and has every velocity -- and therefore every dv -- wrong by a
    factor of 22.14/11.0. That is a hard failure, not a warning.

Column handling is by NAME INTERSECTION throughout, because the rig's schema is
about to grow (vbench NEXT_SESSION.md section 3 adds five bank-recharge columns).
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STALE_PITCH_MM = 11.0
PITCH_TOL_MM = 0.01
RESID_LIMIT_US = 2000.0
REQUIRED_CHANNELS = 5


def _num(row, key):
    value = row.get(key, "")
    if value in ("", None, "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_shots(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)]


def check_pitch(rows, label):
    """Hard-fail on the 11.0 placeholder; warn if a file mixes pitches."""
    pitches = {p for p in (_num(r, "sensor_pitch_mm") for r in rows)
               if p is not None}
    if not pitches:
        print(f"  {label}: no sensor_pitch_mm column -- cannot verify the "
              f"calibration these velocities depend on", file=sys.stderr)
        return pitches
    stale = [p for p in pitches if abs(p - STALE_PITCH_MM) < PITCH_TOL_MM]
    if stale:
        raise SystemExit(
            f"{label} contains shots recorded with sensor_pitch_mm = "
            f"{STALE_PITCH_MM}. That is the pre-2026-08-01 placeholder: every "
            f"velocity and every dv in those rows is out by a factor of "
            f"{22.14 / STALE_PITCH_MM:.4f}. Re-run them, or drop them "
            f"explicitly -- they cannot be compared as they stand.")
    if len(pitches) > 1:
        print(f"  WARNING: {label} mixes sensor pitches {sorted(pitches)}",
              file=sys.stderr)
    return pitches


def quality_filter(rows, label):
    """Drop shots whose measurement provenance says they are not comparable."""
    kept, dropped = [], []
    for row in rows:
        reasons = []
        for key in ("n_ch_in", "n_ch_out"):
            n = _num(row, key)
            if n is not None and n < REQUIRED_CHANNELS:
                reasons.append(f"{key}={n:.0f}")
        for key in ("resid_in_us", "resid_out_us"):
            resid = _num(row, key)
            if resid is not None and resid > RESID_LIMIT_US:
                reasons.append(f"{key}={resid:.0f}us")
        if _num(row, "dv_mps") is None:
            reasons.append("no dv")
        (dropped if reasons else kept).append((row, reasons))
    if dropped:
        print(f"  {label}: dropped {len(dropped)} of {len(rows)} shots on "
              f"measurement quality:")
        for _, reasons in dropped[:5]:
            print(f"    - {', '.join(reasons)}")
        if len(dropped) > 5:
            print(f"    ... and {len(dropped) - 5} more")
    return [row for row, _ in kept]


def pair_shots(real, sim, mode):
    """Match each real shot to a simulated one. Returns (real, sim) tuples."""
    pairs = []
    if mode == "cans":
        by_cans = {}
        for row in sim:
            by_cans.setdefault(_num(row, "cans"), []).append(row)
        for row in real:
            candidates = by_cans.get(_num(row, "cans"))
            if candidates:
                pairs.append((row, candidates[0]))
        return pairs

    # nearest-v_in within a can bucket: the rig varies release height on
    # purpose, so same-cans shots are not the same experiment.
    for row in real:
        cans = _num(row, "cans")
        v_in = _num(row, "v_in_mps")
        bucket = [s for s in sim if _num(s, "cans") == cans]
        if not bucket:
            continue
        if v_in is None:
            pairs.append((row, bucket[0]))
            continue
        best = min(bucket, key=lambda s: abs((_num(s, "v_in_mps") or 0) - v_in))
        pairs.append((row, best))
    return pairs


def fit_pitch_scale(pairs):
    """Single multiplicative k minimising ||k*dv_real - dv_sim||."""
    num = sum((_num(r, "dv_mps") or 0) * (_num(s, "dv_mps") or 0) for r, s in pairs)
    den = sum((_num(r, "dv_mps") or 0) ** 2 for r, _ in pairs)
    return num / den if den else None


def plot(pairs, out_path, zeta_crossing_uF=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib unavailable -- skipping the plot", file=sys.stderr)
        return

    fig, (ax_force, ax_circuit) = plt.subplots(1, 2, figsize=(13, 5))

    c_real = [_num(r, "C_uF") for r, _ in pairs]
    ax_force.plot(c_real, [_num(r, "dv_mps") for r, _ in pairs],
                  "o", label="measured", color="#c0392b")
    ax_force.plot([_num(s, "C_uF") for _, s in pairs],
                  [_num(s, "dv_mps") for _, s in pairs],
                  "s--", label="sim", color="#2c7fb8")
    ax_force.set_xlabel("bank capacitance (uF)")
    ax_force.set_ylabel("dv (m/s)")
    ax_force.set_title("Force model: dv vs C")

    ax_circuit.plot(c_real, [_num(r, "i_peak_A") for r, _ in pairs],
                    "o", label="measured I_peak (scope)", color="#c0392b")
    ax_circuit.plot([_num(s, "C_uF") for _, s in pairs],
                    [_num(s, "sim_i_peak_A_model") for _, s in pairs],
                    "s--", label="sim I_peak (model)", color="#2c7fb8")
    ax_circuit.set_xlabel("bank capacitance (uF)")
    ax_circuit.set_ylabel("peak current (A)")
    ax_circuit.set_title("Circuit model: I_peak vs C")

    for ax in (ax_force, ax_circuit):
        if zeta_crossing_uF:
            ax.axvline(zeta_crossing_uF, color="#888", ls=":", lw=1)
            ax.annotate("zeta = 1", xy=(zeta_crossing_uF, ax.get_ylim()[1]),
                        xytext=(4, -12), textcoords="offset points",
                        fontsize=8, color="#666")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"  plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", type=Path, required=True,
                        help="the rig's shots.csv")
    parser.add_argument("--sim", type=Path, required=True,
                        help="simulated shots (scripts/simulate_rig_shot.py --shots-out)")
    parser.add_argument("--pair-by", choices=["nearest-v-in", "cans"],
                        default="nearest-v-in")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "plots" / "sim_vs_real.png")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "results" / "sim_vs_real.json")
    parser.add_argument("--fit-pitch", action="store_true",
                        help="fit a single scale factor between measured and "
                             "simulated dv (a consistency check, NOT a validation)")
    parser.add_argument("--no-quality-filter", action="store_true")
    args = parser.parse_args()

    for path in (args.real, args.sim):
        if not path.exists():
            raise SystemExit(f"not found: {path}")

    real = load_shots(args.real)
    sim = load_shots(args.sim)
    print(f"loaded {len(real)} real and {len(sim)} simulated shots")

    check_pitch(real, "real")
    check_pitch(sim, "sim")

    if not args.no_quality_filter:
        real = quality_filter(real, "real")
        sim = quality_filter(sim, "sim")

    pairs = pair_shots(real, sim, args.pair_by)
    if not pairs:
        raise SystemExit("no shots could be paired -- check the can counts overlap")
    print(f"paired {len(pairs)} shots by {args.pair_by}")

    residuals = []
    print(f"\n{'cans':>5} {'C_uF':>7} {'v_in real':>10} {'v_in sim':>9} "
          f"{'dv real':>9} {'dv sim':>9} {'resid':>9} {'rel':>7}")
    print("-" * 76)
    for r, s in pairs:
        dv_r, dv_s = _num(r, "dv_mps"), _num(s, "dv_mps")
        if dv_r is None or dv_s is None:
            continue
        resid = dv_s - dv_r
        rel = resid / dv_r if dv_r else float("nan")
        residuals.append({"cans": _num(r, "cans"), "C_uF": _num(r, "C_uF"),
                          "dv_real": dv_r, "dv_sim": dv_s,
                          "residual": resid, "relative": rel})
        print(f"{_num(r,'cans') or 0:>5.0f} {_num(r,'C_uF') or 0:>7.0f} "
              f"{_num(r,'v_in_mps') or 0:>10.4f} {_num(s,'v_in_mps') or 0:>9.4f} "
              f"{dv_r:>9.4f} {dv_s:>9.4f} {resid:>9.4f} {rel:>6.1%}")

    summary = {}
    if residuals:
        errs = [abs(x["residual"]) for x in residuals]
        rels = [abs(x["relative"]) for x in residuals if not math.isnan(x["relative"])]
        summary = {
            "n_pairs": len(residuals),
            "mean_abs_residual_mps": sum(errs) / len(errs),
            "max_abs_residual_mps": max(errs),
            "mean_abs_relative": sum(rels) / len(rels) if rels else None,
        }
        print(f"\nmean |residual| {summary['mean_abs_residual_mps']:.4f} m/s, "
              f"max {summary['max_abs_residual_mps']:.4f} m/s"
              + (f", mean relative {summary['mean_abs_relative']:.1%}"
                 if summary["mean_abs_relative"] is not None else ""))

    if args.fit_pitch:
        k = fit_pitch_scale(pairs)
        if k:
            summary["fitted_scale"] = k
            summary["implied_pitch_mm"] = 22.14 * k
            print(f"\nfitted scale k = {k:.4f} -> implied pitch "
                  f"{22.14 * k:.3f} mm")
            print("  NOTE: pitch was FITTED, not measured. dv scales linearly "
                  "with pitch, so this calibrated a constant -- it did NOT "
                  "validate the physics. The pitch is already measured at "
                  "22.14 +/-0.05 mm; if this disagrees materially, the "
                  "disagreement IS your result. k near 2 means someone is "
                  "running stale firmware.")

    plot(pairs, args.out, zeta_crossing_uF=3818.0)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"summary": summary, "pairs": residuals}, indent=2) + "\n",
            encoding="utf-8")
        print(f"  report -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
