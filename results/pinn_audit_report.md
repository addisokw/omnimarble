# PINN B-Field Model Audit Report

Last updated: 2026-07-02

## 1. Executive Summary

The Physics-Informed Neural Network (PINN) replaces the analytical Biot-Savart
B-field computation for a finite solenoid coilgun. It takes spatial position and
coil design parameters as input and produces the magnetic vector potential and
field components in a single forward pass.

**Current status: v8 "physics-by-construction" model (step-250k checkpoint)
deployed as the single production checkpoint `pinn_best.pt`. 4 of 5 levels
pass; Level 5 passes fully for the first time (div(B) = 0 exact by
architecture). The only remaining FAIL is the informational L1 max criterion
(27.8%, improved from v7's 37.2%; concentrated at winding singularities the
smooth network cannot represent). The v7-era dual-baseline strategy is
retired — see Section 12.**

| Check | v3 | v4 (field) | v5 (regressed) | v7 | **v8 (production)** |
|-------|----|----|-----|------|------|
| Level 1: Field accuracy | FAIL | **PASS** | FAIL | FAIL (max only) | FAIL (max only, 27.8%) |
| Level 2: Gradient accuracy | FAIL | FAIL | FAIL | **PASS** (P99) | **PASS** (P99=3.05%) |
| Level 3: Force accuracy | FAIL | **PASS** | **PASS** | **PASS** | **PASS** (2.7%/3.0%) |
| Level 4: Design space | FAIL | FAIL (75%) | FAIL (61%) | **PASS (100%)** | **PASS (95.3%)** |
| Level 5: Physics consistency | FAIL (0/3) | FAIL (2/3) | FAIL (2/3) | FAIL (2/3, div=0.014) | **PASS (3/3, div=0 exact)** |

## 2. Problem Statement

The analytical Biot-Savart solenoid field computation requires 30 loop iterations
with scipy elliptic integrals per evaluation point. For real-time force computation
in Omniverse Kit at 500 Hz PhysX stepping, and for design exploration across coil
geometries, a faster surrogate model was needed.

## 3. Model Architecture

- **Backbone**: NVIDIA PhysicsNeMo FullyConnected
- **Hidden layers**: 6 x 256, SiLU activation, skip connections
- **Parameters**: ~331k
- **Inputs**: (r, z, I, N, R_mean, L) -- spatial position + coil design params
- **Outputs**: (A_phi, B_r, B_z)
- **Key feature**: Model learns B/I (T/A), collapsing dynamic range. Checkpoint flag `current_normalized=True`.

## 4. Training Data

### v3/v4 data (1.02M samples)
- **Spatial**: 30k dense + 20k boundary-enriched + 5k on-axis + 20k sparse = 75k pool
- **Configs**: 256 grid corners (N x R_mean x L at 4 currents) + 244 random + 11 default = 511
- **Points per config**: 1500 general + 500 boundary-enriched = 2000

### v5/v6 data (1.02M samples, same count)
- Same spatial pool and configs as v4
- **Added**: mid-grid parameter configs (N=20,40,65 x R=10,17 x L=22,38,52) for L4 coverage
- **Added**: analytical gradient targets (dBz/dz, dBz/dr) via vectorized finite differences
- **Added**: vectorized batch computation (~50x faster data generation)

## 5. Training History

### v1: Current range too narrow
- **Current range**: [0.5, 20] A
- **Result**: 97% error at operational currents (RLC peak ~320A)
- **Root cause**: Extrapolation failure -- model never saw currents above 20A

### v2: Double-normalization bug
- **Current range**: Fixed to [0.5, 4000] A
- **Result**: Still 97% error
- **Root cause**: `output_scale` double-normalization bug. Targets were divided by
  output_scale during preprocessing, then model output was multiplied by output_scale
  again in forward(), squaring the scaling.

### v3: B/I normalization fix (first passing model)
- **Key change**: Model learns B/I (T/A) not B(T). Collapses dynamic range.
- **Training**: 100k steps, batch_size=32768, lr=1e-3 cosine to 1e-6
- **Loss weights**: data=1.0, curl=0.1, div=0.1, BC=0.01
- **Best loss**: 8.28e-09
- **Hardware**: Local GPU
- **Validation result**: ALL 5 LEVELS FAILED (see Section 6.1)

### v4: Improved data + physics losses + symmetry (current best)
- **Data changes**: 1.02M samples, boundary enrichment, grid corners
- **Loss weight changes**: curl 0.1->1.0, div 0.1->1.0, BC 0.01->0.1
- **New losses**: Symmetry (z-mirror + on-axis Br=0, weight=0.5), gradient self-consistency (weight=0.1)
- **Progressive ramp**: Physics weights scale from 0.1x to 1.0x over first 20k steps
- **Training**: 200k steps, batch_size=131072, PDE batch=8192, BC/sym batch=4096
- **Best loss**: 1.70e-08
- **Hardware**: Remote RTX 5090 (32GB VRAM)
- **Validation result**: 3 of 5 levels pass (see Section 6.2)

### v5: Aggressive gradient supervision (REGRESSED)
- **Key change**: Direct gradient supervision from analytical dBz/dz, dBz/dr targets
- **Loss weights**: div 1.0->5.0, sym 0.5->2.0, grad 0.1->1.0 (against analytical targets)
- **Gradient loss**: Every 2nd step (was every 4th, self-consistency only)
- **Training**: 300k steps, batch_size=131072
- **Best loss**: 1.79e-07 (higher than v4 due to gradient loss dominating)
- **Hardware**: Remote RTX 5090
- **Validation result**: REGRESSED -- only 2/5 pass (see Section 6.3)
- **Root cause**: Gradient supervision weight (1.0) too high. Analytical gradient targets
  have extreme values near coil winding singularities (dBz/dr up to 877 T/mm/A). The model
  spent capacity fitting noisy gradient outliers at the expense of field accuracy. L1 mean
  error tripled (0.6% -> 1.6%), L4 coverage dropped (75% -> 61%), div(B) doubled.
- **Lesson**: Gradient supervision must be gentle (low weight + clipped targets) to act as
  a regularizer, not a primary loss.

### v6: Balanced gradient supervision (TRAINING)
- **Changes from v5**: grad weight 1.0->0.05, div 5.0->2.0, sym 2.0->1.0
- **Gradient targets**: Clipped to p0.5/p99.5 to remove singularity outliers
- **Gradient loss**: Every 4th step
- **Training**: 300k steps, batch_size=131072
- **Hardware**: Remote RTX 5090
- **Status**: Training in progress

## 6. Validation Results

### 6.1. v3 Checkpoint (baseline -- all failed)

Validation run: 2026-03-18, step=100000, loss=8.28e-09

#### Level 1: Field Accuracy -- FAIL
Grid: r in [0.1, 60] mm x z in [-80, 80] mm (12,800 points per current)

| Current | Bz mean err | Bz max err | Result |
|---------|-------------|------------|--------|
| I=1A | 0.53% | 22.15% | FAIL |
| I=10A | 0.53% | 22.20% | FAIL |
| I=100A | 0.53% | 22.75% | FAIL |
| I=500A | 0.56% | 24.94% | FAIL |
| I=1000A | 0.59% | 27.03% | FAIL |
| I=3000A | 0.72% | 26.88% | FAIL |

#### Level 2: Gradient Accuracy -- FAIL

| Component | Mean err | P95 err | Max err |
|-----------|----------|---------|---------|
| dBr_dr | 0.36% | 0.14% | 100.22% |
| dBr_dz | 0.44% | 0.28% | 99.65% |
| dBz_dr | 0.17% | 0.16% | 109.94% |
| dBz_dz | 0.63% | 0.28% | 100.35% |

#### Level 3: Force Accuracy -- FAIL

| Position | Peak F err | Pearson r | Zero-cross err | Result |
|----------|------------|-----------|----------------|--------|
| r=0mm | 4.2% | 0.9987 | 39.78 mm | FAIL |
| r=5mm | 0.2% | 0.9988 | 0.45 mm | PASS |

#### Level 4: Design Space -- FAIL
- Configs with center Bz error < 5%: **62.5%** (need >= 95%)
- Worst center error: **29.55%** (need < 20%)

#### Level 5: Physics Consistency -- FAIL (0/3)

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| div(B) = 0 | 0.037459 | < 0.01 | FAIL |
| Axial symmetry | 0.019203 | < 0.01 | FAIL |
| z-symmetry | 0.026788 | < 0.02 | FAIL |

---

### 6.2. v4 Checkpoint (current best -- 3/5 pass)

Validation run: 2026-03-18, step=200000, loss=3.04e-08

#### Level 1: Field Accuracy -- PASS

| Current | Bz mean err | Bz max err | Result |
|---------|-------------|------------|--------|
| I=1A | 0.67% | 19.46% | PASS |
| I=10A | 0.67% | 19.46% | PASS |
| I=100A | 0.66% | 19.45% | PASS |
| I=500A | 0.61% | 19.40% | PASS |
| I=1000A | 0.57% | 19.31% | PASS |
| I=3000A | 0.58% | 18.79% | PASS |

#### Level 2: Gradient Accuracy -- FAIL

| Component | Mean err | P95 err | Max err |
|-----------|----------|---------|---------|
| dBr_dr | 0.34% | 0.06% | 100.08% |
| dBr_dz | 0.42% | 0.11% | 99.87% |
| dBz_dr | 0.18% | 0.15% | 112.59% |
| dBz_dz | 0.62% | 0.18% | 99.84% |

#### Level 3: Force Accuracy -- PASS

| Position | Peak F err | Pearson r | Zero-cross err | Result |
|----------|------------|-----------|----------------|--------|
| r=0mm | 1.9% | 0.9986 | 0.65 mm | PASS |
| r=5mm | 3.2% | 0.9970 | 0.22 mm | PASS |

#### Level 4: Design Space -- FAIL (improved)
- Configs with center Bz error < 5%: **75.0%** (need >= 95%, was 62.5%)
- Worst center error: **13.23%** (need < 20%, was 29.55% -- now passes worst-case)

#### Level 5: Physics Consistency -- FAIL (2/3 pass)

| Check | v3 value | v4 value | Threshold | Result |
|-------|----------|----------|-----------|--------|
| div(B) = 0 | 0.037459 | **0.019185** | < 0.01 | FAIL |
| Axial symmetry | 0.019203 | **0.002932** | < 0.01 | **PASS** |
| z-symmetry | 0.026788 | **0.012165** | < 0.02 | **PASS** |

---

### 6.3. v5 Checkpoint (regressed -- 2/5 pass)

Validation run: 2026-03-18, step=300000, loss=1.79e-07

**Root cause of regression**: Gradient supervision weight (1.0) too aggressive. Model
prioritized fitting noisy analytical gradient targets near singularities over field accuracy.

#### Level 1: Field Accuracy -- FAIL (regressed from v4 PASS)

| Current | Bz mean err | Bz max err | Result |
|---------|-------------|------------|--------|
| I=1A | 1.62% | 30.62% | FAIL |
| I=10A | 1.62% | 30.59% | FAIL |
| I=100A | 1.64% | 30.39% | FAIL |
| I=500A | 1.75% | 34.01% | FAIL |
| I=1000A | 1.62% | 38.80% | FAIL |
| I=3000A | 2.15% | 32.63% | FAIL |

Mean errors tripled (0.6% -> 1.6%), max errors nearly doubled (19% -> 31-39%).

#### Level 2: Gradient Accuracy -- FAIL (similar to v4)

| Component | Mean err | P95 err | Max err |
|-----------|----------|---------|---------|
| dBr_dr | 0.36% | 0.15% | 100.21% |
| dBr_dz | 0.45% | 0.33% | 99.42% |
| dBz_dr | 0.18% | 0.18% | 105.47% |
| dBz_dz | 0.65% | 0.30% | 99.56% |

Despite heavy gradient supervision, max errors unchanged (singularities are fundamental).

#### Level 3: Force Accuracy -- PASS

| Position | Peak F err | Pearson r | Zero-cross err | Result |
|----------|------------|-----------|----------------|--------|
| r=0mm | 11.6% | 0.9975 | 0.19 mm | PASS |
| r=5mm | 9.4% | 0.9945 | 0.17 mm | PASS |

Peak errors higher than v4 (1.9/3.2% -> 11.6/9.4%) but still within threshold.

#### Level 4: Design Space -- FAIL (regressed)
- Configs with center Bz error < 5%: **60.9%** (was 75.0%)
- Worst center error: **32.87%** (was 13.23%)

#### Level 5: Physics Consistency -- FAIL (2/3 pass, div regressed)

| Check | v4 value | v5 value | Threshold | Result |
|-------|----------|----------|-----------|--------|
| div(B) = 0 | 0.019185 | **0.037530** | < 0.01 | FAIL |
| Axial symmetry | 0.002932 | **0.006640** | < 0.01 | PASS |
| z-symmetry | 0.012165 | **0.013792** | < 0.02 | PASS |

div(B) doubled despite 5x weight increase -- gradient loss competed with div loss.

## 7. Remaining Failure Analysis

### Level 2: Gradient max errors at singularities
The ~100% max gradient errors occur at isolated grid points near the coil winding surface
(r ~ R_mean = 15mm, z ~ +/-L/2 = +/-15mm). The analytical field has a near-discontinuity
from the current sheet model. A smooth PINN cannot reproduce this. P95 errors (< 0.3%)
confirm accuracy everywhere else. Level 3 (force) passing confirms these points don't
affect integrated quantities. The max threshold of 50% may be unreachable.

### Level 4: Design space corner coverage
~25% of the 64 grid configs still exceed 5% center error. These are configs where geometry
is extreme (e.g. N=80 in L=15mm = very tight winding, or N=10 in L=60mm = very sparse).

### Level 5: Divergence-free residual
div(B) = 0.019 (v4) vs 0.01 threshold. Concentrated near coil boundaries where spatial
gradients are steepest. v5 showed that increasing div weight alone doesn't help if it
competes with other losses.

## 8. Known Limitations

1. **Coil winding singularities**: Discrete current loop model creates near-singularities
   at (r=R_mean, z=loop_position). PINN cannot reproduce these.
2. **Extrapolation**: Accuracy degrades outside training bounds (N < 10 or > 80,
   R_mean < 6 or > 20 mm, L < 15 or > 60 mm, I > 4000 A).
3. **Gradient amplification**: Force uses autograd gradients -- field errors amplified.
4. **Linear B-I assumption**: B/I normalization breaks with ferromagnetic saturation.

## 9. Integration

- **Kit Extension**: `omni.marble.coaster` loads the PINN via `torch.load()` with
  `weights_only=False` (needed for PhysicsNeMo custom layers)
- **Dependencies**: torch and nvidia-physicsnemo installed via `omni.kit.pipapi`
  on first launch. Added `nvidia-physicsnemo` to `pyproject.toml` for standalone use.
- **Force path**: `PINNForceComputer` calls `model.forward()` with autograd enabled,
  computes (chi*V/mu0)(B.grad)B, applies via PhysX `apply_force_at_pos`
- **`current_normalized` flag**: Stored as a buffer in the checkpoint. When True,
  model outputs B/I -- all inference code multiplies by I before computing force
- **`warp_bfield_solver.py`**: Fixed to handle `current_normalized` buffer and I-scaling

### v7 ("design-space"): Audit-driven retraining (superseded by v8)
- **Key changes**: Active failure mining (N=10/R=8 families), gentle gradient supervision
  (weight 0.03, clipped targets), data-loss checkpoint selection, periodic saves
- **Data**: 1.42M samples (711 configs including failure-mined families)
- **Loss weights**: curl=1.0, div=1.5, BC=0.1, sym=0.5, grad=0.03
- **Training**: 300k steps, batch_size=131072
- **Best data loss**: 1.39e-08
- **Hardware**: Remote RTX 5090
- **Validation result**: L2+L3+L4 pass, L1 max and L5 div still fail (see Section 6.4)
- **Deployed as**: `pinn_designspace.pt` (2026-03; retired 2026-07 — was
  byte-identical to `pinn_best.pt`, which is now the single production name)

## 10. Checkpoint Comparison and Dual-Baseline Decision (v7 era, retired)

### Checkpoint sweep (v7 training run)

All periodic checkpoints were evaluated to determine whether an intermediate
checkpoint could achieve v7's L4 coverage while preserving lower L1 max error:

| Checkpoint | L1 mean | L1 P99 | L1 max | L4 %<5 | L4 worst | L3 pk err | L5 div |
|-----------|---------|--------|--------|--------|----------|-----------|--------|
| step 50k  | 1.25% | 11.02% | 71.0% | 64% | 15.4% | 7.1% | 0.024 |
| step 100k | 0.75% | 6.57% | 58.1% | 78% | 10.4% | 0.3% | 0.019 |
| step 150k | 0.78% | 4.41% | 42.7% | 92% | 5.4% | 4.9% | 0.016 |
| step 200k | 0.51% | 3.83% | 40.3% | 100% | 4.9% | 5.0% | 0.015 |
| step 250k | 0.58% | 3.49% | 38.0% | 98% | 5.4% | 6.1% | 0.014 |
| **final** | **0.50%** | **3.39%** | **37.2%** | **100%** | **4.8%** | **6.0%** | **0.014** |

**Conclusion**: L4 coverage and L1 max trade off monotonically. No intermediate
checkpoint achieves both v4-style L1 max (<20%) and v7-style L4 completeness (100%).
This is a real tradeoff in the learned solution, not a checkpoint-selection issue.

### Dual-baseline deployment decision (RETIRED 2026-07-02)

**Retirement note**: the strategy below never had a working second leg. The
original v4 checkpoint was lost during the v5-v7 iteration cycle, and the
recipe retrain produced L1 max = 47% — worse at pointwise field accuracy
than the design-space model it was meant to back up. `pinn_fieldaccuracy.pt`
and its training scripts were deleted (recoverable from git history), and
v8 (Section 12) improved L1 max to 27.8% anyway, weakening the case for a
separate field baseline. Preserved as written for the historical record:

Two named baselines are maintained:

**`pinn_designspace.pt`** (v7 final) -- primary production model
- Use for: Kit integration, force computation, design exploration
- Passes: L2 (P99=3.7%), L3 (force), L4 (100% design space)
- L1 max error (37%) concentrated in singular-boundary regions; does not prevent
  Level 3 force accuracy from passing
- L5 div(B) = 0.014, close to 0.01 threshold

**`pinn_fieldaccuracy.pt`** (v4, needs retraining) -- conservative field baseline
- Use for: applications requiring strict pointwise field accuracy near boundaries
- Would pass: L1 (max <20%), L3 (force)
- Known gap: L4 at 75% (design space incomplete)
- Note: original v4 checkpoint was overwritten during v5-v7 iteration cycle.
  Retraining with the v4 recipe is required to restore this baseline.

### Audit findings addressed

| Audit finding | Action taken | Result |
|--------------|-------------|--------|
| Level 2 mis-specified (max at singularities) | Changed to P99 < 10% | **L2 now PASS** (P99=3.7%) |
| Level 4 under-covered by data | Active failure mining of N=10/R=8 families | **L4 100%** (was 75%) |
| Training loss not reliable for selection | Checkpoint selected by data loss, periodic saves | Better proxy + post-hoc sweep |
| Gradient supervision too aggressive (v5) | Weight 0.03 with clipped targets | No regression from v4 baseline |
| Process optimizing for wrong things | Dual-baseline framing, not single pass/fail | Honest tradeoff documented |

### Next experiment

The remaining L5 div(B) gap (0.014 vs 0.01) and the L1 boundary max error are both
candidates for a **physics-by-construction** approach: deriving B_r and B_z from A_phi
via autograd (B_r = -dA/dz, B_z = (1/r)d(rA)/dr) would guarantee div(B)=0 exactly.
This is the recommended next structural improvement, not further loss weight tuning.

**Done — implemented as v8, see Section 12.**

## 11. Training Infrastructure

- **Local machine**: Windows 11 (development, validation), RTX 5060 Ti
- **Remote training**: Windows SSH to RTX 5090 (32GB VRAM) at 10.10.1.187
  (ICMP firewalled — preflight with `ssh <host> "echo ok"`, not ping)
- **Workflow**: `scripts/remote_train.sh` or manual SCP + SSH
- **Data generation**: Vectorized batch computation (~50x faster than per-point loop)
- **v4 training time**: ~15min for 200k steps at batch_size=131072 on RTX 5090
- **v5 training time**: ~22min for 300k steps at batch_size=131072 on RTX 5090
- **v8 training time**: ~4.7h for 300k steps at batch_size=131072 on RTX 5090
  (the data loss backprops through an autograd derivative, so every step is a
  double backward — ~13x the v7 step cost)

## 12. v8: Physics by Construction (2026-07-02, production)

### Architecture change

The network head was reduced to a single scalar f(r, z, I, N, R_mean, L)
with the ansatz **A_phi = r · f**. The field is derived inside forward()
via autograd:

    B_r = -dA/dz         = -r · df/dz
    B_z = (1/r) d(rA)/dr = 2f + r · df/dr

Consequences, exact by architecture (not by training):
- div(B) = 0 (Level 5's div check is now structural)
- curl consistency B = curl(A)
- A_phi(0, z) = 0 and B_r(0, z) = 0 (the r·f parameterization removes the
  1/r axis singularity — no l'Hopital special case needed)

Losses removed as now-redundant: curl, div, on-axis boundary conditions,
and analytical gradient supervision (third-order through the derived field).
Remaining: data MSE on derived (Br/I, Bz/I), far-field decay (0.1),
z-mirror symmetry (0.5), progressive ramp. Checkpoint selection by data
loss with periodic saves, as in v7. Model/loader code:
`scripts/pinn_loader.py` (BFieldPINNDerived; the loader transparently
supports legacy and derived checkpoints via the `derived_b` flag).

Force gradients are now SECOND derivatives of f (double backward). The Kit
per-step cost was offset by merging the previous two PINN calls per physics
step (get_Bz + compute_force) into one graph build.

ONNX export was dropped (autograd inside forward is not traceable; nothing
consumed it).

### v8 training run

- **Data**: regenerated 1.42M samples / 711 configs (v7 failure-mining recipe)
- **Training**: 300k steps, batch 131072, RTX 5090, ~4.7h
- **Best data loss**: 1.57e-08 (at step 250k)

### Checkpoint sweep (v8)

| Checkpoint | L1 mean | L1 p99 | L1 max | L2 p99 | L3 pk / r | L4 %<5 | L4 worst | L5 div |
|-----------|---------|--------|--------|--------|-----------|--------|----------|--------|
| step 50k  | 8.33% | 61.6% | 82.5% | 4.52% | 94.1% / 0.497 | 4.7% | 83.2% | 0.00000 |
| step 100k | 5.29% | 28.1% | 72.6% | 4.16% | 36.9% / 0.926 | 6.2% | 66.6% | 0.00000 |
| step 150k | 1.95% | 5.9% | 35.0% | 3.47% | 1.6% / 0.997 | 71.9% | 21.9% | 0.00000 |
| step 200k | 1.35% | 4.8% | 29.0% | 3.17% | 1.7% / 0.996 | 87.5% | 9.4% | 0.00000 |
| **step 250k** | **0.86%** | **2.8%** | **27.8%** | **3.05%** | **3.0% / 0.995** | **95.3%** | **7.75%** | **0.00000** |
| final 300k | 0.77% | 2.9% | 28.0% | 3.04% | 4.2% / 0.994 | 92.2% | 9.0% | 0.00000 |

div(B) is identically zero for every checkpoint — structural, as designed.
The v7 L1-max-vs-L4 tradeoff persists (final 300k has better L1 mean but
dips below the 95% L4 gate); **step 250k** is the only checkpoint passing
every gate criterion and was promoted.

### Adoption gate (all criteria met by step-250k)

| Criterion | Required | v8 step-250k | v7 (previous) |
|-----------|----------|--------------|----------------|
| L3 force, both radii | peak<15%, r>0.95, zc<2mm | 2.7%/3.0%, 0.998/0.995, <0.4mm | 7.5%/6.0% |
| L4 design space | >=95% under 5%, worst<20% | 95.3%, 7.75% | 100%, 4.8% |
| L1 field | max <= 37.2% (no worse than v7), mean<5% | 27.8%, 0.86% | 37.2%, 0.50% |
| L5 physics | all pass | div=0 exact, axial+z-sym pass | div FAIL (0.0138) |
| Optimizer ranking | Spearman >= 0.99 | **1.000** (Kendall 0.992, max shift +/-1) | 0.9997 |

Promoted 2026-07-02: `models/pinn_checkpoint/pinn_best.pt` = v8 step-250k
(`model_version=8`, `derived_b=True`). v7 remains recoverable from git
history (commit 9a0c15d's parent).

### First Kit/PhysX launch artifact (300V)

Run via the extension's scripted autorun
(`--/exts/omni.marble.coaster/autorun=true --/exts/omni.marble.coaster/autorunVoltage=300`):

- Approach velocity 208 mm/s (vel_in gate pair); coil fired at entry gate
- 300V / 470uF = 21.1J stored, underdamped (zeta=0.339)
- MOSFET pulse cut at the cutoff gate
- **Gate-measured exit velocity 937.5 mm/s -> 4.5x boost** (README's
  predicted ~4.5x at 300V)
- Artifact: `results/trajectories/kit_launch_300V_470uF_20260702_174018.csv`
  (528 steps at 500 Hz, with metadata headers) — the first trajectory from
  the real Kit/PhysX/PINN path; all earlier committed trajectories came
  from the headless numpy integrators.
- Headless cross-check (`validate_kit_simulation.py --field pinn
  --voltage 300`): 4.2x boost — ratios agree within ~7%. Absolute
  velocities differ through the simpler sphere-bounce track collision
  (approach 149 vs 208 mm/s). Note the two pipelines report I_peak
  differently: the extension logs the undamped amplitude V/(omega_d*L)
  (1963A at 300V) while rlc_circuit reports the true damped peak (1187A);
  same circuit, different definitions (cross-checked in
  tests/test_coil_physics.py).

### Remaining known gaps

1. L1 max error (27.8%) at winding singularities — fundamental to the
   current-sheet analytical model vs a smooth network; informational.
2. All validation is against the analytical model; no physical-hardware
   validation exists yet (the sim-to-real loop of the original plan).
3. `coil_physics.py` (Kit) and `rlc_circuit.py` (headless) duplicate RLC
   physics — cross-checked by tests, unification is tracked tech debt.


---

### Validation run: 2026-03-18 22:38:41 (step=300000, loss=2.02e-08, 3/5 pass)

| Check | Result |
|-------|--------|
| Level 1: Field accuracy | FAIL |
| Level 2: Gradient accuracy | PASS |
| Level 3: Force accuracy | PASS |
| Level 4: Design space | PASS |
| Level 5: Physics consistency | FAIL |

**Level 1: Field Accuracy**

| Current | Bz mean err | Bz max err | Result |
|---------|-------------|------------|--------|
| I=1A | 0.50% | 37.68% | FAIL |
| I=10A | 0.50% | 37.63% | FAIL |
| I=100A | 0.50% | 37.17% | FAIL |
| I=500A | 0.49% | 35.49% | FAIL |
| I=1000A | 0.49% | 34.26% | FAIL |
| I=3000A | 0.62% | 38.98% | FAIL |

**Level 2: Gradient Accuracy** (I=318A)

| Component | Mean err | P95 err | Max err |
|-----------|----------|---------|---------|  
| dBr_dr | 0.35% | 0.08% | 100.02% |
| dBr_dz | 0.43% | 0.16% | 99.85% |
| dBz_dr | 0.16% | 0.23% | 107.39% |
| dBz_dz | 0.62% | 0.22% | 100.01% |

**Level 3: Force Accuracy**

| Position | Peak F err | Pearson r | Zero-cross err | Result |
|----------|------------|-----------|----------------|--------|
| r=0mm | 7.5% | 0.9941 | 0.68 mm | PASS |
| r=5mm | 6.0% | 0.9980 | 0.02 mm | PASS |

**Level 4: Design Space**
- Configs under 5%: 100.0% (need 95%)
- Worst: 4.80% (need <20%)

**Level 5: Physics Consistency**
- div_B: mean_normalized=0.013843 [FAIL]
- axial_symmetry: max_Br_over_Bz=0.003147 [PASS]
- z_symmetry: max_asymmetry=0.005018 [PASS]


---

### Validation run: 2026-03-19 10:43:05 (step=200000, loss=2.64e-08, 2/5 pass)

| Check | Result |
|-------|--------|
| Level 1: Field accuracy | FAIL |
| Level 2: Gradient accuracy | PASS |
| Level 3: Force accuracy | PASS |
| Level 4: Design space | FAIL |
| Level 5: Physics consistency | FAIL |

**Level 1: Field Accuracy**

| Current | Bz mean err | Bz max err | Result |
|---------|-------------|------------|--------|
| I=1A | 0.60% | 47.01% | FAIL |
| I=10A | 0.59% | 47.00% | FAIL |
| I=100A | 0.59% | 46.91% | FAIL |
| I=500A | 0.60% | 46.52% | FAIL |
| I=1000A | 0.61% | 46.02% | FAIL |
| I=3000A | 0.70% | 44.19% | FAIL |

**Level 2: Gradient Accuracy** (I=318A)

| Component | Mean err | P95 err | Max err |
|-----------|----------|---------|---------|  
| dBr_dr | 0.35% | 0.08% | 100.03% |
| dBr_dz | 0.43% | 0.18% | 99.95% |
| dBz_dr | 0.16% | 0.27% | 106.22% |
| dBz_dz | 0.62% | 0.24% | 99.91% |

**Level 3: Force Accuracy**

| Position | Peak F err | Pearson r | Zero-cross err | Result |
|----------|------------|-----------|----------------|--------|
| r=0mm | 0.0% | 0.9966 | 0.17 mm | PASS |
| r=5mm | 2.2% | 0.9977 | 0.03 mm | PASS |

**Level 4: Design Space**
- Configs under 5%: 90.6% (need 95%)
- Worst: 11.69% (need <20%)

**Level 5: Physics Consistency**
- div_B: mean_normalized=0.021017 [FAIL]
- axial_symmetry: max_Br_over_Bz=0.002915 [PASS]
- z_symmetry: max_asymmetry=0.007274 [PASS]


---

### Validation run: 2026-07-02 12:43:08 (step=300000, loss=2.02e-08, 3/5 pass)

| Check | Result |
|-------|--------|
| Level 1: Field accuracy | FAIL |
| Level 2: Gradient accuracy | PASS |
| Level 3: Force accuracy | PASS |
| Level 4: Design space | PASS |
| Level 5: Physics consistency | FAIL |

**Level 1: Field Accuracy**

| Current | Bz mean err | Bz max err | Result |
|---------|-------------|------------|--------|
| I=1A | 0.50% | 37.68% | FAIL |
| I=10A | 0.50% | 37.63% | FAIL |
| I=100A | 0.50% | 37.17% | FAIL |
| I=500A | 0.49% | 35.49% | FAIL |
| I=1000A | 0.49% | 34.26% | FAIL |
| I=3000A | 0.62% | 38.98% | FAIL |

**Level 2: Gradient Accuracy** (I=318A)

| Component | Mean err | P95 err | Max err |
|-----------|----------|---------|---------|  
| dBr_dr | 0.35% | 0.08% | 100.02% |
| dBr_dz | 0.43% | 0.16% | 99.85% |
| dBz_dr | 0.16% | 0.23% | 107.39% |
| dBz_dz | 0.62% | 0.22% | 100.01% |

**Level 3: Force Accuracy**

| Position | Peak F err | Pearson r | Zero-cross err | Result |
|----------|------------|-----------|----------------|--------|
| r=0mm | 7.5% | 0.9941 | 0.68 mm | PASS |
| r=5mm | 6.0% | 0.9980 | 0.02 mm | PASS |

**Level 4: Design Space**
- Configs under 5%: 100.0% (need 95%)
- Worst: 4.80% (need <20%)

**Level 5: Physics Consistency**
- div_B: mean_normalized=0.013843 [FAIL]
- axial_symmetry: max_Br_over_Bz=0.003147 [PASS]
- z_symmetry: max_asymmetry=0.005018 [PASS]


---

### Validation run: 2026-07-02 17:32:11 (step=300000, loss=2.80e-08, 3/5 pass)

| Check | Result |
|-------|--------|
| Level 1: Field accuracy | FAIL |
| Level 2: Gradient accuracy | PASS |
| Level 3: Force accuracy | PASS |
| Level 4: Design space | FAIL |
| Level 5: Physics consistency | PASS |

**Level 1: Field Accuracy**

| Current | Bz mean err | Bz max err | Result |
|---------|-------------|------------|--------|
| I=1A | 0.82% | 28.30% | FAIL |
| I=10A | 0.82% | 28.28% | FAIL |
| I=100A | 0.77% | 28.04% | FAIL |
| I=500A | 0.61% | 26.90% | FAIL |
| I=1000A | 0.59% | 25.73% | FAIL |
| I=3000A | 1.24% | 29.25% | FAIL |

**Level 2: Gradient Accuracy** (I=318A)

| Component | Mean err | P95 err | Max err |
|-----------|----------|---------|---------|  
| dBr_dr | 0.34% | 0.05% | 100.02% |
| dBr_dz | 0.42% | 0.12% | 99.78% |
| dBz_dr | 0.16% | 0.15% | 109.26% |
| dBz_dz | 0.61% | 0.15% | 100.02% |

**Level 3: Force Accuracy**

| Position | Peak F err | Pearson r | Zero-cross err | Result |
|----------|------------|-----------|----------------|--------|
| r=0mm | 0.7% | 0.9981 | 0.19 mm | PASS |
| r=5mm | 4.2% | 0.9943 | 0.33 mm | PASS |

**Level 4: Design Space**
- Configs under 5%: 92.2% (need 95%)
- Worst: 8.99% (need <20%)

**Level 5: Physics Consistency**
- div_B: mean_normalized=0.000000 [PASS]
- axial_symmetry: max_Br_over_Bz=0.002266 [PASS]
- z_symmetry: max_asymmetry=0.012700 [PASS]


---

### Validation run: 2026-07-02 17:33:29 (step=250000, loss=1.57e-08, 4/5 pass)

| Check | Result |
|-------|--------|
| Level 1: Field accuracy | FAIL |
| Level 2: Gradient accuracy | PASS |
| Level 3: Force accuracy | PASS |
| Level 4: Design space | PASS |
| Level 5: Physics consistency | PASS |

**Level 1: Field Accuracy**

| Current | Bz mean err | Bz max err | Result |
|---------|-------------|------------|--------|
| I=1A | 0.92% | 28.04% | FAIL |
| I=10A | 0.91% | 28.03% | FAIL |
| I=100A | 0.86% | 27.85% | FAIL |
| I=500A | 0.70% | 26.83% | FAIL |
| I=1000A | 0.71% | 25.61% | FAIL |
| I=3000A | 1.46% | 29.67% | FAIL |

**Level 2: Gradient Accuracy** (I=318A)

| Component | Mean err | P95 err | Max err |
|-----------|----------|---------|---------|  
| dBr_dr | 0.34% | 0.05% | 100.01% |
| dBr_dz | 0.42% | 0.11% | 99.78% |
| dBz_dr | 0.16% | 0.15% | 109.16% |
| dBz_dz | 0.61% | 0.16% | 99.99% |

**Level 3: Force Accuracy**

| Position | Peak F err | Pearson r | Zero-cross err | Result |
|----------|------------|-----------|----------------|--------|
| r=0mm | 2.7% | 0.9981 | 0.33 mm | PASS |
| r=5mm | 3.0% | 0.9945 | 0.39 mm | PASS |

**Level 4: Design Space**
- Configs under 5%: 95.3% (need 95%)
- Worst: 7.75% (need <20%)

**Level 5: Physics Consistency**
- div_B: mean_normalized=0.000000 [PASS]
- axial_symmetry: max_Br_over_Bz=0.002320 [PASS]
- z_symmetry: max_asymmetry=0.017698 [PASS]


---

### Validation run: 2026-07-02 17:47:03 (step=250000, loss=1.57e-08, 4/5 pass)

| Check | Result |
|-------|--------|
| Level 1: Field accuracy | FAIL |
| Level 2: Gradient accuracy | PASS |
| Level 3: Force accuracy | PASS |
| Level 4: Design space | PASS |
| Level 5: Physics consistency | PASS |

**Level 1: Field Accuracy**

| Current | Bz mean err | Bz max err | Result |
|---------|-------------|------------|--------|
| I=1A | 0.92% | 28.04% | FAIL |
| I=10A | 0.91% | 28.03% | FAIL |
| I=100A | 0.86% | 27.85% | FAIL |
| I=500A | 0.70% | 26.83% | FAIL |
| I=1000A | 0.71% | 25.61% | FAIL |
| I=3000A | 1.46% | 29.67% | FAIL |

**Level 2: Gradient Accuracy** (I=318A)

| Component | Mean err | P95 err | Max err |
|-----------|----------|---------|---------|  
| dBr_dr | 0.34% | 0.05% | 100.01% |
| dBr_dz | 0.42% | 0.11% | 99.78% |
| dBz_dr | 0.16% | 0.15% | 109.16% |
| dBz_dz | 0.61% | 0.16% | 99.99% |

**Level 3: Force Accuracy**

| Position | Peak F err | Pearson r | Zero-cross err | Result |
|----------|------------|-----------|----------------|--------|
| r=0mm | 2.7% | 0.9981 | 0.33 mm | PASS |
| r=5mm | 3.0% | 0.9945 | 0.39 mm | PASS |

**Level 4: Design Space**
- Configs under 5%: 95.3% (need 95%)
- Worst: 7.75% (need <20%)

**Level 5: Physics Consistency**
- div_B: mean_normalized=0.000000 [PASS]
- axial_symmetry: max_Br_over_Bz=0.002320 [PASS]
- z_symmetry: max_asymmetry=0.017698 [PASS]
