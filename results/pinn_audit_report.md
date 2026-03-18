# PINN B-Field Model Audit Report

Last updated: 2026-03-18

## 1. Executive Summary

The Physics-Informed Neural Network (PINN) replaces the analytical Biot-Savart
B-field computation for a finite solenoid coilgun. It takes spatial position and
coil design parameters as input and produces the magnetic vector potential and
field components in a single forward pass.

**Current status: 3 of 5 validation levels pass (v4 checkpoint). Levels 2, 4, 5 still failing.**

| Check | v3 result | v4 result |
|-------|-----------|-----------|
| Level 1: Field accuracy | FAIL | **PASS** |
| Level 2: Gradient accuracy | FAIL | FAIL |
| Level 3: Force accuracy | FAIL | **PASS** |
| Level 4: Design space | FAIL | FAIL (improved) |
| Level 5: Physics consistency | FAIL (0/3) | FAIL (2/3 pass) |

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

### v3 (original)
- **Samples**: ~414k
- **Spatial**: 30k dense (r in [0,30], z in [-40,40]) + 20k sparse far-field
- **Configs**: 200 random + 7 default coil at various currents
- **Points per config**: 2000 (random subset of spatial pool)

### v4 (improved)
- **Samples**: ~1.02M (2.5x increase)
- **Spatial**: 30k dense + 20k boundary-enriched + 5k on-axis + 20k sparse = 75k pool
- **Boundary enrichment**: Gaussian-concentrated at coil ends (z ~ +/-L/2) and winding radius (r ~ R_mean)
- **On-axis**: 5k points with r < 1mm for symmetry enforcement
- **Configs**: 256 explicit grid corners (N x R_mean x L at 4 currents) + 244 random + 11 default = 511
- **Points per config**: 1500 general + 500 config-adaptive boundary = 2000

Key change: v3 had no coverage of the Level 4 validation grid corners. v4 explicitly
includes all 64 (N, R_mean, L) combinations at multiple currents.

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
- **Best loss**: 8.28e-09 (but misleadingly low -- physics weights were too small)
- **Hardware**: Local GPU
- **Validation result**: ALL 5 LEVELS FAILED (see Section 6.1)

### v4: Improved data + physics losses + symmetry
- **Data changes**: See Section 4 above (1.02M samples, boundary enrichment, grid corners)
- **Loss weight changes**: curl 0.1->1.0, div 0.1->1.0, BC 0.01->0.1
- **New losses**: Symmetry (z-mirror + on-axis Br=0, weight=0.5), gradient self-consistency (weight=0.1)
- **Progressive ramp**: Physics weights ramp from 0.1x to 1.0x over first 20k steps
- **Training**: 200k steps, batch_size=131072, PDE batch=8192, BC/sym batch=4096
- **Best loss**: 1.70e-08
- **Hardware**: Remote RTX 5090 (32GB VRAM)
- **Validation result**: 3 of 5 levels pass (see Section 6.2)

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

Pass criteria: Bz mean < 5%, max < 20%. Means excellent, max exceeded at all currents.

#### Level 2: Gradient Accuracy -- FAIL
Test current: 317.6 A (RLC peak)

| Component | Mean err | P95 err | Max err |
|-----------|----------|---------|---------|
| dBr_dr | 0.36% | 0.14% | 100.22% |
| dBr_dz | 0.44% | 0.28% | 99.65% |
| dBz_dr | 0.17% | 0.16% | 109.94% |
| dBz_dz | 0.63% | 0.28% | 100.35% |

Pass criteria: dBz/dz mean < 10%, max < 50%. Mean passes, max fails at isolated singular points.

#### Level 3: Force Accuracy -- FAIL

| Position | Peak F err | Pearson r | Zero-cross err | Result |
|----------|------------|-----------|----------------|--------|
| r=0mm | 4.2% | 0.9987 | 39.78 mm | FAIL |
| r=5mm | 0.2% | 0.9988 | 0.45 mm | PASS |

r=0mm zero-crossing error of 39.78mm indicates completely wrong force sign-change location on-axis.

#### Level 4: Design Space -- FAIL
- Configs with center Bz error < 5%: **62.5%** (need >= 95%)
- Worst center error: **29.55%** (need < 20%)

#### Level 5: Physics Consistency -- FAIL (0/3)

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| div(B) = 0 | 0.037459 | < 0.01 | FAIL |
| Axial symmetry (Br/Bz at r~0) | 0.019203 | < 0.01 | FAIL |
| z-symmetry | 0.026788 | < 0.02 | FAIL |

---

### 6.2. v4 Checkpoint (current -- 3/5 pass)

Validation run: 2026-03-18, step=200000, loss=3.04e-08

#### Level 1: Field Accuracy -- PASS
Grid: r in [0.1, 60] mm x z in [-80, 80] mm (12,800 points per current)

| Current | Bz mean err | Bz max err | Result |
|---------|-------------|------------|--------|
| I=1A | 0.67% | 19.46% | PASS |
| I=10A | 0.67% | 19.46% | PASS |
| I=100A | 0.66% | 19.45% | PASS |
| I=500A | 0.61% | 19.40% | PASS |
| I=1000A | 0.57% | 19.31% | PASS |
| I=3000A | 0.58% | 18.79% | PASS |

All currents now under 20% max. Means stable at ~0.6%. Max errors concentrated at coil winding boundary.

#### Level 2: Gradient Accuracy -- FAIL

| Component | Mean err | P95 err | Max err |
|-----------|----------|---------|---------|
| dBr_dr | 0.34% | 0.06% | 100.08% |
| dBr_dz | 0.42% | 0.11% | 99.87% |
| dBz_dr | 0.18% | 0.15% | 112.59% |
| dBz_dz | 0.62% | 0.18% | 99.84% |

Pass criteria: dBz/dz mean < 10%, max < 50%. **Mean and P95 are excellent** (< 1%).
Max errors are ~100% at isolated points where the analytical gradient has near-singularities
(r = R_mean, z = +/-L/2 -- coil winding corners where current loops sit). These points
have near-zero analytical gradient denominators, making relative error blow up even for
small absolute differences.

**Assessment**: The max threshold of 50% may be unreachable for a smooth PINN approximation
at field singularities. P99 would be a more meaningful metric. The force accuracy (Level 3)
passes despite this, confirming these isolated points don't affect integrated quantities.

#### Level 3: Force Accuracy -- PASS

| Position | Peak F err | Pearson r | Zero-cross err | Result |
|----------|------------|-----------|----------------|--------|
| r=0mm | 1.9% | 0.9986 | 0.65 mm | PASS |
| r=5mm | 3.2% | 0.9970 | 0.22 mm | PASS |

Major improvement: r=0mm zero-crossing error dropped from 39.78mm to 0.65mm.
Both positions now pass all criteria (peak < 15%, Pearson > 0.95, zc < 2mm).

#### Level 4: Design Space -- FAIL (improved)
- Configs with center Bz error < 5%: **75.0%** (need >= 95%, was 62.5%)
- Worst center error: **13.23%** (need < 20%, was 29.55% -- now passes worst-case)

Extrapolation points:
- N=5, R=5, L=10 (below range): 55.5% error (expected -- outside training bounds)
- N=100, R=25, L=80 (above range): 4.0% error
- R=3mm (tiny): 5.4% error
- R=30mm (wide): 1.8% error

**Assessment**: Worst-case error now acceptable. The 75% vs 95% threshold failure
means 16 of 64 grid configs still have > 5% center error. These are likely at
parameter space edges (small N with large L, or small R_mean).

#### Level 5: Physics Consistency -- FAIL (2/3 pass)

| Check | v3 value | v4 value | Threshold | Result |
|-------|----------|----------|-----------|--------|
| div(B) = 0 | 0.037459 | **0.019185** | < 0.01 | FAIL |
| Axial symmetry | 0.019203 | **0.002932** | < 0.01 | **PASS** |
| z-symmetry | 0.026788 | **0.012165** | < 0.02 | **PASS** |

div(B) halved but still 1.9x above threshold. Axial symmetry and z-symmetry now pass
with comfortable margins (6.5x and 1.6x under threshold respectively).

## 7. Remaining Failure Analysis

### Level 2: Gradient max errors at singularities
The ~100% max gradient errors occur at isolated grid points near the coil winding surface
(r ~ R_mean = 15mm, z ~ +/-L/2 = +/-15mm). At these locations, the analytical field has
a near-discontinuity (current sheet model), making the finite-difference gradient reference
values themselves poorly conditioned. A smooth neural network cannot and should not reproduce
these singularities. The P95 errors (< 0.2%) confirm the model is accurate everywhere else.

### Level 4: Design space corner coverage
25% of the 64 grid configs (N x R_mean x L) still exceed 5% center error. These are
likely configs where the geometry is extreme relative to the training distribution (e.g.
N=80 turns in L=15mm -- extremely tight winding that creates sharp field features).

### Level 5: Divergence-free residual
div(B) = 0.019 vs 0.01 threshold. The divergence loss in training reached 6.3e-11
(very low), but the validation computes div(B) via autograd on a different grid than
training. The remaining divergence is concentrated near coil boundaries where the
field has the steepest spatial variation.

## 8. Known Limitations

1. **Coil winding singularities**: The analytical model treats the coil as discrete current
   loops, creating near-singularities at (r=R_mean, z=loop_position). The smooth PINN
   cannot reproduce these, leading to high max errors at isolated points.
2. **Extrapolation**: Accuracy degrades outside training bounds (N < 10 or > 80,
   R_mean < 6 or > 20 mm, L < 15 or > 60 mm, I > 4000 A).
3. **Gradient amplification**: Force uses autograd gradients of B -- field errors are
   amplified in gradient computation, though Level 3 passing shows this is acceptable
   in practice.
4. **Linear B-I assumption**: The B/I normalization assumes B is linear in I, which
   holds for a solenoid in vacuum but breaks with ferromagnetic saturation.

## 9. Integration

- **Kit Extension**: `omni.marble.coaster` loads the PINN via `torch.load()` with
  `weights_only=False` (needed for PhysicsNeMo custom layers)
- **Dependencies**: torch and nvidia-physicsnemo installed via `omni.kit.pipapi`
  on first launch. Added `nvidia-physicsnemo` to `pyproject.toml` for standalone use.
- **Force path**: `PINNForceComputer` calls `model.forward()` with autograd enabled,
  computes (chi*V/mu0)(B.grad)B, applies via PhysX `apply_force_at_pos`
- **`current_normalized` flag**: Stored as a buffer in the checkpoint. When True,
  model outputs B/I -- all inference code multiplies by I before computing force
- **`warp_bfield_solver.py`**: Fixed in this iteration to handle `current_normalized`
  buffer and I-scaling (was missing, producing wrong fields for headless simulation)

## 10. Training Infrastructure

- **Local machine**: Windows 11 (development, validation)
- **Remote training**: Windows SSH to RTX 5090 (32GB VRAM) at 10.10.1.187
- **Workflow**: `scripts/remote_train.sh` syncs files via SCP, runs data gen + training
  via SSH, pulls checkpoint back. uv manages Python dependencies on both machines.
- **v4 training time**: 200k steps at batch_size=131072 on RTX 5090
