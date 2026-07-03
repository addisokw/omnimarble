# OmniMarble -- Magnetic Coil Marble Coaster Digital Twin

A physics-accurate digital twin of a magnetic coil-launched marble roller coaster, built on NVIDIA's simulation stack (OpenUSD, PhysX, Kit SDK) with a Physics-Informed Neural Network (PINN) for real-time electromagnetic field computation.

## How It Works

A solenoid coil fires an RLC capacitor discharge through a coil to accelerate a ferromagnetic marble via electromagnetic force. The complete physics chain:

```
RLC Circuit (V, C, L, R)
    |
    v
Current I(t) -- underdamped oscillation with flyback diode
    |
    v
B-field (PINN: 6 inputs -> Br, Bz in single forward pass)
    |
    v
Gradient dB/dr, dB/dz (torch.autograd on PINN output)
    |
    v
Force F = (chi*V/mu0) * (B . nabla)B  -- with saturation + eddy braking
    |
    v
PhysX apply_force_at_pos() -- marble accelerates in Omniverse
```

### The PINN

The analytical Biot-Savart computation (30 loop iterations x scipy elliptic integrals) is replaced by a Physics-Informed Neural Network for real-time inference:

- **Architecture (v8, "physics by construction")**: NVIDIA PhysicsNeMo FullyConnected, 6x256 hidden, SiLU, skip connections. The network outputs a single scalar f; the vector potential is A_phi = r*f and the field is derived inside forward() via autograd: B_r = -r*df/dz, B_z = 2f + r*df/dr. This makes div(B)=0, curl consistency, and the on-axis boundary conditions **exact identities of the architecture**, not trained approximations.
- **Inputs**: (r, z, I, N, R_mean, L) -- spatial position + coil design parameters
- **Outputs**: (A_phi, B_r, B_z) -- same column contract as earlier direct-B models; the shared loader (`scripts/pinn_loader.py`) handles both generations via the checkpoint's `derived_b` flag
- **Key feature**: Model learns B/I (T/A), collapsing dynamic range across the full current range [0.5, 4000] A
- **Training**: 1.42M samples from analytical Biot-Savart (711 configs incl. failure-mined families), losses: data + far-field decay + z-mirror symmetry
- **Validation**: 5-level test suite -- field accuracy, gradient accuracy, force accuracy, design space coverage, physics consistency
- **Force**: needs field gradients, which are second derivatives of f (double backward through the forward-pass graph)

### RLC Circuit

The coil is driven by a capacitor discharge (underdamped RLC):

```
I(t) = (V0 / omega_d * L) * exp(-alpha * t) * sin(omega_d * t)
```

With coupled electromechanical ODE (back-EMF from marble motion), RK4 sub-stepping at 10us resolution, and flyback diode clamping.

Default circuit: 470uF capacitor, 50V charge, L=12.4uH, R_dc=0.080 ohm (config `resistance_ohm`) + 0.01 ESR + 0.02 wiring = R_total~=0.110 ohm -> underdamped (zeta=0.34). Peak current: 327A undamped amplitude V/(omega_d*L); ~198A true damped peak (the two pipelines historically reported different definitions -- see tests/test_coil_physics.py).

### Force Model

Force on a ferromagnetic sphere in a non-uniform field:

```
F = (chi_eff * V / mu0) * (B . nabla)B
```

Where:
- chi_eff = 3.0 (demagnetization-corrected susceptibility for a sphere)
- V = marble volume (mm^3)
- mu0 = 4*pi*1e-4 T*mm/A (mm-scaled)
- B and gradients from PINN autograd

Additional physics:
- **Saturation**: When B_internal > B_sat (1.8T), switches to saturated magnetization model
- **Eddy current braking**: F_eddy proportional to dB/dt and marble conductivity
- **Wire heating**: I^2*R thermal model with copper temperature coefficient

### Units

All computations use mm-based units:
- Distance: mm
- Current: A
- Field: T (Tesla)
- Force: mN (milliNewtons) -- because `[chi * mm^3 / (T*mm/A)] * T * T/mm = A*T*mm = mN`
- Mass: kg (PhysX requirement)
- mu0 = 4*pi*1e-4 T*mm/A

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- NVIDIA GPU (CUDA) for PINN training and inference
- [NVIDIA Kit SDK](https://github.com/NVIDIA-Omniverse/kit-app-template) for the Omniverse app

### 1. Install Dependencies

```bash
uv sync
```

This installs torch (CUDA 12.8), nvidia-physicsnemo, scipy, matplotlib, numpy, and other dependencies from `pyproject.toml`.

### 2. Train the PINN (or use existing checkpoint)

A trained checkpoint is included at `models/pinn_checkpoint/pinn_best.pt`
(v8 step-250k, `derived_b=True`). To retrain:

```bash
# Generate training data from analytical Biot-Savart (~1.42M samples)
uv run python scripts/generate_training_data.py

# Train v8 PINN (300k steps, ~4.7h on RTX 5090 — the data loss backprops
# through an autograd derivative, so every step is a double backward)
uv run python scripts/train_pinn.py
# Smoke test without touching the production checkpoint:
uv run python scripts/train_pinn.py --steps 500 --batch-size 16384 --out-name pinn_smoke
```

For remote GPU training (e.g., a more powerful machine on your network):
```bash
# Edit REMOTE_HOST and REMOTE_USER in the script, then:
bash scripts/remote_train.sh
# Pulls the result as pinn_v8_candidate.pt + step checkpoints to
# candidates/v8/ — promotion to pinn_best.pt is a manual decision after
# the adoption-gate evaluation (see below).
```

### 3. Validate the PINN

```bash
uv run python scripts/evaluate_pinn.py            # validates pinn_best.pt
uv run python scripts/evaluate_pinn.py --ckpt <path>   # validate a candidate
```

Runs a 5-level validation suite:
1. **Field accuracy**: Bz error vs analytical at 6 currents (12,800 points each)
2. **Gradient accuracy**: dB/dr, dB/dz via autograd vs finite-difference
3. **Force accuracy**: End-to-end F_z comparison at r=0 and r=5mm
4. **Design space**: 64 coil geometries (N x R_mean x L grid)
5. **Physics consistency**: div(B)=0, axial symmetry, z-mirror symmetry

Outputs:
- Console: PASS/FAIL per check
- Plots: `results/plots/pinn_validation/` (5 figures)
- JSON: `results/pinn_validation_report.json`
- Audit: `results/pinn_audit_report.md`

### 4. Analytical B-Field Validation

```bash
uv run python scripts/analytical_bfield.py
```

Generates reference plots in `results/plots/`.

### 5. Coil Design Optimizer

#### Web UI (Gradio)

```bash
uv run python scripts/coil_optimizer_app.py
```

Opens a browser UI for interactive multi-objective coil design optimization.
Features: constraint sliders, target-boost mode, Pareto plot, coupled ODE reranking.

#### CLI

```bash
uv run python scripts/optimize_coil_design.py
```

Headless optimization — prints top designs to console with Pareto plot saved to `results/`.

### 6. Run in NVIDIA Kit (Omniverse)

#### First-time setup

```bash
# Clone kit-app-template next to this project
git clone https://github.com/NVIDIA-Omniverse/kit-app-template.git ../kit-app-template
cd ../kit-app-template

# Build Kit SDK (downloads ~10GB, one-time)
.\repo.bat build

# Generate the launch script
.\repo.bat launch --app omnimarble
cd ../omnimarble
```

#### Launch

```cmd
omnimarble.kit.bat
```

This launches Kit with the extension loaded directly from `source/extensions/` (live editing).

#### In the Kit app

1. Click **Load Scene** to open the USD marble coaster scene
2. Click **Configure PhysX** to set CCD, 500Hz timestep, damping
3. Click **Start Simulation** -- marble rolls down the track, triggers IR gates, fires the coil
4. Watch the console log for real-time telemetry (current, field, force, velocity)
5. Adjust **Charge Voltage** and **Capacitance** in the UI panel, click **Update Parameters**
6. Click **Stop Simulation** -- the run's trajectory is written to
   `results/trajectories/kit_launch_<V>V_<C>uF_<timestamp>.csv`

#### Scripted autorun (reproducible launch runs)

The full sequence (load scene, configure PhysX, voltage override, start,
stop after the exit gates fire, write CSV, quit) can be driven without
clicking:

```cmd
omnimarble.kit.bat --/exts/omni.marble.coaster/autorun=true --/exts/omni.marble.coaster/autorunVoltage=300
```

Optional: `autorunMaxSimSeconds` (default 12), `autorunKeepOpen=true` to
leave the app running after the CSV is written.

#### Trajectory export

Every physics step records time, position (world + coil-local z_along/r),
velocities, coil current, capacitor voltage, PINN force and Bz, wire
temperature, and trigger/cutoff flags. The CSV starts with `# key=value`
metadata lines (run parameters, PINN checkpoint + step, gate times,
approach/exit velocity, boost ratio).

A committed reference artifact from the real Kit/PhysX/PINN path:
`results/trajectories/kit_launch_300V_470uF_20260702_174018.csv` — 300V run,
approach 208 mm/s, gate-measured exit 937.5 mm/s (**4.5x boost**).

#### What happens during simulation

1. Marble rolls down the starter slope under gravity
2. Two velocity-measurement IR gates compute approach velocity
3. Entry gate triggers the RLC discharge
4. Coupled electromechanical ODE computes current with 10us sub-stepping
5. PINN computes B-field, autograd computes gradients, force is applied via PhysX
6. Cutoff gate (MOSFET switch) kills the pulse when marble passes coil center
7. Two exit gates measure the launch velocity and compute boost ratio

## Project Structure

```
config/
  coil_params.json              # Source of truth: coil geometry + RLC circuit params

scripts/
  analytical_bfield.py          # Biot-Savart B-field (elliptic integrals) -- ground truth
  generate_training_data.py     # PINN training data from analytical solution (1.42M samples)
  pinn_loader.py                # Shared checkpoint loader + inference (legacy & derived-B)
  train_pinn.py                 # v8 derived-B PINN training (PhysicsNeMo)
  evaluate_pinn.py              # 5-level PINN validation suite (--ckpt for candidates)
  evaluate_candidates.py        # Multi-checkpoint Pareto comparison
  v7_optimization_diagnostic.py # PINN ranking stability diagnostic (--ckpt)
  physical_accuracy_audit.py    # Physics consistency validation
  validate_physics.py           # Physics validation utilities
  validate_kit_simulation.py    # Headless mirror of Kit physics (--field pinn|analytical)
  coil_optimizer_core.py        # Shared optimization engine (screening + coupled reranking)
  coil_optimizer_app.py         # Gradio web UI for coil design optimizer
  optimize_coil_design.py       # CLI coil design optimizer
  warp_bfield_solver.py         # PINN-based solver for headless simulation
  rlc_circuit.py                # RLC circuit utilities (saturation, eddy braking)
  remote_train.sh               # Remote GPU training via SSH
  generate_coil.py              # Parametric coil mesh generation
  em_force_injection.py         # Standalone EM force simulation
  simulation_loop.py            # Full orchestration loop
  optimize_launch.py            # Differentiable parameter optimization
  create_marble.py              # Marble actor creation
  compose_scene.py              # Scene composition from USD layers
  convert_stl_to_usd.py         # STL → USD conversion
  add_visuals.py                # Visual enhancement
  setup_launch_scene.py         # Launch scene setup
  run_physics_test.py           # Physics test runner

source/                         # NVIDIA Kit extension
  apps/omnimarble/
    omnimarble.kit              # Kit app configuration
  extensions/omni.marble.coaster/
    config/extension.toml       # Extension settings (overridable defaults)
    omni/marble/coaster/
      extension.py              # Main extension: RLC + PINN + PhysX force + UI + autorun
      coil_physics.py           # Pure-Python coil/RLC physics (unit-testable, no Kit deps)

tests/                          # pytest suite (uv run pytest)
  test_coil_physics.py          # Derived values vs config, RLC, gates, cross-checks
  test_rlc_circuit.py           # Saturation/eddy/thermal helpers, mm/mN/T unit system
  test_pinn_loader.py           # Loader version detection, derived-B structural guarantees

models/
  pinn_checkpoint/
    pinn_best.pt                # Production checkpoint: v8 step-250k, derived_b=True
                                # (historical v3-v7 checkpoints live in git history;
                                #  candidates/ holds gitignored periodic checkpoints)

usd/                            # USD scene layers
  marble_coaster_scene.usda     # Composed scene (references all layers)
  marble_actor.usda             # Marble rigid body (mass=0.00408kg)
  coil_geometry.usda            # Coil mesh
  coil_properties.usda          # Coil electrical properties
  track_geometry.usda           # Track collision mesh
  marble_trajectory.usda        # Trajectory reference
  physics_config.usda           # PhysX configuration
  visual_config.usda            # Visual settings

results/
  pinn_validation_report.json   # Validation metrics (machine-readable)
  pinn_audit_report.md          # Full PINN audit trail
  plots/                        # Generated validation plots

omnimarble.kit.bat              # Launch Kit from project root
```

## Configuration

All physical parameters live in `config/coil_params.json`:

```json
{
  "inner_radius_mm": 12.0,
  "outer_radius_mm": 18.0,
  "length_mm": 30.0,
  "num_turns": 30,
  "capacitance_uF": 470.0,
  "charge_voltage_V": 50.0,
  "esr_ohm": 0.01,
  "wiring_resistance_ohm": 0.02,
  "has_flyback_diode": true,
  "switch_type": "MOSFET"
}
```

The Kit extension reads this JSON as the primary source of truth. UI fields in Kit override individual values at runtime.

**Physical-build note:** the persisted `inductance_uH`/`resistance_ohm`
derive from a 12mm-former winding assumption, while the field model (and
the PINN trained on it) places the windings at R_mean = 15mm — no single
physical coil satisfies both. Build real coils to the field geometry and
overwrite these two values with LCR-measured ones; see
`hardware/README.md` ("Supported coils" / build sheet) for the full
analysis.

### Getting a stronger launch

The default 50V/470uF circuit stores only 0.59J — at 50V the EM force barely
changes the trajectory versus a plain gravity roll. For a visible launch:
- **300V / 470uF**: 21J stored — **demonstrated in Kit: 4.5x gate-measured
  boost** (artifact: `results/trajectories/kit_launch_300V_470uF_20260702_174018.csv`)
- To improve efficiency, increase coil inductance (more turns, tighter winding) so the discharge period (~5-20ms) matches the marble's transit time through the coil

## PINN Training Details

### Architecture (v8, physics by construction)
- NVIDIA PhysicsNeMo FullyConnected backbone
- 6 inputs: (r, z, I, N, R_mean, L); 1 raw output: scalar f
- A_phi = r*f; B_r = -r*df/dz and B_z = 2f + r*df/dr derived via autograd
  in forward() — div(B)=0, curl consistency, and on-axis BCs are exact
- 6 hidden layers x 256 units, SiLU activation, skip connections
- B/I normalization: model learns B/I (T/A), caller multiplies by I
- Canonical model + loader: `scripts/pinn_loader.py` (`BFieldPINNDerived`)

### Training data
- 1.42M samples from analytical Biot-Savart
- 75k spatial point pool: 30k dense near coil + 20k boundary-enriched + 5k on-axis + 20k far-field
- 711 parameter configs: grid corners + random + failure-mined N=10/R=8 families
- Per-config: 1500 general + 500 config-adaptive boundary points

### Loss function (v8)
- **Data loss**: MSE on derived B_r/I and B_z/I (backprops through the
  autograd derivative — the dominant per-step cost)
- **Far-field decay** (weight 0.1): fields -> 0 far from coil
- **z-mirror symmetry** (weight 0.5): Bz even, Br odd in z
- Progressive ramp: physics weights scale from 0.1x to 1.0x over first 20k steps
- Removed vs v7 (now exact by construction): curl, divergence-free, on-axis
  boundary conditions; gradient supervision dropped (third-order, marginal)
- Checkpoint selection by data loss + periodic saves for a post-hoc sweep

### Current validation status (v8 step-250k, production — 2026-07-02)

| Level | Check | Result |
|-------|-------|--------|
| 1 | Field accuracy (mean/max Bz error) | FAIL max-only (mean 0.86%, max 27.8% at winding singularities; informational) |
| 2 | Gradient accuracy (dBz/dz mean + P99) | PASS (P99 3.05%) |
| 3 | Force accuracy (peak err, correlation, zero-crossing) | PASS (2.7% / 3.0%, r>0.994) |
| 4 | Design space (64 configs under 5%) | PASS (95.3%, worst 7.75%) |
| 5 | Physics consistency (div=0, symmetry) | PASS (div(B)=0 exact by construction) |

Optimizer-ranking diagnostic vs analytical reference: Spearman r=1.000,
Kendall tau=0.992 over 60 candidates — safe for design optimization.

See `results/pinn_audit_report.md` for the full audit trail including all
training iterations (v1-v8) and the adoption-gate decision.

## License

MIT
