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

- **Architecture**: NVIDIA PhysicsNeMo FullyConnected, 6x256 hidden, SiLU, skip connections, 331k params
- **Inputs**: (r, z, I, N, R_mean, L) -- spatial position + coil design parameters
- **Outputs**: (A_phi, B_r, B_z)
- **Key feature**: Model learns B/I (T/A), collapsing dynamic range across the full current range [0.5, 4000] A
- **Training**: 1.02M samples from analytical Biot-Savart, physics losses (curl, div-free, boundary, symmetry)
- **Validation**: 5-level test suite -- field accuracy, gradient accuracy, force accuracy, design space coverage, physics consistency

### RLC Circuit

The coil is driven by a capacitor discharge (underdamped RLC):

```
I(t) = (V0 / omega_d * L) * exp(-alpha * t) * sin(omega_d * t)
```

With coupled electromechanical ODE (back-EMF from marble motion), RK4 sub-stepping at 10us resolution, and flyback diode clamping.

Default circuit: 470uF capacitor, 50V charge, L=12.4uH, R=0.11 ohm -> peak current 327A, underdamped (zeta=0.34).

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

A trained checkpoint is included at `models/pinn_checkpoint/pinn_best.pt`. To retrain:

```bash
# Generate training data from analytical Biot-Savart (~1M samples)
uv run python scripts/generate_training_data.py

# Train PINN (200k steps, ~30min on RTX 4090)
uv run python scripts/train_pinn.py
```

For remote GPU training (e.g., a more powerful machine on your network):
```bash
# Edit REMOTE_HOST and REMOTE_USER in the script, then:
bash scripts/remote_train.sh
```

### 3. Validate the PINN

```bash
uv run python scripts/evaluate_pinn.py
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

### 5. Run in NVIDIA Kit (Omniverse)

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
  generate_training_data.py     # PINN training data from analytical solution (1M+ samples)
  train_pinn.py                 # PhysicsNeMo PINN training with physics losses
  evaluate_pinn.py              # 5-level PINN validation suite
  warp_bfield_solver.py         # PINN-based solver for headless simulation
  rlc_circuit.py                # RLC circuit utilities (saturation, eddy braking)
  remote_train.sh               # Remote GPU training via SSH
  generate_coil.py              # Parametric coil mesh generation
  em_force_injection.py         # Standalone EM force simulation
  simulation_loop.py            # Full orchestration loop
  optimize_launch.py            # Differentiable parameter optimization

source/                         # NVIDIA Kit extension
  apps/omnimarble/
    omnimarble.kit              # Kit app configuration
  extensions/omni.marble.coaster/
    config/extension.toml       # Extension settings (overridable defaults)
    omni/marble/coaster/
      extension.py              # Main extension: RLC + PINN + PhysX force + UI

models/
  pinn_checkpoint/
    pinn_best.pt                # Trained PINN checkpoint (current_normalized=True)

usd/                            # USD scene layers
  marble_coaster_scene.usda     # Composed scene (references all layers)
  marble_actor.usda             # Marble rigid body (mass=0.00408kg)
  coil_geometry.usda            # Coil mesh
  coil_properties.usda          # Coil electrical properties
  track_geometry.usda           # Track collision mesh

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

### Getting a stronger launch

The default 50V/470uF circuit stores only 0.59J. For a visible marble launch:
- **300V / 470uF**: 21J stored, ~4.5x velocity boost, I_peak=1963A
- To improve efficiency, increase coil inductance (more turns, tighter winding) so the discharge period (~5-20ms) matches the marble's transit time through the coil

## PINN Training Details

### Architecture
- NVIDIA PhysicsNeMo FullyConnected backbone
- 6 inputs: (r, z, I, N, R_mean, L)
- 3 outputs: (A_phi, B_r, B_z)
- 6 hidden layers x 256 units, SiLU activation, skip connections
- B/I normalization: model learns B/I (T/A), caller multiplies by I

### Training data (v2)
- 1.02M samples from analytical Biot-Savart
- 75k spatial point pool: 30k dense near coil + 20k boundary-enriched + 5k on-axis + 20k far-field
- 511 parameter configs: 256 grid corners + 244 random + 11 default
- Per-config: 1500 general + 500 config-adaptive boundary points

### Loss function
- **Data loss**: MSE on B_r/I and B_z/I
- **Curl consistency** (weight 1.0): B_r = -dA/dz, B_z = (1/r)d(rA)/dr
- **Divergence-free** (weight 1.0): dBr/dr + Br/r + dBz/dz = 0
- **Boundary conditions** (weight 0.1): A_phi=0, Br=0 at r=0; fields->0 far from coil
- **Symmetry** (weight 0.5): Bz(r,z)=Bz(r,-z), Br(r,z)=-Br(r,-z), Br=0 on axis
- **Gradient consistency** (weight 0.1): autograd vs finite-difference self-check
- Progressive ramp: physics weights scale from 0.1x to 1.0x over first 20k steps

### Current validation status (v4 checkpoint)

| Level | Check | Result |
|-------|-------|--------|
| 1 | Field accuracy (mean/max Bz error) | PASS (0.6% / 19%) |
| 2 | Gradient accuracy (dBz/dz) | FAIL (max ~100% at singularities) |
| 3 | Force accuracy (peak err, correlation, zero-crossing) | PASS |
| 4 | Design space (64 configs under 5%) | FAIL (75%, need 95%) |
| 5 | Physics consistency (div=0, symmetry) | FAIL (div 0.019 > 0.01) |

See `results/pinn_audit_report.md` for the full audit trail including all training iterations.

## License

MIT
