# OmniMarble — Magnetic Coil Marble Coaster Digital Twin

A physics-accurate digital twin of a magnetic coil-launched marble roller coaster, built on NVIDIA's simulation stack (OpenUSD, PhysX, Kit SDK).

## What This Is

A complete simulation pipeline that:
1. Converts a track STL into an OpenUSD scene with physics properties
2. Generates a parametric electromagnetic coil
3. Computes analytically-correct B-fields using Biot-Savart with elliptic integrals
4. Simulates marble launch through the coil with LR circuit pulse timing
5. Runs in NVIDIA Kit with real-time PhysX and a control UI

## Physical Accuracy Verification

The simulation has been validated against independent physics calculations:

### B-Field Validation

| Check | Result |
|-------|--------|
| B_z at coil center vs finite solenoid textbook formula | **1.7% error** |
| Single current loop vs `B = mu_0 * I / (2R)` | **Exact match** |
| B_r = 0 on axis (symmetry) | **Exact** |
| Finite/infinite solenoid ratio | 0.695 (correct for L/D ~ 1) |

### Force Calculation Validation

| Check | Result |
|-------|--------|
| mm-unit formula vs SI formula | **Exact match** |
| mu_0 unit conversion (T\*mm/A vs T\*m/A) | **Correct (1000x ratio)** |
| Force direction (attractive toward coil center) | **Correct** |

### Dimensional Analysis

All quantities carry correct units through the pipeline:
- `mu_0 = 4pi * 10^-4 T*mm/A` (mm-scaled)
- Volume in mm^3, mass from `density_g/mm3 * volume_mm3 * 1e-3 = kg`
- Force in mN: `[chi * mm^3 / (T*mm/A)] * T * T/mm = A*T*mm = mN`

### Energy Conservation

With default parameters (30 turns, 15mm radius, 10A, chi_eff=3):
- EM work over 30mm approach: **46.2 uJ**
- Friction loss over same distance: **420.7 uJ**
- Electrical-to-mechanical efficiency: **0.0004%**
- Conclusion: Force insufficient to overcome friction — **physically correct** for a small DC coilgun with stainless steel marble

### Comparison with Published Data

| Parameter | Our Simulation | Hobby Coilgun (typical) | Literature (Belcher 2007) |
|-----------|---------------|------------------------|--------------------------|
| Turns | 30 | 50-100 | 50 |
| Current | 10A DC | 100-500A (cap discharge) | 50-200A |
| Bore | 24mm | 12-15mm | 12mm |
| Projectile chi_eff | 3 (sphere demagnetization) | ~3 | ~3 |
| Velocity | ~0 (can't overcome friction) | 1-5 m/s | 2-5 m/s per stage |

The simulation correctly predicts that this configuration is too weak — matching real-world expectations for these parameters.

### Key Physics: Demagnetization Factor

For a ferromagnetic **sphere**, the effective susceptibility is limited by the demagnetization factor:

```
chi_eff = chi_intrinsic / (1 + chi_intrinsic/3)
```

For any strongly magnetic material (chi_intrinsic >> 1), this saturates at **chi_eff ~ 3**. This is why all ferromagnetic ball bearings behave similarly in a coilgun regardless of material.

## Project Structure

```
config/
  coil_params.json              # Coil geometry, electrical, and pulse parameters

scripts/
  convert_stl_to_usd.py         # STL -> OpenUSD track geometry
  create_marble.py               # Marble rigid body actor
  apply_physics.py               # Physics scene, collisions, materials
  add_visuals.py                 # Lighting and display materials
  compose_scene.py               # Compose all USD layers
  generate_coil.py               # Parametric coil mesh + properties
  analytical_bfield.py           # Biot-Savart B-field (elliptic integrals)
  em_force_injection.py          # EM force + physics simulation
  run_physics_test.py            # Gravity-only marble drop test
  validate_physics.py            # Force/energy/velocity predictions
  physical_accuracy_audit.py     # Cross-check against textbook formulas
  validate_kit_simulation.py     # Headless replica of Kit extension physics
  generate_training_data.py      # PINN training data from analytical solution
  train_pinn.py                  # PhysicsNeMo PINN training (GPU)
  evaluate_pinn.py               # PINN vs analytical comparison
  warp_em_kernel.py              # Warp GPU kernels for EM force
  simulation_loop.py             # Full orchestration loop
  optimize_launch.py             # Differentiable parameter optimization

source/                          # NVIDIA Kit app (copy to kit-app-template)
  apps/omnimarble/
    omnimarble.kit               # Kit app configuration
  extensions/omni.marble.coaster/
    config/extension.toml        # Extension config + default parameters
    omni/marble/coaster/
      extension.py               # PhysX force injection + UI panel

usd/                             # Generated USD files (after running scripts)
data/
  track.stl                      # Input track geometry
```

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A `track.stl` file in `data/`

### 1. Install Dependencies

```bash
uv init  # if fresh clone
uv add trimesh numpy usd-core scipy matplotlib rtree
```

### 2. Generate the USD Scene

Run in order:
```bash
uv run python scripts/convert_stl_to_usd.py
uv run python scripts/create_marble.py
uv run python scripts/apply_physics.py
uv run python scripts/generate_coil.py
uv run python scripts/add_visuals.py
uv run python scripts/compose_scene.py
```

This creates `usd/marble_coaster_scene.usda` with track, marble, coil, physics, lighting, and materials.

### 3. Validate the Physics

```bash
# Full accuracy audit (cross-checks against textbook formulas)
uv run python scripts/physical_accuracy_audit.py

# Force/energy/velocity predictions
uv run python scripts/validate_physics.py

# Headless simulation matching Kit extension logic
uv run python scripts/validate_kit_simulation.py

# Gravity-only drop test
uv run python scripts/run_physics_test.py

# EM force injection simulation
uv run python -c "import sys; sys.path.insert(0,'scripts'); from em_force_injection import main; main()"
```

### 4. Analytical B-Field Plots

```bash
uv run python scripts/analytical_bfield.py
```

Generates `results/plots/bfield_validation.png` and `results/plots/em_force_on_axis.png`.

### 5. Run in NVIDIA Kit (Optional)

Requires the [Kit SDK](https://github.com/NVIDIA-Omniverse/kit-app-template):

```bash
# Clone kit-app-template
git clone https://github.com/NVIDIA-Omniverse/kit-app-template.git ../kit-app-template

# Copy our app and extension
cp -r source/apps/omnimarble ../kit-app-template/source/apps/
cp -r source/extensions/omni.marble.coaster ../kit-app-template/source/extensions/

# Update the path in extension.py to point to your omnimarble project
# (edit OMNIMARBLE_PROJECT in extension.py)

# Build and launch
cd ../kit-app-template
.\repo.bat build      # Windows
.\repo.bat launch omnimarble.kit
```

In the Kit app:
1. Click **Load Scene** to open the USD scene
2. Click **Configure PhysX** to set CCD, timestep, damping
3. Click **Start Simulation** to run with EM force injection
4. Adjust coil parameters in the UI panel

### 6. PINN Training (GPU Required)

```bash
uv add torch nvidia-physicsnemo

# Generate training data from analytical solution
uv run python -c "import sys; sys.path.insert(0,'scripts'); from generate_training_data import main; main()"

# Train PINN
uv run python scripts/train_pinn.py

# Evaluate
uv run python scripts/evaluate_pinn.py
```

## Adjusting Parameters

Edit `config/coil_params.json`:

```json
{
  "inner_radius_mm": 12.0,     // Coil bore inner radius
  "outer_radius_mm": 18.0,     // Coil outer radius
  "length_mm": 30.0,           // Coil length along axis
  "num_turns": 30,             // Number of wire turns
  "position_mm": [0, 30, 0],   // Coil center position (track coords)
  "axis": [0, 1, 0],           // Coil axis direction
  "max_current_A": 10.0,       // Peak current
  "pulse_width_ms": 5.0,       // Max pulse duration
  "supply_voltage_V": 24.0,    // Supply voltage
  "resistance_ohm": 1.2,       // Coil resistance
  "inductance_uH": 150.0       // Coil inductance
}
```

After editing, regenerate: `uv run python scripts/generate_coil.py && uv run python scripts/compose_scene.py`

**To get visible marble launch**, increase current (real coilguns use capacitor discharge at 100-500A) or increase turns.

## How the Physics Works

### B-Field Computation
The solenoid B-field is computed by summing contributions from N circular current loops using the Biot-Savart law with complete elliptic integrals K(m) and E(m). This is exact for any solenoid geometry.

### Force Model
The force on a ferromagnetic sphere in a non-uniform field:
```
F = (chi_eff * V / mu_0) * (B . nabla)B
```
Gradients are computed via central finite differences on the analytical field.

### Pulse Timing
Current follows an LR circuit model: `I(t) = (V/R)(1 - e^(-Rt/L))` during the pulse. The pulse is cut when the marble passes the coil center to prevent deceleration from the symmetric return force.

### Collision
The standalone simulation uses trimesh ray-casting for marble-track collision detection with coefficient of restitution and friction. The Kit version uses NVIDIA PhysX.

## License

MIT
