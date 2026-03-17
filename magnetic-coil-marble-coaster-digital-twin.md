# Magnetic Coil Marble Roller Coaster — Digital Twin Implementation Plan

## Project overview

Build a digital twin of a magnetic coil-launched marble roller coaster using NVIDIA's simulation stack. The system simulates electromagnetic coil launchers accelerating a steel marble along a 3D-printed track, with the goal of predicting physical behavior digitally before building, and eventually running a live sim-to-real feedback loop.

### What we have

- **One STL file** of the roller coaster track (triangle mesh, no semantic separation, no physics)
- **No CAD model of the coil** — we need to generate a parametric solenoid geometry programmatically
- **Target marble**: ~10mm diameter steel ball bearing (density ~7800 kg/m³)
- **Target coil**: single-stage solenoid launcher (copper wire, ~20-40 turns, ~12-24V pulse)

### What we're building

1. **STL → OpenUSD pipeline** — convert the track mesh into a physics-ready USD scene
2. **Parametric coil generator** — programmatically create solenoid geometry in USD
3. **PhysicsNeMo PINN electromagnetic surrogate** — train a neural network on Maxwell's magnetostatic equations to predict B-field for any coil configuration in real-time
4. **PhysX rigid body simulation** — marble dynamics on the track with gravity, friction, contact
5. **Warp integration** — GPU-accelerated marble + EM force coupling, differentiable for optimization
6. **Omniverse scene** — composed USD layers for visualization and sim

---

## Phase 1: STL to OpenUSD conversion

### 1.1 Mesh preparation

The STL is raw triangle soup. Before converting, we need to clean and optionally split it.

**Dependencies**: `trimesh`, `numpy`, `pxr` (OpenUSD Python bindings)

```bash
pip install trimesh numpy --break-system-packages
```

For the `pxr` USD bindings, install via:
```bash
pip install usd-core --break-system-packages
```

**Steps**:

1. Load the STL with trimesh
2. Fix normals (ensure consistent winding)
3. Remove degenerate triangles (zero-area faces)
4. Optionally decimate if triangle count > 100k (for PhysX performance)
5. Identify and separate disconnected components (track sections, supports, etc.) using `trimesh.graph.connected_components`
6. Export each component as a separate mesh in the USD scene

```python
import trimesh
import numpy as np

mesh = trimesh.load("track.stl")
mesh.fix_normals()
mesh.remove_degenerate_faces()
mesh.remove_duplicate_faces()

# Split into connected components
components = mesh.split(only_watertight=False)
print(f"Found {len(components)} connected components")

# Identify the main track (largest component by face count)
track_mesh = max(components, key=lambda m: len(m.faces))
support_meshes = [m for m in components if m is not track_mesh]
```

### 1.2 Write to OpenUSD

Create a USD stage with proper units, up-axis, and the track mesh as a `UsdGeom.Mesh`.

**Critical**: STL files have no unit metadata. You must know what units your CAD software used. If the STL is in millimeters, set `metersPerUnit = 0.001`.

```python
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt, Sdf

stage = Usd.Stage.CreateNew("marble_coaster.usda")

# Set scene metadata — CRITICAL: match your STL's units
UsdGeom.SetStageMetersPerUnit(stage, 0.001)  # if STL is in mm
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)  # match your CAD

# Create world root
world = UsdGeom.Xform.Define(stage, "/World")

# Create the track mesh
track_prim = UsdGeom.Mesh.Define(stage, "/World/Track")
track_prim.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in track_mesh.vertices]))
track_prim.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(track_mesh.faces)))
track_prim.CreateFaceVertexIndicesAttr(Vt.IntArray(track_mesh.faces.flatten().tolist()))
track_prim.CreateSubdivisionSchemeAttr("none")  # keep triangles as-is

# Compute and set normals
normals = track_mesh.vertex_normals
track_prim.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals]))
track_prim.SetNormalsInterpolation("vertex")

# Set extent (bounding box)
bounds = track_mesh.bounds
track_prim.CreateExtentAttr(Vt.Vec3fArray([
    Gf.Vec3f(*bounds[0]), Gf.Vec3f(*bounds[1])
]))

stage.GetRootLayer().Save()
```

### 1.3 Apply physics schemas to the track

The track is a **static collider** — it doesn't move, but the marble collides with it.

```python
# Make the track a static collider using exact triangle mesh
track_p = stage.GetPrimAtPath("/World/Track")
UsdPhysics.CollisionAPI.Apply(track_p)
mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(track_p)
mesh_collision.GetApproximationAttr().Set("none")  # exact triangle mesh, no simplification

# Create a physics scene
physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
physics_scene.CreateGravityMagnitudeAttr(9810.0)  # mm/s² if stage is in mm
```

### 1.4 Create the marble

The marble is a **dynamic rigid body** with a sphere collider.

```python
# Marble transform
marble_xform = UsdGeom.Xform.Define(stage, "/World/Marble")
marble_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 50))  # starting position (adjust)

# Marble visual geometry
marble_geom = UsdGeom.Sphere.Define(stage, "/World/Marble/Geom")
marble_geom.CreateRadiusAttr(5.0)  # 5mm radius = 10mm diameter marble

# Rigid body API on the Xform (parent)
rb = UsdPhysics.RigidBodyAPI.Apply(marble_xform.GetPrim())

# Collision API on the sphere geometry
UsdPhysics.CollisionAPI.Apply(marble_geom.GetPrim())

# Mass properties
mass_api = UsdPhysics.MassAPI.Apply(marble_xform.GetPrim())
mass_api.CreateDensityAttr(7.8e-3)  # steel density in g/mm³ (7800 kg/m³)
```

### 1.5 Physics materials

```python
# Track material (PLA/resin surface)
track_mat_path = "/World/Materials/TrackMaterial"
track_mat_prim = stage.DefinePrim(track_mat_path)
track_phys_mat = UsdPhysics.MaterialAPI.Apply(track_mat_prim)
track_phys_mat.CreateStaticFrictionAttr(0.4)
track_phys_mat.CreateDynamicFrictionAttr(0.3)
track_phys_mat.CreateRestitutionAttr(0.3)  # low bounce on track

# Marble material (steel)
marble_mat_path = "/World/Materials/MarbleMaterial"
marble_mat_prim = stage.DefinePrim(marble_mat_path)
marble_phys_mat = UsdPhysics.MaterialAPI.Apply(marble_mat_prim)
marble_phys_mat.CreateStaticFrictionAttr(0.5)
marble_phys_mat.CreateDynamicFrictionAttr(0.35)
marble_phys_mat.CreateRestitutionAttr(0.6)  # steel bounces moderately

# Bind materials (using UsdShade binding)
from pxr import UsdShade
UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Track")).Bind(
    UsdShade.Material(track_mat_prim))
UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Marble/Geom")).Bind(
    UsdShade.Material(marble_mat_prim))
```

### 1.6 PhysX simulation tuning for the marble

```python
from pxr import PhysxSchema

# Enable CCD on the marble (critical — small fast object after launch)
physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(marble_xform.GetPrim())
physx_rb.CreateEnableCCDAttr(True)

# Increase contact offset slightly for better contact detection on curved track
physx_col = PhysxSchema.PhysxCollisionAPI.Apply(marble_geom.GetPrim())
physx_col.CreateContactOffsetAttr(2.0)  # 2mm contact offset
physx_col.CreateRestOffsetAttr(0.0)

# Configure the physics scene for accurate marble simulation
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
physx_scene.CreateTimeStepsPerSecondAttr(120)  # 120 Hz minimum for small marble
physx_scene.CreateEnableCCDAttr(True)
# TGS solver is the default in PhysX 5.x — no explicit setting needed
# It applies friction every iteration, which gives smooth rolling behavior

stage.GetRootLayer().Save()
```

---

## Phase 2: Parametric coil generator

Since we don't have a coil CAD model, we generate one programmatically. The coil is a solenoid — a helical winding of wire around a cylindrical bore.

### 2.1 Coil geometry parameters

```python
# Coil parameters (adjust to your physical build)
COIL_PARAMS = {
    "inner_radius": 8.0,      # mm — must be > marble radius (5mm) + clearance
    "outer_radius": 14.0,     # mm — wire winding thickness
    "length": 30.0,           # mm — coil length along axis
    "num_turns": 30,          # number of wire turns
    "wire_diameter": 0.8,     # mm — wire gauge (AWG 20 ≈ 0.8mm)
    "coil_position": [0, 0, 20],  # mm — center position on track
    "coil_axis": [1, 0, 0],   # direction the marble travels through
}
```

### 2.2 Generate coil mesh

Create a simplified cylindrical shell representation (not individual wire turns — that's for the EM solver, not the visual/collision mesh).

```python
def create_coil_mesh(stage, path, params):
    """Create a cylindrical coil housing as a USD mesh."""
    import math

    r_inner = params["inner_radius"]
    r_outer = params["outer_radius"]
    length = params["length"]
    n_segments = 32  # circumferential resolution

    # Generate vertices for inner and outer cylinders, top and bottom caps
    verts = []
    faces = []
    counts = []

    for i in range(n_segments):
        angle = 2 * math.pi * i / n_segments
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        # 4 rings: inner-bottom, inner-top, outer-bottom, outer-top
        verts.append(Gf.Vec3f(0, r_inner * cos_a, r_inner * sin_a))           # 0: inner bottom
        verts.append(Gf.Vec3f(length, r_inner * cos_a, r_inner * sin_a))      # 1: inner top
        verts.append(Gf.Vec3f(0, r_outer * cos_a, r_outer * sin_a))           # 2: outer bottom
        verts.append(Gf.Vec3f(length, r_outer * cos_a, r_outer * sin_a))      # 3: outer top

    # Generate quad faces connecting adjacent segments
    for i in range(n_segments):
        j = (i + 1) % n_segments
        base_i = i * 4
        base_j = j * 4

        # Outer wall
        faces.extend([base_i + 2, base_j + 2, base_j + 3, base_i + 3])
        counts.append(4)
        # Inner wall
        faces.extend([base_i + 0, base_i + 1, base_j + 1, base_j + 0])
        counts.append(4)
        # Bottom cap
        faces.extend([base_i + 0, base_j + 0, base_j + 2, base_i + 2])
        counts.append(4)
        # Top cap
        faces.extend([base_i + 1, base_i + 3, base_j + 3, base_j + 1])
        counts.append(4)

    coil_mesh = UsdGeom.Mesh.Define(stage, path)
    coil_mesh.CreatePointsAttr(Vt.Vec3fArray(verts))
    coil_mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    coil_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces))
    coil_mesh.CreateSubdivisionSchemeAttr("none")

    # Position the coil
    xform = UsdGeom.Xformable(coil_mesh.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*params["coil_position"]))

    # Static collider (marble passes through the bore, but can hit the housing)
    UsdPhysics.CollisionAPI.Apply(coil_mesh.GetPrim())
    col = UsdPhysics.MeshCollisionAPI.Apply(coil_mesh.GetPrim())
    col.GetApproximationAttr().Set("convexDecomposition")

    return coil_mesh
```

### 2.3 Store coil parameters as USD custom attributes

These custom properties let the simulation code read coil specs directly from the USD scene.

```python
coil_prim = stage.GetPrimAtPath("/World/Coil")

# Custom attributes for the EM solver to read
coil_prim.CreateAttribute("coilLauncher:numTurns", Sdf.ValueTypeNames.Int).Set(30)
coil_prim.CreateAttribute("coilLauncher:wireGauge_mm", Sdf.ValueTypeNames.Float).Set(0.8)
coil_prim.CreateAttribute("coilLauncher:innerRadius_mm", Sdf.ValueTypeNames.Float).Set(8.0)
coil_prim.CreateAttribute("coilLauncher:length_mm", Sdf.ValueTypeNames.Float).Set(30.0)
coil_prim.CreateAttribute("coilLauncher:maxCurrent_A", Sdf.ValueTypeNames.Float).Set(10.0)
coil_prim.CreateAttribute("coilLauncher:pulseWidth_ms", Sdf.ValueTypeNames.Float).Set(5.0)
coil_prim.CreateAttribute("coilLauncher:resistance_ohm", Sdf.ValueTypeNames.Float).Set(1.2)
coil_prim.CreateAttribute("coilLauncher:inductance_uH", Sdf.ValueTypeNames.Float).Set(150.0)
```

---

## Phase 3: PhysicsNeMo PINN electromagnetic surrogate

Train a physics-informed neural network to predict the magnetic field of the solenoid for any configuration, enabling real-time EM force computation during simulation.

### 3.1 Problem formulation

The coil launcher is a **magnetostatic** problem (quasi-static — current pulse is slow vs speed of light). We solve in 2D axisymmetric cylindrical coordinates (r, z) exploiting the solenoid's rotational symmetry.

**Governing equations** (magnetostatic Maxwell in cylindrical coords):

- ∇ × **B** = μ₀**J** (Ampere's law)
- ∇ · **B** = 0 (divergence-free)

In terms of the magnetic vector potential **A** (where **B** = ∇ × **A**), for axisymmetric problems the only nonzero component is A_φ, and the PDE becomes:

```
-∂²A_φ/∂r² - (1/r)∂A_φ/∂r + A_φ/r² - ∂²A_φ/∂z² = μ₀ J_φ(r,z)
```

where J_φ is the current density (nonzero only inside the coil winding region).

### 3.2 PINN architecture

**Inputs**: (r, z, I, N, R_coil, L_coil) — position + coil parameters
**Outputs**: (A_φ, B_r, B_z) — vector potential and field components
**Hidden layers**: 6 layers × 256 neurons, sin activation (good for oscillatory fields)

```python
# PhysicsNeMo Sym setup (pseudocode — adapt to current API)
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.key import Key

# Network architecture
net = instantiate_arch(
    input_keys=[Key("r"), Key("z"), Key("I"), Key("N"), Key("R"), Key("L")],
    output_keys=[Key("A_phi"), Key("B_r"), Key("B_z")],
    cfg={
        "arch_type": "fully_connected",
        "layer_size": 256,
        "nr_layers": 6,
        "activation_fn": "sin",
    }
)
```

### 3.3 Loss function components

```python
# 1. PDE residual loss (interior of domain)
#    -d²A/dr² - (1/r)dA/dr + A/r² - d²A/dz² = mu0 * J(r,z)
#    Sampled at ~10,000 random (r,z) points per batch

# 2. Divergence-free constraint
#    ∂B_r/∂r + B_r/r + ∂B_z/∂z = 0
#    (enforced as soft constraint on the B outputs)

# 3. Boundary conditions
#    A_phi → 0 as r,z → ∞ (far-field decay)
#    A_phi = 0 at r = 0 (symmetry axis)
#    B_r = 0 at r = 0 (axial symmetry)

# 4. Curl consistency
#    B_r = -∂A_phi/∂z
#    B_z = (1/r)∂(r*A_phi)/∂r
#    (ensures B outputs are consistent with A output)

# 5. Optional: data loss from FEM reference solutions or measurements
#    If you have COMSOL/FEMM solutions, add as supervision anchors
```

### 3.4 Training configuration

```python
# Domain bounds (in mm)
DOMAIN = {
    "r": (0, 100),        # radial extent (0 to 10× coil radius)
    "z": (-150, 150),     # axial extent (±5× coil length)
}

# Parameter ranges for the parameterized PINN
PARAM_RANGES = {
    "I": (0.5, 20.0),     # current in amps
    "N": (10, 60),        # number of turns
    "R": (6.0, 20.0),     # inner radius in mm
    "L": (15.0, 60.0),    # coil length in mm
}

# Training
TRAINING = {
    "max_steps": 100_000,
    "batch_size": 4096,
    "learning_rate": 1e-3,
    "lr_scheduler": "cosine_annealing",
    "interior_points_per_step": 10000,
    "boundary_points_per_step": 2000,
    "coil_region_points_per_step": 3000,  # extra density in coil region
}
```

### 3.5 Force computation from the trained PINN

Once trained, the PINN predicts B(r,z) for any coil config in ~1ms. The force on the ferromagnetic marble is:

```python
def compute_marble_force(pinn, marble_pos, marble_radius, coil_params):
    """
    Compute electromagnetic force on a ferromagnetic sphere.

    F = (χ_eff * V / μ₀) * ∇(B²/2)

    where:
    - χ_eff = effective susceptibility of steel (~200 for mild steel)
    - V = (4/3)πr³ = marble volume
    - B = magnetic flux density from PINN
    - ∇(B²/2) = B_r * ∂B_r/∂r + B_z * ∂B_z/∂z (gradient of field energy)
    """
    r, z = marble_pos  # axisymmetric coordinates
    I, N, R, L = coil_params

    # Query PINN (differentiable — can compute gradients)
    B_r, B_z = pinn.predict(r, z, I, N, R, L)

    # Compute field gradient (automatic differentiation through PINN)
    dBr_dr, dBr_dz = pinn.gradient(B_r, r, z)
    dBz_dr, dBz_dz = pinn.gradient(B_z, r, z)

    # Force on ferromagnetic sphere
    chi_eff = 200.0  # steel susceptibility (tune to your marble)
    mu0 = 4 * np.pi * 1e-7  # T·m/A (convert to mm units in implementation)
    V = (4/3) * np.pi * marble_radius**3

    # F = (χV/μ₀) * (B · ∇)B
    F_r = (chi_eff * V / mu0) * (B_r * dBr_dr + B_z * dBr_dz)
    F_z = (chi_eff * V / mu0) * (B_r * dBz_dr + B_z * dBz_dz)

    return F_r, F_z
```

---

## Phase 4: Warp GPU simulation kernel

NVIDIA Warp provides the glue between the PINN EM model and the PhysX rigid body dynamics, running on GPU with full differentiability.

### 4.1 Dependencies

```bash
pip install warp-lang --break-system-packages
```

### 4.2 Marble dynamics kernel

```python
import warp as wp

@wp.struct
class CoilParams:
    position: wp.vec3     # coil center in world coords
    axis: wp.vec3         # coil axis direction (normalized)
    current: wp.float32   # instantaneous current (A)
    num_turns: wp.int32
    inner_radius: wp.float32
    length: wp.float32

@wp.struct
class MarbleState:
    position: wp.vec3
    velocity: wp.vec3
    angular_velocity: wp.vec3
    radius: wp.float32
    mass: wp.float32

@wp.kernel
def compute_em_force_kernel(
    marble: wp.array(dtype=MarbleState),
    coils: wp.array(dtype=CoilParams),
    num_coils: wp.int32,
    em_force: wp.array(dtype=wp.vec3),  # output
    dt: wp.float32,
):
    """Compute electromagnetic force on the marble from all coils."""
    tid = wp.tid()

    m = marble[tid]
    total_force = wp.vec3(0.0, 0.0, 0.0)

    for c in range(num_coils):
        coil = coils[c]

        # Transform marble position to coil-local cylindrical coords
        rel = m.position - coil.position
        along_axis = wp.dot(rel, coil.axis)  # z in coil frame
        perp = rel - along_axis * coil.axis
        r = wp.length(perp)  # radial distance from coil axis

        # Query PINN surrogate for B-field (placeholder — actual PINN inference here)
        # In practice, this calls into a pre-trained model or lookup table
        B_z, B_r, dBz_dz, dBr_dr = evaluate_bfield(
            r, along_axis, coil.current, coil.num_turns,
            coil.inner_radius, coil.length
        )

        # Force on ferromagnetic sphere: F = (χV/μ₀)(B·∇)B
        chi_eff = 200.0
        mu0 = 1.2566e-3  # μ₀ in mm-scale units (T·mm/A * conversion)
        V = 4.0 / 3.0 * 3.14159 * m.radius * m.radius * m.radius

        coeff = chi_eff * V / mu0
        F_axial = coeff * (B_r * dBr_dr + B_z * dBz_dz)  # simplified
        F_radial = coeff * (B_r * dBr_dr + B_z * dBr_dz)  # simplified

        # Transform back to world coordinates
        force_world = F_axial * coil.axis
        if r > 0.01:
            radial_dir = perp / r
            force_world = force_world + F_radial * radial_dir

        total_force = total_force + force_world

    em_force[tid] = total_force
```

### 4.3 Coil current pulse model

The coil current isn't constant — it follows an LR circuit discharge profile.

```python
@wp.func
def coil_current_pulse(
    t: wp.float32,             # time since trigger (ms)
    V_supply: wp.float32,      # supply voltage (V)
    R: wp.float32,             # coil resistance (Ω)
    L: wp.float32,             # coil inductance (H)
    pulse_width: wp.float32,   # how long switch is closed (ms)
) -> wp.float32:
    """LR circuit current: I(t) = (V/R)(1 - e^(-Rt/L)) during pulse, then decay."""
    tau = L / R  # time constant in seconds
    t_sec = t * 0.001  # convert ms to seconds
    pw_sec = pulse_width * 0.001

    if t_sec < 0.0:
        return 0.0
    elif t_sec < pw_sec:
        # Rising phase
        return (V_supply / R) * (1.0 - wp.exp(-t_sec / tau))
    else:
        # Decay phase after switch opens
        I_peak = (V_supply / R) * (1.0 - wp.exp(-pw_sec / tau))
        return I_peak * wp.exp(-(t_sec - pw_sec) / tau)
```

### 4.4 Integration with PhysX via Omniverse

Each simulation frame:

```python
def simulation_step(sim_time, dt):
    """Called every PhysX step from an Omniverse extension or Isaac Sim script."""

    # 1. Read marble state from USD
    marble_pos = read_rigid_body_position(stage, "/World/Marble")
    marble_vel = read_rigid_body_velocity(stage, "/World/Marble")

    # 2. Read coil parameters from USD custom attributes
    coil_prim = stage.GetPrimAtPath("/World/Coil")
    coil_params = read_coil_params(coil_prim)

    # 3. Compute current at this time step
    coil_params.current = coil_current_pulse(
        sim_time - trigger_time,
        V_supply=24.0, R=1.2, L=150e-6,
        pulse_width=5.0
    )

    # 4. Compute EM force via Warp kernel (GPU)
    wp.launch(compute_em_force_kernel, dim=1,
              inputs=[marble_state, coil_array, num_coils, em_force_out, dt])

    # 5. Inject force into PhysX
    force = em_force_out.numpy()[0]
    psi = get_physx_simulation_interface()
    psi.apply_force_at_pos(
        stage_id, marble_prim_id,
        carb.Float3(force[0], force[1], force[2]),
        carb.Float3(marble_pos[0], marble_pos[1], marble_pos[2])
    )
```

---

## Phase 5: Differentiable optimization (optional but powerful)

Because both the PINN and Warp are differentiable, we can optimize coil design parameters end-to-end.

### 5.1 Optimization targets

```python
# Define what "good" means for the coil launcher
OPTIMIZATION_TARGETS = {
    "marble_exit_velocity": 2000.0,    # mm/s target exit speed
    "marble_stays_on_track": True,     # no derailing
    "energy_efficiency": "maximize",    # minimize wasted energy
    "coil_current_limit": 15.0,        # max safe current (A)
}
```

### 5.2 Loss function for optimization

```python
def launch_quality_loss(coil_params, track_geometry):
    """
    Differentiable loss: simulate marble trajectory, penalize bad outcomes.

    Backpropagation flows through:
    coil_params → current pulse → PINN B-field → force → Warp trajectory → loss
    """
    trajectory = simulate_marble(coil_params, track_geometry)

    exit_vel = trajectory.velocity_at_coil_exit
    max_height = trajectory.max_height_reached
    derailed = trajectory.left_track

    loss = 0.0
    loss += (exit_vel - target_velocity) ** 2           # velocity accuracy
    loss += 100.0 * derailed                            # heavy penalty for derailing
    loss += 0.01 * coil_params.current_integral ** 2    # energy cost

    return loss
```

### 5.3 Parameters to optimize

- **Pulse timing**: when to fire relative to marble position
- **Pulse width**: how long to keep the current flowing
- **Supply voltage**: higher = more force, but also more heat
- **Coil position**: where along the track to mount the coil
- **Number of turns**: more turns = stronger field but more inductance (slower response)

---

## Phase 6: USD layer composition

Structure the final scene as composable USD layers for clean separation of concerns.

### 6.1 Layer structure

```
marble_coaster_scene.usda          ← root (sublayers everything)
├── track_geometry.usda            ← converted STL mesh (Phase 1)
├── coil_geometry.usda             ← generated coil mesh (Phase 2)
├── physics_config.usda            ← collision, rigid body, materials (Phase 1.3-1.6)
├── coil_properties.usda           ← custom EM attributes (Phase 2.3)
├── marble_actor.usda              ← marble rigid body + collision (Phase 1.4)
└── visualization_materials.usda   ← MDL render materials (optional, for RTX)
```

### 6.2 Root scene composition

```python
# marble_coaster_scene.usda
root_stage = Usd.Stage.CreateNew("marble_coaster_scene.usda")
root_layer = root_stage.GetRootLayer()

# Sublayer all components (strongest-to-weakest opinion)
root_layer.subLayerPaths = [
    "./physics_config.usda",
    "./coil_properties.usda",
    "./marble_actor.usda",
    "./coil_geometry.usda",
    "./track_geometry.usda",
]

UsdGeom.SetStageMetersPerUnit(root_stage, 0.001)
UsdGeom.SetStageUpAxis(root_stage, UsdGeom.Tokens.z)
root_stage.GetRootLayer().Save()
```

---

## Phase 7: Implementation order for Claude Code

### Step 1 — STL to USD (do first, get something visible)

1. Write `convert_stl_to_usd.py` — loads the STL, cleans it, writes `track_geometry.usda`
2. Write `create_marble.py` — creates `marble_actor.usda` with rigid body + sphere collider
3. Write `apply_physics.py` — creates `physics_config.usda` with materials, scene, CCD settings
4. Write `compose_scene.py` — creates the root `marble_coaster_scene.usda`
5. **Test**: open in `usdview` (ships with `usd-core` pip package) or Omniverse to verify geometry looks correct

### Step 2 — Parametric coil generator

1. Write `generate_coil.py` — creates `coil_geometry.usda` + `coil_properties.usda`
2. Accept coil parameters as command-line arguments or a JSON config file
3. **Test**: verify coil mesh appears correctly positioned relative to track

### Step 3 — Basic PhysX marble drop test (no EM yet)

1. Write `run_physics_test.py` — loads the composed scene, drops the marble onto the track under gravity, runs N steps
2. Use the `pxr` physics simulation API or Isaac Sim
3. **Test**: marble should fall, hit the track, roll down under gravity with friction
4. Tune: contact offset, CCD, friction coefficients, time step until rolling looks physical

### Step 4 — Analytical EM force (before training a PINN)

1. Write `analytical_bfield.py` — computes B-field for a finite solenoid using the Biot-Savart law (elliptic integral solution)
2. This serves as ground truth for PINN training AND as a fallback if PINN training hasn't converged
3. Write `em_force_injection.py` — reads marble position each step, computes EM force from analytical model, applies to PhysX
4. **Test**: marble should visibly accelerate when "fired" through the coil

### Step 5 — PhysicsNeMo PINN training

1. Write `generate_training_data.py` — sample the analytical B-field solution across the domain and parameter space
2. Write `train_pinn.py` — PhysicsNeMo Sym training script with the magnetostatic PDE loss
3. Write `evaluate_pinn.py` — compare PINN predictions vs analytical solution, plot error maps
4. **Test**: PINN should match analytical solution to <5% error across the domain

### Step 6 — Warp integration

1. Write `warp_em_kernel.py` — GPU kernel that queries the trained PINN and computes marble force
2. Write `simulation_loop.py` — orchestrates the full loop: read USD → Warp EM force → PhysX step → write USD
3. **Test**: full marble launch through coil, rolling along track

### Step 7 — Optimization (stretch goal)

1. Write `optimize_launch.py` — uses Warp's autodiff to optimize coil timing/voltage for target exit velocity
2. **Test**: optimizer converges on a pulse configuration that hits the target velocity

---

## Key dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `usd-core` | >=24.11 | OpenUSD Python bindings (pxr) |
| `trimesh` | >=4.0 | STL loading and mesh processing |
| `numpy` | >=1.24 | Array operations |
| `warp-lang` | >=1.0 | GPU simulation kernels |
| `physicsnemo` | >=0.1 | PINN training (PhysicsNeMo) |
| `torch` | >=2.0 | Backend for PhysicsNeMo |
| `scipy` | >=1.10 | Elliptic integrals for analytical B-field |
| `matplotlib` | >=3.7 | Visualization and validation plots |

### Optional (for full Omniverse integration)

| Package | Purpose |
|---------|---------|
| Isaac Sim | Full Omniverse physics runtime with GUI |
| Omniverse Kit | Extension development for live digital twin |
| `omni.physx` | PhysX simulation interface in Omniverse |

### Hardware

- **Minimum**: NVIDIA GPU with CUDA support (GTX 1060+) for Warp kernels
- **Recommended**: RTX 3060+ for PINN training + PhysX GPU pipeline
- **Ideal**: RTX 4070+ for real-time RTX rendering of the digital twin

---

## File structure

```
magnetic-coil-marble-coaster/
├── README.md
├── config/
│   └── coil_params.json          # coil geometry + electrical parameters
├── data/
│   └── track.stl                 # input track mesh
├── scripts/
│   ├── convert_stl_to_usd.py     # Phase 1: STL → USD
│   ├── create_marble.py          # Phase 1: marble rigid body
│   ├── apply_physics.py          # Phase 1: physics schemas + materials
│   ├── generate_coil.py          # Phase 2: parametric coil
│   ├── compose_scene.py          # Phase 6: layer composition
│   ├── analytical_bfield.py      # Phase 4: Biot-Savart reference
│   ├── em_force_injection.py     # Phase 4: force injection into PhysX
│   ├── generate_training_data.py # Phase 5: PINN training data
│   ├── train_pinn.py             # Phase 5: PhysicsNeMo training
│   ├── evaluate_pinn.py          # Phase 5: validation
│   ├── warp_em_kernel.py         # Phase 6: GPU EM force kernel
│   ├── simulation_loop.py        # Phase 6: full sim orchestration
│   └── optimize_launch.py        # Phase 7: differentiable optimization
├── usd/
│   ├── track_geometry.usda
│   ├── coil_geometry.usda
│   ├── coil_properties.usda
│   ├── marble_actor.usda
│   ├── physics_config.usda
│   └── marble_coaster_scene.usda
├── models/
│   └── pinn_checkpoint/          # trained PINN weights
└── results/
    ├── plots/                    # validation plots
    └── trajectories/             # simulated marble trajectories
```

---

## Notes for implementation

### Units consistency

Everything in this project uses **millimeters** as the base length unit because the STL is likely in mm (common for 3D printing). This means:

- Gravity = 9810 mm/s² (not 9.81 m/s²)
- Density = g/mm³ (steel = 0.0078 g/mm³, equivalent to 7800 kg/m³)
- μ₀ = 4π × 10⁻⁷ T·m/A = 4π × 10⁻⁴ T·mm/A (convert carefully)
- Forces come out in mN (millinewtons) — PhysX handles this fine

### The PINN vs analytical tradeoff

The analytical Biot-Savart solution (Phase 4) is perfectly adequate for a single coil geometry. The PINN (Phase 5) becomes valuable when you want to:
- Explore many coil configurations interactively (the PINN handles parameter sweeps instantly)
- Run the sim faster than real-time (PINN inference is ~100× faster than Biot-Savart per evaluation)
- Optimize coil design with gradient-based methods (the PINN is differentiable)

If you just need to simulate one fixed coil, skip Phase 5 and use the analytical solution.

### PhysX rolling behavior

PhysX doesn't model "true" rolling resistance — a sphere on a flat surface will roll forever. For realistic marble behavior, you may need to add a small linear/angular damping:

```python
physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(marble_xform.GetPrim())
physx_rb.CreateLinearDampingAttr(0.05)    # small air resistance
physx_rb.CreateAngularDampingAttr(0.02)   # rolling resistance approximation
```

Calibrate these against physical measurements of your marble decelerating on the real track.

### Sim-to-real feedback (future)

When you have the physical build:
1. Add photogates or hall-effect sensors at 2-3 points along the track
2. Measure marble velocity at each point
3. Compare against sim predictions
4. Tune friction coefficients, coil susceptibility, and damping to match
5. Re-run PINN training with measurement data as additional supervision points
