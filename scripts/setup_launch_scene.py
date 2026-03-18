"""Set up the coilgun launch scene with the starter-slope track.

Converts track-starter-slope.stl to USD, positions the marble at the top
of the slope, places the coil in the flat section, and adds IR sensor
visualizations to the scene.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parent.parent
STL_PATH = ROOT / "data" / "track-starter-slope.stl"
CONFIG_PATH = ROOT / "config" / "coil_params.json"
TRACK_USD = ROOT / "usd" / "track_geometry.usda"
MARBLE_USD = ROOT / "usd" / "marble_actor.usda"
SCENE_USD = ROOT / "usd" / "marble_coaster_scene.usda"

MARBLE_RADIUS = 5.0  # mm


def load_track():
    """Load and clean the track STL."""
    raw = trimesh.load(str(STL_PATH))
    if isinstance(raw, trimesh.Scene):
        meshes = list(raw.geometry.values())
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    else:
        mesh = raw
    mesh.fix_normals()
    mask = mesh.nondegenerate_faces()
    mesh.update_faces(mask)
    unique = mesh.unique_faces()
    mesh.update_faces(unique)
    return mesh


def analyze_track(mesh):
    """Find key positions on the track for placement."""
    verts = mesh.vertices

    # Track layout (along Y axis):
    #   Y=-128  [LOW END - marble start, shallow slope down]
    #   Y=0     [flat section]
    #   Y=200   [flat section ends]
    #   Y=225   [steep slope begins]
    #   Y=335   [HIGH END - top of steep slope, Z=180]
    #
    # The marble starts at the LOW Y end, rolls along the flat toward
    # positive Y, gets launched by the coil, and flies up the steep slope.

    # Track channel center X
    flat_mask = verts[:, 2] < 10
    flat_verts = verts[flat_mask]
    channel_x = flat_verts[:, 0].mean()
    flat_y_min = flat_verts[:, 1].min()
    flat_y_max = flat_verts[:, 1].max()

    # Marble start: low Y end of track (shallow end), near the edge
    low_y_mask = verts[:, 1] < flat_y_min + 20
    low_verts = verts[low_y_mask]
    start_y = -20.0  # near the low-Y end of track
    start_x = low_verts[:, 0].mean() if len(low_verts) > 0 else channel_x
    # Z at start position
    start_z_mask = np.abs(verts[:, 1] - start_y) < 8
    start_z_track = verts[start_z_mask, 2].max() if start_z_mask.sum() > 0 else 5.0
    start_z = start_z_track + MARBLE_RADIUS + 2

    # Bottom of steep slope (where flat meets the ramp up)
    slope_bottom_y = 200.0
    for y_test in range(200, 260, 5):
        mask = np.abs(verts[:, 1] - y_test) < 6
        if mask.sum() > 2:
            z_top = verts[mask, 2].max()
            if z_top > 15:
                slope_bottom_y = y_test
                break

    # Coil position: closer to the marble start so the launch happens
    # earlier, giving the marble the full flat+slope to travel.
    # Place roughly 1/4 into the flat section from the start.
    coil_y = flat_y_min + (flat_y_max - flat_y_min) * 0.25

    # Track top surface Z at coil position
    coil_mask = np.abs(verts[:, 1] - coil_y) < 8
    coil_z = verts[coil_mask, 2].max() if coil_mask.sum() > 0 else 5.0

    return {
        "marble_start": (float(start_x), float(start_y), float(start_z)),
        "flat_y_range": (float(flat_y_min), float(flat_y_max)),
        "slope_bottom_y": float(slope_bottom_y),
        "coil_position": (float(channel_x), float(coil_y), float(coil_z)),
        "channel_x": float(channel_x),
    }


def create_track_usd(mesh):
    """Convert track mesh to USD."""
    TRACK_USD.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(TRACK_USD))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Track")

    # Split into components
    components = mesh.split(only_watertight=False)
    components = sorted(components, key=lambda m: len(m.faces), reverse=True)

    for i, comp in enumerate(components):
        name = "MainTrack" if i == 0 else f"Support_{i-1}"
        prim_path = f"/World/Track/{name}"
        geom = UsdGeom.Mesh.Define(stage, prim_path)
        points = [Gf.Vec3f(*v) for v in comp.vertices.tolist()]
        geom.CreatePointsAttr(points)
        geom.CreateFaceVertexCountsAttr([3] * len(comp.faces))
        geom.CreateFaceVertexIndicesAttr(comp.faces.flatten().tolist())
        normals = [Gf.Vec3f(*n) for n in comp.vertex_normals.tolist()]
        geom.CreateNormalsAttr(normals)
        geom.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        geom.CreateExtentAttr([Gf.Vec3f(*comp.bounds[0]), Gf.Vec3f(*comp.bounds[1])])
        geom.CreateSubdivisionSchemeAttr("none")
        print(f"  {name}: {len(comp.vertices)} verts, {len(comp.faces)} faces")

    stage.GetRootLayer().Save()
    print(f"Saved: {TRACK_USD}")


def create_marble_usd(position):
    """Create marble actor at the given position."""
    MARBLE_USD.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(MARBLE_USD))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    UsdGeom.Xform.Define(stage, "/World")
    marble_xform = UsdGeom.Xform.Define(stage, "/World/Marble")
    marble_xform.AddTranslateOp().Set(Gf.Vec3d(*position))

    sphere = UsdGeom.Sphere.Define(stage, "/World/Marble/Geom")
    sphere.CreateRadiusAttr(MARBLE_RADIUS)
    sphere.CreateExtentAttr([
        Gf.Vec3f(-MARBLE_RADIUS, -MARBLE_RADIUS, -MARBLE_RADIUS),
        Gf.Vec3f(MARBLE_RADIUS, MARBLE_RADIUS, MARBLE_RADIUS),
    ])

    # Rigid body
    marble_prim = stage.GetPrimAtPath("/World/Marble")
    UsdPhysics.RigidBodyAPI.Apply(marble_prim)
    mass_api = UsdPhysics.MassAPI.Apply(marble_prim)
    volume = (4 / 3) * math.pi * MARBLE_RADIUS ** 3
    mass_g = volume * 7.8e-3
    mass_api.CreateMassAttr(mass_g)

    # Collision
    geom_prim = stage.GetPrimAtPath("/World/Marble/Geom")
    UsdPhysics.CollisionAPI.Apply(geom_prim)

    stage.GetRootLayer().Save()
    print(f"Saved: {MARBLE_USD}")
    print(f"  Position: {position}")
    print(f"  Mass: {mass_g:.2f} g")


def create_ir_sensor_visual(stage, name, position, axis, color):
    """Create a visual representation of an IR break-beam sensor.

    Two small cylinders (emitter + receiver) on opposite sides of the bore,
    with a thin red/green line between them representing the beam.
    """
    sensor_xform = UsdGeom.Xform.Define(stage, f"/World/Sensors/{name}")
    sensor_xform.AddTranslateOp().Set(Gf.Vec3d(*position))

    bore_radius = 12.0  # inner radius of coil = bore size
    sensor_size = 2.0  # mm, small cylinder
    sensor_length = 3.0

    # Emitter (one side of bore)
    emitter = UsdGeom.Cylinder.Define(stage, f"/World/Sensors/{name}/Emitter")
    emitter.CreateRadiusAttr(sensor_size)
    emitter.CreateHeightAttr(sensor_length)
    emitter.CreateAxisAttr("X")
    emitter_xform = UsdGeom.Xformable(emitter.GetPrim())
    emitter_xform.AddTranslateOp().Set(Gf.Vec3d(bore_radius + sensor_length / 2, 0, 0))

    # Receiver (opposite side)
    receiver = UsdGeom.Cylinder.Define(stage, f"/World/Sensors/{name}/Receiver")
    receiver.CreateRadiusAttr(sensor_size)
    receiver.CreateHeightAttr(sensor_length)
    receiver.CreateAxisAttr("X")
    receiver_xform = UsdGeom.Xformable(receiver.GetPrim())
    receiver_xform.AddTranslateOp().Set(Gf.Vec3d(-(bore_radius + sensor_length / 2), 0, 0))

    # Beam line (thin cylinder connecting them)
    beam = UsdGeom.Cylinder.Define(stage, f"/World/Sensors/{name}/Beam")
    beam.CreateRadiusAttr(0.3)
    beam.CreateHeightAttr(2 * bore_radius)
    beam.CreateAxisAttr("X")

    # Color the beam
    beam.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    emitter.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.2, 0.2)])  # dark housing
    receiver.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.2, 0.2)])


def main():
    print("=== Setting up Coilgun Launch Scene ===\n")

    if not STL_PATH.exists():
        print(f"ERROR: {STL_PATH} not found")
        sys.exit(1)

    # Load and analyze track
    print("Loading track...")
    mesh = load_track()
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    print(f"  Bounds: {mesh.bounds[0]} to {mesh.bounds[1]}")

    info = analyze_track(mesh)
    print(f"\n  Track analysis:")
    print(f"    Marble start (top of slope): {info['marble_start']}")
    print(f"    Flat section Y range: {info['flat_y_range']}")
    print(f"    Slope bottom Y: {info['slope_bottom_y']}")
    print(f"    Coil position: {info['coil_position']}")

    # Convert track to USD
    print("\nConverting track STL to USD...")
    create_track_usd(mesh)

    # Create marble at top of slope
    print("\nCreating marble...")
    create_marble_usd(info["marble_start"])

    # Update coil config
    print("\nUpdating coil config...")
    params = json.loads(CONFIG_PATH.read_text())
    coil_pos = list(info["coil_position"])
    params["position_mm"] = coil_pos
    params["axis"] = [0, 1, 0]  # Along Y (track direction)

    # IR gate positions relative to coil center along coil axis [0,1,0].
    # z_along = marble_Y - coil_Y
    # Marble comes from LOW Y (negative z_along), launched toward HIGH Y (positive).
    gate_positions = {
        "vel_in_1": -60.0,   # approach velocity measurement
        "vel_in_2": -40.0,   # approach velocity measurement
        "entry":    -20.0,   # fires the coil
        "cutoff":     5.0,   # cuts the pulse (RLC pulse is sub-ms, so this is a safety net)
        "vel_out_1":  60.0,  # exit velocity measurement (wide spacing for fast marble)
        "vel_out_2": 120.0,  # exit velocity measurement
    }
    params["gate_positions"] = gate_positions
    # Keep old keys for backward compat
    params["sensor_entry_offset_mm"] = gate_positions["entry"]
    params["sensor_cutoff_offset_mm"] = gate_positions["cutoff"]

    CONFIG_PATH.write_text(json.dumps(params, indent=2) + "\n")
    print(f"  Coil at: {coil_pos}")
    print(f"  Axis: {params['axis']}")
    print(f"  Entry sensor: +{params['sensor_entry_offset_mm']}mm from center")
    print(f"  Cutoff sensor: {params['sensor_cutoff_offset_mm']}mm from center")

    # Regenerate coil geometry with new position
    print("\nRegenerating coil geometry...")
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_coil import create_coil_geometry, create_coil_properties, derive_electrical_params
    params = derive_electrical_params(params)
    create_coil_geometry(params)
    create_coil_properties(params)

    # Create scene with IR sensors
    print("\nComposing scene with IR sensors...")
    SCENE_USD.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(SCENE_USD))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Physics scene
    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics_scene.CreateGravityMagnitudeAttr(9810.0)  # mm/s^2

    # Reference sublayers
    world = UsdGeom.Xform.Define(stage, "/World")

    # Add track, marble, coil as sublayer references
    for usd_file in ["track_geometry.usda", "marble_actor.usda",
                      "coil_geometry.usda", "coil_properties.usda",
                      "visual_config.usda"]:
        usd_path = ROOT / "usd" / usd_file
        if usd_path.exists():
            stage.GetRootLayer().subLayerPaths.append(f"./{usd_file}")

    # Add collision to track
    track_prim = stage.OverridePrim("/World/Track/MainTrack")
    UsdPhysics.CollisionAPI.Apply(track_prim)
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(track_prim)
    mesh_col.CreateApproximationAttr("meshSimplification")

    # Physics material for track
    material_path = "/World/PhysicsMaterial"
    mat = UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(material_path))
    mat.CreateStaticFrictionAttr(0.3)
    mat.CreateDynamicFrictionAttr(0.2)
    mat.CreateRestitutionAttr(0.5)

    # IR Gate visuals
    coil_axis = np.array(params["axis"], dtype=float)
    coil_axis = coil_axis / np.linalg.norm(coil_axis)
    coil_center = np.array(coil_pos)

    UsdGeom.Xform.Define(stage, "/World/Sensors")

    # Color scheme: blue=velocity, green=entry, red=cutoff, cyan=exit velocity
    gate_colors = {
        "vel_in_1":  (0.2, 0.4, 1.0),   # blue
        "vel_in_2":  (0.2, 0.4, 1.0),   # blue
        "entry":     (0.0, 1.0, 0.0),   # green
        "cutoff":    (1.0, 0.0, 0.0),   # red
        "vel_out_1": (0.0, 1.0, 1.0),   # cyan
        "vel_out_2": (0.0, 1.0, 1.0),   # cyan
    }
    gate_labels = {
        "vel_in_1":  "VelIn_1",
        "vel_in_2":  "VelIn_2",
        "entry":     "Entry",
        "cutoff":    "Cutoff",
        "vel_out_1": "VelOut_1",
        "vel_out_2": "VelOut_2",
    }

    print(f"\n  IR Gates (relative to coil center at Y={coil_pos[1]:.0f}):")
    for gate_name, offset in gate_positions.items():
        gate_world_pos = coil_center + offset * coil_axis
        label = gate_labels[gate_name]
        color = gate_colors[gate_name]
        create_ir_sensor_visual(stage, label, tuple(gate_world_pos), coil_axis, color=color)
        print(f"    {label:10s}: Y={gate_world_pos[1]:7.1f}mm  (offset={offset:+.0f}mm)  "
              f"{'[FIRE]' if gate_name == 'entry' else '[CUT]' if gate_name == 'cutoff' else '[VEL]'}")

    stage.GetRootLayer().Save()
    print(f"\nSaved scene: {SCENE_USD}")

    print(f"\n{'='*60}")
    print(f"Scene ready!")
    print(f"  Marble starts at Y={info['marble_start'][1]:.0f}mm, rolls toward coil at Y={coil_pos[1]:.0f}mm")
    print(f"  Blue gates measure approach speed")
    print(f"  Green gate fires the coil")
    print(f"  Red gate cuts the pulse")
    print(f"  Cyan gates measure exit speed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
