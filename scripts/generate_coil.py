"""Generate parametric coil geometry and properties as USD layers."""

import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "coil_params.json"
GEOM_PATH = ROOT / "usd" / "coil_geometry.usda"
PROPS_PATH = ROOT / "usd" / "coil_properties.usda"

NUM_SEGMENTS = 32  # circumferential segments


def generate_hollow_cylinder_mesh(inner_r, outer_r, length, num_seg=NUM_SEGMENTS):
    """Generate a hollow cylinder (coil shell) mesh.

    4 vertex rings: inner-bottom, inner-top, outer-bottom, outer-top
    Faces: outer wall, inner wall, bottom cap, top cap
    """
    angles = np.linspace(0, 2 * math.pi, num_seg, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    half_l = length / 2

    # Vertex rings (along Y and Z for a coil with axis along X)
    # We'll generate in local frame with axis along X
    inner_bottom = np.column_stack([np.full(num_seg, -half_l), inner_r * cos_a, inner_r * sin_a])
    inner_top = np.column_stack([np.full(num_seg, half_l), inner_r * cos_a, inner_r * sin_a])
    outer_bottom = np.column_stack([np.full(num_seg, -half_l), outer_r * cos_a, outer_r * sin_a])
    outer_top = np.column_stack([np.full(num_seg, half_l), outer_r * cos_a, outer_r * sin_a])

    # Stack: [inner_bottom, inner_top, outer_bottom, outer_top]
    vertices = np.vstack([inner_bottom, inner_top, outer_bottom, outer_top])
    # Offsets for each ring
    ib, it, ob, ot = 0, num_seg, 2 * num_seg, 3 * num_seg

    faces = []

    for i in range(num_seg):
        j = (i + 1) % num_seg

        # Outer wall (ob-ot): two triangles per quad
        faces.append([ob + i, ob + j, ot + j])
        faces.append([ob + i, ot + j, ot + i])

        # Inner wall (ib-it): two triangles per quad (reversed winding)
        faces.append([ib + i, it + i, it + j])
        faces.append([ib + i, it + j, ib + j])

        # Bottom cap (ib-ob): ring between inner and outer at bottom
        faces.append([ib + i, ob + i, ob + j])
        faces.append([ib + i, ob + j, ib + j])

        # Top cap (it-ot): ring between inner and outer at top
        faces.append([it + i, ot + j, ot + i])
        faces.append([it + i, it + j, ot + j])

    return vertices, np.array(faces, dtype=np.int32)


def compute_vertex_normals(vertices, faces):
    """Compute per-vertex normals by averaging face normals."""
    normals = np.zeros_like(vertices)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    # Accumulate
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    # Normalize
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-10)
    return normals / lengths


def create_coil_geometry(params: dict):
    """Create coil geometry USD file."""
    inner_r = params["inner_radius_mm"]
    outer_r = params["outer_radius_mm"]
    length = params["length_mm"]
    position = params["position_mm"]
    axis = params.get("axis", [1, 0, 0])

    vertices, faces = generate_hollow_cylinder_mesh(inner_r, outer_r, length)
    normals = compute_vertex_normals(vertices, faces)

    GEOM_PATH.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(GEOM_PATH))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    UsdGeom.Xform.Define(stage, "/World")

    coil_xform = UsdGeom.Xform.Define(stage, "/World/Coil")
    coil_xform.AddTranslateOp().Set(Gf.Vec3d(*position))

    # Rotate coil to align with specified axis if not default (1,0,0)
    ax = np.array(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    default_axis = np.array([1, 0, 0])
    if not np.allclose(ax, default_axis):
        # Rotation from default X-axis to target axis
        cross = np.cross(default_axis, ax)
        dot = np.dot(default_axis, ax)
        if np.linalg.norm(cross) > 1e-6:
            angle = math.degrees(math.acos(np.clip(dot, -1, 1)))
            cross_norm = cross / np.linalg.norm(cross)
            coil_xform.AddRotateXYZOp().Set(Gf.Vec3f(
                angle * cross_norm[0],
                angle * cross_norm[1],
                angle * cross_norm[2],
            ))

    geom = UsdGeom.Mesh.Define(stage, "/World/Coil/Geom")
    points = [Gf.Vec3f(*v) for v in vertices.tolist()]
    geom.CreatePointsAttr(points)
    geom.CreateFaceVertexCountsAttr([3] * len(faces))
    geom.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    norm_list = [Gf.Vec3f(*n) for n in normals.tolist()]
    geom.CreateNormalsAttr(norm_list)
    geom.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    geom.CreateExtentAttr([Gf.Vec3f(*bmin), Gf.Vec3f(*bmax)])
    geom.CreateSubdivisionSchemeAttr("none")

    # Collision
    coil_prim = stage.GetPrimAtPath("/World/Coil/Geom")
    UsdPhysics.CollisionAPI.Apply(coil_prim)
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(coil_prim)
    mesh_col.CreateApproximationAttr("convexDecomposition")

    stage.GetRootLayer().Save()
    print(f"Saved: {GEOM_PATH}")
    print(f"  Vertices: {len(vertices)}, Faces: {len(faces)}")
    print(f"  Position: {position}")
    print(f"  Inner/Outer radius: {inner_r}/{outer_r} mm, Length: {length} mm")


def create_coil_properties(params: dict):
    """Create coil properties USD file with custom attributes."""
    PROPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(PROPS_PATH))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    coil_prim = stage.OverridePrim("/World/Coil")

    # Store parameters as custom attributes under coilLauncher namespace
    attr_map = {
        "coilLauncher:innerRadius": ("inner_radius_mm", Sdf.ValueTypeNames.Float),
        "coilLauncher:outerRadius": ("outer_radius_mm", Sdf.ValueTypeNames.Float),
        "coilLauncher:length": ("length_mm", Sdf.ValueTypeNames.Float),
        "coilLauncher:numTurns": ("num_turns", Sdf.ValueTypeNames.Int),
        "coilLauncher:wireDiameter": ("wire_diameter_mm", Sdf.ValueTypeNames.Float),
        "coilLauncher:maxCurrent": ("max_current_A", Sdf.ValueTypeNames.Float),
        "coilLauncher:pulseWidth": ("pulse_width_ms", Sdf.ValueTypeNames.Float),
        "coilLauncher:resistance": ("resistance_ohm", Sdf.ValueTypeNames.Float),
        "coilLauncher:inductance": ("inductance_uH", Sdf.ValueTypeNames.Float),
        "coilLauncher:supplyVoltage": ("supply_voltage_V", Sdf.ValueTypeNames.Float),
    }

    for attr_name, (key, type_name) in attr_map.items():
        val = params[key]
        attr = coil_prim.CreateAttribute(attr_name, type_name)
        if type_name == Sdf.ValueTypeNames.Int:
            attr.Set(int(val))
        else:
            attr.Set(float(val))

    stage.GetRootLayer().Save()
    print(f"Saved: {PROPS_PATH}")
    print(f"  Custom attributes: {list(attr_map.keys())}")


def main():
    params = json.loads(CONFIG_PATH.read_text())
    create_coil_geometry(params)
    create_coil_properties(params)


if __name__ == "__main__":
    main()
