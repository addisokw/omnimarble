"""Convert track STL to OpenUSD geometry."""

import sys
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom

ROOT = Path(__file__).resolve().parent.parent
STL_PATH = ROOT / "data" / "track.stl"
USD_PATH = ROOT / "usd" / "track_geometry.usda"


def load_and_clean(path: Path) -> trimesh.Trimesh:
    raw = trimesh.load(str(path))
    if isinstance(raw, trimesh.Scene):
        meshes = list(raw.geometry.values())
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    else:
        mesh = raw

    mesh.fix_normals()
    # Remove degenerate and duplicate faces
    mask = mesh.nondegenerate_faces()
    mesh.update_faces(mask)
    unique = mesh.unique_faces()
    mesh.update_faces(unique)

    if len(mesh.faces) > 100_000:
        target = 100_000
        print(f"Decimating from {len(mesh.faces)} to {target} faces")
        mesh = mesh.simplify_quadric_decimation(target)

    return mesh


def split_components(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    components = mesh.split(only_watertight=False)
    # Sort by face count descending so largest is first
    components = sorted(components, key=lambda m: len(m.faces), reverse=True)
    return components


def add_mesh_to_stage(stage, prim_path: str, mesh: trimesh.Trimesh):
    geom = UsdGeom.Mesh.Define(stage, prim_path)
    points = [Gf.Vec3f(*v) for v in mesh.vertices.tolist()]
    geom.CreatePointsAttr(points)
    geom.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    geom.CreateFaceVertexIndicesAttr(mesh.faces.flatten().tolist())

    # Vertex normals
    normals = [Gf.Vec3f(*n) for n in mesh.vertex_normals.tolist()]
    geom.CreateNormalsAttr(normals)
    geom.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    # Extent
    bounds = mesh.bounds  # (2, 3) array: min, max
    geom.CreateExtentAttr([Gf.Vec3f(*bounds[0]), Gf.Vec3f(*bounds[1])])
    geom.CreateSubdivisionSchemeAttr("none")


def main():
    if not STL_PATH.exists():
        print(f"ERROR: {STL_PATH} not found. Place track.stl in the data/ directory.")
        sys.exit(1)

    print(f"Loading {STL_PATH} ...")
    mesh = load_and_clean(STL_PATH)
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    print(f"  Bounding box: {mesh.bounds[0]} to {mesh.bounds[1]}")

    components = split_components(mesh)
    print(f"  Connected components: {len(components)}")

    # Create USD stage
    USD_PATH.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(USD_PATH))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)  # mm
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    world = UsdGeom.Xform.Define(stage, "/World")
    track = UsdGeom.Xform.Define(stage, "/World/Track")

    # Largest component → MainTrack
    add_mesh_to_stage(stage, "/World/Track/MainTrack", components[0])
    print(f"  MainTrack: {len(components[0].vertices)} verts, {len(components[0].faces)} faces")

    # Remaining → Support_N
    for i, comp in enumerate(components[1:]):
        add_mesh_to_stage(stage, f"/World/Track/Support_{i}", comp)
        print(f"  Support_{i}: {len(comp.vertices)} verts, {len(comp.faces)} faces")

    stage.GetRootLayer().Save()
    print(f"\nSaved: {USD_PATH}")

    # Suggest marble start position
    highest_z = mesh.bounds[1][2]
    center_xy = mesh.bounds.mean(axis=0)[:2]
    print(f"  Suggested marble start: ({center_xy[0]:.1f}, {center_xy[1]:.1f}, {highest_z + 10:.1f})")


if __name__ == "__main__":
    main()
