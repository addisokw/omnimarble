"""Create a marble rigid body actor in USD."""

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parent.parent
USD_PATH = ROOT / "usd" / "marble_actor.usda"

# Defaults — can be overridden by reading track bounds
DEFAULT_POSITION = (0, 0, 50)
MARBLE_RADIUS = 5.0  # mm
MARBLE_DENSITY = 7.8e-3  # g/mm³ (steel)


def main():
    USD_PATH.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(USD_PATH))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Try to read track mesh to find the highest point for marble placement
    track_stl = ROOT / "data" / "track.stl"
    position = DEFAULT_POSITION
    if track_stl.exists():
        import numpy as np
        import trimesh
        raw = trimesh.load(str(track_stl))
        if isinstance(raw, trimesh.Scene):
            meshes = list(raw.geometry.values())
            tmesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        else:
            tmesh = raw
        verts = tmesh.vertices
        # Place marble at the start of the flat section (low Y end)
        # Track tube has radius ~5mm, centered at Z~0
        # Find vertices in the first 20mm of Y to get the center
        start_mask = verts[:, 1] < 20
        start_verts = verts[start_mask]
        if len(start_verts) > 0:
            cx = start_verts[:, 0].mean()
            track_top_z = start_verts[:, 2].max()
        else:
            cx = 0.0
            track_top_z = 5.0
        cy = 10.0  # Near start of track
        cz = track_top_z + MARBLE_RADIUS + 1  # On top of track tube
        position = (float(cx), float(cy), float(cz))
        print(f"Positioning marble at track start: {position}")

    UsdGeom.Xform.Define(stage, "/World")

    marble_xform = UsdGeom.Xform.Define(stage, "/World/Marble")
    marble_xform.AddTranslateOp().Set(Gf.Vec3d(*position))

    sphere = UsdGeom.Sphere.Define(stage, "/World/Marble/Geom")
    sphere.CreateRadiusAttr(MARBLE_RADIUS)
    sphere.CreateExtentAttr([
        Gf.Vec3f(-MARBLE_RADIUS, -MARBLE_RADIUS, -MARBLE_RADIUS),
        Gf.Vec3f(MARBLE_RADIUS, MARBLE_RADIUS, MARBLE_RADIUS),
    ])

    # Physics APIs
    marble_prim = stage.GetPrimAtPath("/World/Marble")
    UsdPhysics.RigidBodyAPI.Apply(marble_prim)

    geom_prim = stage.GetPrimAtPath("/World/Marble/Geom")
    UsdPhysics.CollisionAPI.Apply(geom_prim)

    mass_api = UsdPhysics.MassAPI.Apply(marble_prim)
    mass_api.CreateDensityAttr(MARBLE_DENSITY)

    stage.GetRootLayer().Save()
    print(f"Saved: {USD_PATH}")
    print(f"  Position: {position}")
    print(f"  Radius: {MARBLE_RADIUS} mm")
    print(f"  Density: {MARBLE_DENSITY} g/mm^3")


if __name__ == "__main__":
    main()
