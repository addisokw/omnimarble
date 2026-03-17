"""Apply physics scene, collision, and material properties to the USD scene."""

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parent.parent
USD_PATH = ROOT / "usd" / "physics_config.usda"


def main():
    USD_PATH.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(USD_PATH))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Physics scene with gravity
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9810.0)  # mm/s²

    # Track collision — override prim so it layers on top of track_geometry
    track_prim = stage.OverridePrim("/World/Track")
    UsdPhysics.CollisionAPI.Apply(track_prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(track_prim)
    mesh_collision.CreateApproximationAttr("none")  # Use exact triangle mesh

    # Physics materials
    # Track material
    track_mtl = UsdShade.Material.Define(stage, "/World/Materials/TrackPhysicsMtl")
    track_phys = UsdPhysics.MaterialAPI.Apply(track_mtl.GetPrim())
    track_phys.CreateStaticFrictionAttr(0.4)
    track_phys.CreateDynamicFrictionAttr(0.3)
    track_phys.CreateRestitutionAttr(0.3)

    # Marble material
    marble_mtl = UsdShade.Material.Define(stage, "/World/Materials/MarblePhysicsMtl")
    marble_phys = UsdPhysics.MaterialAPI.Apply(marble_mtl.GetPrim())
    marble_phys.CreateStaticFrictionAttr(0.5)
    marble_phys.CreateDynamicFrictionAttr(0.35)
    marble_phys.CreateRestitutionAttr(0.6)

    # Bind materials
    track_binding = UsdShade.MaterialBindingAPI.Apply(track_prim)
    track_binding.Bind(track_mtl, materialPurpose="physics")

    marble_prim = stage.OverridePrim("/World/Marble")
    marble_binding = UsdShade.MaterialBindingAPI.Apply(marble_prim)
    marble_binding.Bind(marble_mtl, materialPurpose="physics")

    # NOTE: PhysxSchema settings (CCD, contact offset, solver iterations,
    # linear/angular damping, fixed timestep) should be configured in
    # NVIDIA Omniverse or Isaac Sim — these APIs are not available in usd-core.

    stage.GetRootLayer().Save()
    print(f"Saved: {USD_PATH}")
    print("  Physics scene: gravity = (0,0,-1) @ 9810 mm/s^2")
    print("  Track collision: exact triangle mesh")
    print("  Track material: us=0.4, ud=0.3, e=0.3")
    print("  Marble material: us=0.5, ud=0.35, e=0.6")


if __name__ == "__main__":
    main()
