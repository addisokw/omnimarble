"""Compose all USD layers into a single scene."""

from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parent.parent
USD_PATH = ROOT / "usd" / "marble_coaster_scene.usda"

# Sublayers listed strongest-first (physics overrides > actors > geometry)
SUBLAYERS = [
    "./visual_config.usda",
    "./physics_config.usda",
    "./coil_properties.usda",
    "./coil_geometry.usda",
    "./marble_actor.usda",
    "./track_geometry.usda",
]


def main():
    USD_PATH.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(USD_PATH))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root_layer = stage.GetRootLayer()

    # Only add sublayers that exist
    existing = []
    for sl in SUBLAYERS:
        full_path = USD_PATH.parent / sl
        if full_path.exists():
            existing.append(sl)
        else:
            print(f"  Skipping (not found): {sl}")

    root_layer.subLayerPaths = existing
    root_layer.Save()

    # Verify — reopen and traverse
    print(f"\nSaved: {USD_PATH}")
    print(f"Sublayers: {existing}")

    verify_stage = Usd.Stage.Open(str(USD_PATH))
    print("\nPrim hierarchy:")
    for prim in verify_stage.Traverse():
        indent = "  " * prim.GetPath().pathElementCount
        apis = []
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            apis.append("RigidBody")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            apis.append("Collision")
        if prim.HasAPI(UsdPhysics.MassAPI):
            apis.append("Mass")
        api_str = f" [{', '.join(apis)}]" if apis else ""
        print(f"{indent}{prim.GetPath()}{api_str}")


if __name__ == "__main__":
    main()
