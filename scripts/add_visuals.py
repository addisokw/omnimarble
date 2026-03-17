"""Add lighting and display materials to the USD scene."""

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

ROOT = Path(__file__).resolve().parent.parent
USD_PATH = ROOT / "usd" / "visual_config.usda"


def main():
    USD_PATH.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(USD_PATH))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # === Lighting ===

    # Key light — dome light for ambient
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(500)
    dome.CreateColorAttr(Gf.Vec3f(0.9, 0.93, 1.0))

    # Main directional light
    distant = UsdLux.DistantLight.Define(stage, "/World/Lights/KeyLight")
    distant.CreateIntensityAttr(3000)
    distant.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
    distant.CreateAngleAttr(1.0)
    xform = UsdGeom.Xformable(distant.GetPrim())
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    # Fill light from opposite side
    fill = UsdLux.DistantLight.Define(stage, "/World/Lights/FillLight")
    fill.CreateIntensityAttr(1000)
    fill.CreateColorAttr(Gf.Vec3f(0.7, 0.8, 1.0))
    fill.CreateAngleAttr(2.0)
    xform2 = UsdGeom.Xformable(fill.GetPrim())
    xform2.AddRotateXYZOp().Set(Gf.Vec3f(-30, -120, 0))

    # === Display Materials (UsdPreviewSurface) ===

    # Track material — metallic gray
    track_mtl = UsdShade.Material.Define(stage, "/World/Materials/TrackDisplayMtl")
    track_shader = UsdShade.Shader.Define(stage, "/World/Materials/TrackDisplayMtl/Shader")
    track_shader.CreateIdAttr("UsdPreviewSurface")
    track_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.55))
    track_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.7)
    track_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    track_mtl.CreateSurfaceOutput().ConnectToSource(track_shader.ConnectableAPI(), "surface")

    # Marble material — shiny steel
    marble_mtl = UsdShade.Material.Define(stage, "/World/Materials/MarbleDisplayMtl")
    marble_shader = UsdShade.Shader.Define(stage, "/World/Materials/MarbleDisplayMtl/Shader")
    marble_shader.CreateIdAttr("UsdPreviewSurface")
    marble_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.2, 0.15))
    marble_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.9)
    marble_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.15)
    marble_mtl.CreateSurfaceOutput().ConnectToSource(marble_shader.ConnectableAPI(), "surface")

    # Coil material — copper
    coil_mtl = UsdShade.Material.Define(stage, "/World/Materials/CoilDisplayMtl")
    coil_shader = UsdShade.Shader.Define(stage, "/World/Materials/CoilDisplayMtl/Shader")
    coil_shader.CreateIdAttr("UsdPreviewSurface")
    coil_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.72, 0.45, 0.2))
    coil_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.95)
    coil_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
    coil_mtl.CreateSurfaceOutput().ConnectToSource(coil_shader.ConnectableAPI(), "surface")

    # === Bind display materials ===

    # Track
    track_prim = stage.OverridePrim("/World/Track")
    UsdShade.MaterialBindingAPI.Apply(track_prim).Bind(track_mtl)

    # Marble
    marble_prim = stage.OverridePrim("/World/Marble/Geom")
    UsdShade.MaterialBindingAPI.Apply(marble_prim).Bind(marble_mtl)

    # Coil
    coil_prim = stage.OverridePrim("/World/Coil/Geom")
    UsdShade.MaterialBindingAPI.Apply(coil_prim).Bind(coil_mtl)

    # === Ground plane ===
    ground = UsdGeom.Mesh.Define(stage, "/World/GroundPlane")
    size = 500
    ground.CreatePointsAttr([
        Gf.Vec3f(-size, -size, 0),
        Gf.Vec3f(size, -size, 0),
        Gf.Vec3f(size, size, 0),
        Gf.Vec3f(-size, size, 0),
    ])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    ground.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    ground.CreateExtentAttr([Gf.Vec3f(-size, -size, 0), Gf.Vec3f(size, size, 0)])
    ground.CreateSubdivisionSchemeAttr("none")

    ground_mtl = UsdShade.Material.Define(stage, "/World/Materials/GroundMtl")
    ground_shader = UsdShade.Shader.Define(stage, "/World/Materials/GroundMtl/Shader")
    ground_shader.CreateIdAttr("UsdPreviewSurface")
    ground_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.25, 0.25, 0.28))
    ground_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    ground_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    ground_mtl.CreateSurfaceOutput().ConnectToSource(ground_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_mtl)

    stage.GetRootLayer().Save()
    print(f"Saved: {USD_PATH}")
    print("  Added: DomeLight, KeyLight, FillLight")
    print("  Added: Track (gray), Marble (red), Coil (copper), Ground (dark) materials")


if __name__ == "__main__":
    main()
