"""Build the Kit/USD scene for the vbench rig track.

    uv run python scripts/setup_rig_scene.py
    uv run python scripts/setup_rig_scene.py --release-x -230   # lower v_in

THE RIG'S OWN FRAME IS KEPT, DELIBERATELY. rig_assembly.stl is authored with
the coil centre at the origin, x along travel and z up -- the same convention
rig_geometry.json publishes. Rotating it to match the legacy scene's y-travel
would mean transforming every station and channel position too, and any error
there would look exactly like a physics disagreement. Instead the coil axis is
set to [1,0,0], which makes the sim's `z_along` identically the rig's x, so the
imported channel positions are usable with no conversion at all.

The coil prim sits at [0, ball_centre_y, coil_axis_z], NOT on the ball-centre
line: the rig deliberately rides its coil 0.300mm above the track axis so the
ball's centre height stays continuous through it, and a ball resting on the
datum is therefore 0.65mm off the coil axis. Mounting them concentric instead
is what puts a lip at the coil exit.

The coil is visual ONLY. Its bore is 14.0mm against a 12.7mm ball, and handing
0.65mm of clearance to a convex-decomposed collision mesh is a jam waiting to
happen; the track's own bore already constrains the ball.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "source" / "extensions" / "omni.marble.coaster"
                       / "omni" / "marble" / "coaster"))
from rig_profile import load_profile  # noqa: E402

VBENCH = ROOT.parent / "omnimarble-vbench"
RIG_STL = VBENCH / "track" / "stl" / "rig_assembly.stl"

TRACK_USD = ROOT / "usd" / "rig_track_geometry.usda"
MARBLE_USD = ROOT / "usd" / "rig_marble_actor.usda"
SCENE_USD = ROOT / "usd" / "rig_scene.usda"

GRAVITY_MM_S2 = 9810.0


def ball_centre_at(mesh, x, y, r_ball, search_top=250.0):
    """Ball-centre height resting in the bore at (x, y), or None off the track.

    Raycast down onto the channel floor and add the ball radius. That is exact
    here rather than approximate: the ball touches the bottom of the bore, so
    centre = floor + r. (Bore radius 6.70 - ball radius 6.35 = 0.35, which is
    precisely the 'ball sits 0.35 low' the rig quotes for the track bore.)
    """
    origins = np.array([[x, y, search_top]])
    directions = np.array([[0.0, 0.0, -1.0]])
    hits, _, _ = mesh.ray.intersects_location(origins, directions)
    if len(hits) == 0:
        return None
    return float(hits[:, 2].max()) + r_ball


def build_coil_barrel(profile, y, z, wall_outer=14.835):
    """The barrel through the coil section, which rig_assembly.stl omits.

    side_entry ends at x = -19.98 and side_exit starts at +19.98, so the
    assembly has a 40mm OPEN GAP at the coil -- because physically the wound
    FORMER is the barrel there, and the former is a jig part, not a track part.
    Import the assembly alone and the ball drops straight out of the bore at
    the coil, which is exactly what happened the first time.

    Modelled as an annulus on the COIL axis (0.300mm above the track axis, as
    the rig specifies) with the coil bore of 14.0mm, matching the printed
    former: groove root 14.385, barrel OR 14.835.
    """
    bore_r = profile.coil["bore_radius_mm"]
    # Butt against the printed ends (side_entry stops at -19.98, side_exit
    # starts at +19.98) rather than overlapping them: an overlap puts the
    # barrel's outer wall through the channel walls and leaves a lip for
    # the ball to strike. 128 sections so the bore is round enough that the
    # ball is not rolling across a polygon.
    length = 2 * 19.98
    tube = trimesh.creation.annulus(r_min=bore_r, r_max=wall_outer,
                                    height=length, sections=128)
    # annulus() is built along +Z; the barrel runs along travel (+X).
    tube.apply_transform(trimesh.transformations.rotation_matrix(
        math.pi / 2, [0, 1, 0]))
    tube.apply_translation([0.0, y, z])
    return tube


def build_track_usd(mesh):
    TRACK_USD.parent.mkdir(parents=True, exist_ok=True)
    if TRACK_USD.exists():
        TRACK_USD.unlink()
    stage = Usd.Stage.CreateNew(str(TRACK_USD))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Track")

    components = sorted(mesh.split(only_watertight=False),
                        key=lambda m: len(m.faces), reverse=True)
    if not components:
        components = [mesh]

    for i, comp in enumerate(components):
        name = "MainTrack" if i == 0 else f"Part_{i}"
        geom = UsdGeom.Mesh.Define(stage, f"/World/Track/{name}")
        geom.CreatePointsAttr([Gf.Vec3f(*v) for v in comp.vertices.tolist()])
        geom.CreateFaceVertexCountsAttr([3] * len(comp.faces))
        geom.CreateFaceVertexIndicesAttr(comp.faces.flatten().tolist())
        geom.CreateNormalsAttr([Gf.Vec3f(*n) for n in comp.vertex_normals.tolist()])
        geom.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        geom.CreateExtentAttr([Gf.Vec3f(*comp.bounds[0]), Gf.Vec3f(*comp.bounds[1])])
        geom.CreateSubdivisionSchemeAttr("none")
        print(f"  {name}: {len(comp.vertices)} verts, {len(comp.faces)} faces")

    stage.GetRootLayer().Save()
    print(f"Saved: {TRACK_USD.name}")
    return [("MainTrack" if i == 0 else f"Part_{i}") for i in range(len(components))]


def build_marble_usd(position, radius, mass_kg):
    if MARBLE_USD.exists():
        MARBLE_USD.unlink()
    stage = Usd.Stage.CreateNew(str(MARBLE_USD))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    UsdGeom.Xform.Define(stage, "/World")
    xform = UsdGeom.Xform.Define(stage, "/World/Marble")
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))

    sphere = UsdGeom.Sphere.Define(stage, "/World/Marble/Geom")
    sphere.CreateRadiusAttr(radius)
    sphere.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -radius),
                             Gf.Vec3f(radius, radius, radius)])

    prim = stage.GetPrimAtPath("/World/Marble")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    # Stage units are mm with mass in grams, matching the legacy scene.
    mass_api.CreateMassAttr(mass_kg * 1000.0)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/Marble/Geom"))

    stage.GetRootLayer().Save()
    print(f"Saved: {MARBLE_USD.name}  pos={tuple(round(v, 3) for v in position)} "
          f"r={radius} mass={mass_kg*1000:.2f}g")


def build_scene_usd(profile, track_parts, coil_pos):
    if SCENE_USD.exists():
        SCENE_USD.unlink()
    stage = Usd.Stage.CreateNew(str(SCENE_USD))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(GRAVITY_MM_S2)

    UsdGeom.Xform.Define(stage, "/World")
    for name in (TRACK_USD.name, MARBLE_USD.name):
        stage.GetRootLayer().subLayerPaths.append(f"./{name}")

    # Collision on every track part -- the ramp legs and cradles are separate
    # solids and the ball must not fall through any of them.
    for name in track_parts:
        prim = stage.OverridePrim(f"/World/Track/{name}")
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
        # The ball runs inside a bore, so the collider must be concave.
        # meshSimplification would round the channel out and let it escape.
        mesh_col.CreateApproximationAttr("none")

    mat_path = "/World/PhysicsMaterial"
    mat = UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(mat_path))
    mat.CreateStaticFrictionAttr(0.3)      # steel on PLA, unmeasured
    mat.CreateDynamicFrictionAttr(0.25)
    mat.CreateRestitutionAttr(0.2)         # a bore, not a bouncy track

    # Lighting. The legacy scene inherits this from visual_config.usda; without
    # it the viewport renders black and the rig looks broken when it is not.
    lights = UsdGeom.Xform.Define(stage, "/World/Lights")
    UsdGeom.Imageable(lights.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.default_)

    dome = stage.DefinePrim("/World/Lights/DomeLight", "DomeLight")
    dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(900.0)
    dome.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.86, 0.90, 1.0))

    # Raking key light down the length of the rig, so the bore reads as a bore.
    key = stage.DefinePrim("/World/Lights/KeyLight", "DistantLight")
    key.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(2400.0)
    key.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(1.5)
    key.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(1.0, 0.97, 0.92))
    key_x = UsdGeom.Xformable(key)
    key_x.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 25.0))

    fill = stage.DefinePrim("/World/Lights/FillLight", "DistantLight")
    fill.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(700.0)
    fill.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.75, 0.82, 1.0))
    UsdGeom.Xformable(fill).AddRotateXYZOp().Set(Gf.Vec3f(-20.0, 0.0, -140.0))

    # Give the track a visible surface so it is not a silhouette.
    for name in track_parts:
        prim = stage.OverridePrim(f"/World/Track/{name}")
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.64, 0.68)])

    marble_geom = stage.OverridePrim("/World/Marble/Geom")
    UsdGeom.Gprim(marble_geom).CreateDisplayColorAttr([Gf.Vec3f(0.80, 0.82, 0.85)])

    # Coil marker: visual only, deliberately no collision (see module docstring).
    coil = UsdGeom.Cylinder.Define(stage, "/World/CoilMarker")
    coil.CreateRadiusAttr(profile.coil["loop_center_radius_mm"])
    coil.CreateHeightAttr(profile.coil["length_mm"])
    coil.CreateAxisAttr("X")
    coil.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.45, 0.1)])
    UsdGeom.Xform(coil.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*coil_pos))
    UsdGeom.Imageable(coil.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)

    stage.GetRootLayer().Save()
    print(f"Saved: {SCENE_USD.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rig-profile", default="vbench_v0")
    parser.add_argument("--release-x", type=float, default=None,
                        help="release position in mm (default: near the ramp top). "
                             "Releasing part-way down is the rig's own way of "
                             "sweeping v_in")
    args = parser.parse_args()

    if not RIG_STL.exists():
        raise SystemExit(f"rig mesh not found: {RIG_STL}")

    profile = load_profile(ROOT, args.rig_profile)
    if profile.sensing_mode != "stations":
        raise SystemExit(f"profile {profile.name!r} is not a station rig")

    mesh = trimesh.load(str(RIG_STL))
    b = mesh.bounds
    print(f"Rig mesh: {len(mesh.faces)} faces, watertight={mesh.is_watertight}")
    print(f"  X[{b[0][0]:.2f},{b[1][0]:.2f}] Y[{b[0][1]:.2f},{b[1][1]:.2f}] "
          f"Z[{b[0][2]:.2f},{b[1][2]:.2f}]")

    track = profile.track
    r_ball = profile.marble["radius_mm"]
    ball_y = 13.0159            # rig_geometry datums.ball_centre_y
    coil_axis_z = 9.9893        # rig_geometry bores.coil_axis_z
    coil_pos = [0.0, ball_y, coil_axis_z]

    # Sanity: the flat zone must reproduce the published ball-centre datum.
    flat_z = ball_centre_at(mesh, 120.0, ball_y, r_ball)
    expected = track["ball_centre_z_mm"]
    if flat_z is None:
        raise SystemExit("no track surface found in the flat zone -- frame mismatch?")
    print(f"Ball-centre on the flat: {flat_z:.4f} mm "
          f"(rig datum {expected:.4f}, delta {flat_z - expected:+.4f})")
    if abs(flat_z - expected) > 0.5:
        raise SystemExit(
            "the ball-centre line does not match the rig's published datum; "
            "the mesh frame or the ball radius is wrong")

    release_x = args.release_x
    if release_x is None:
        release_x = b[0][0] + 12.0          # just inside the ramp's top end
    start_z = ball_centre_at(mesh, release_x, ball_y, r_ball)
    if start_z is None:
        raise SystemExit(f"no track surface at x={release_x}")
    drop_mm = start_z - flat_z
    v_ideal = math.sqrt(GRAVITY_MM_S2 / 1000.0 * (drop_mm / 1000.0) / 0.7)
    print(f"Release at x={release_x:.1f}: ball centre z={start_z:.3f}, "
          f"drop {drop_mm:.2f} mm")
    print(f"  ideal release velocity {v_ideal:.4f} m/s "
          f"(v = sqrt(g*h/0.7); the ball ROLLS, so 2/7 of the drop goes to spin)")
    print(f"  rig quotes {track['release_v_ideal_mps']:.4f} m/s from a "
          f"{track['ramp_rise_mm']:.2f} mm rise")

    barrel = build_coil_barrel(profile, ball_y, coil_axis_z)
    bb = barrel.bounds
    print(f"Coil barrel: bore r={profile.coil['bore_radius_mm']}mm on the coil "
          f"axis, x[{bb[0][0]:.1f},{bb[1][0]:.1f}] "
          f"(rig_assembly.stl leaves this 40mm span open -- the wound former "
          f"is the barrel there)")
    mesh = trimesh.util.concatenate([mesh, barrel])

    parts = build_track_usd(mesh)
    build_marble_usd([release_x, ball_y, start_z], r_ball,
                     profile.marble["mass_kg"])
    build_scene_usd(profile, parts, coil_pos)

    print(f"\nCoil prim at {coil_pos} with axis [1,0,0]:")
    print(f"  z_along is identically the rig's x, so the profile's channel "
          f"positions need no conversion.")
    print(f"  a ball on the datum sits r={abs(flat_z - coil_axis_z):.3f} mm off "
          f"the coil axis (the rig rides the coil 0.300 mm high on purpose)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
