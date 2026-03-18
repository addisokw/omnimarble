"""Generate parametric coil geometry and properties as USD layers.

Phase 2: Multi-layer winding model, RLC circuit derivation, AC resistance.
"""

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

# Import RLC circuit module
from rlc_circuit import (
    compute_ac_resistance,
    compute_dc_resistance,
    compute_multilayer_inductance,
    compute_rlc_params,
    compute_winding_geometry,
)


def derive_electrical_params(params: dict) -> dict:
    """Derive resistance, inductance, and RLC parameters from config.

    Computes multi-layer winding geometry, DC/AC resistance, multilayer
    inductance, and full RLC discharge characteristics.

    Returns params with added derived fields.
    """
    # --- Winding geometry ---
    geom = compute_winding_geometry(params)

    print(f"  --- Winding Geometry ---")
    print(f"  Wire: {params['wire_diameter_mm']}mm dia ({_wire_gauge(params['wire_diameter_mm'])} AWG)")
    print(f"  Insulation: {params.get('insulation_thickness_mm', 0.035)}mm, "
          f"pitch: {geom['wire_pitch_mm']:.3f}mm")
    print(f"  Turns/layer: {geom['turns_per_layer']}, Layers: {geom['num_layers']}")
    print(f"  Actual outer radius: {geom['actual_outer_radius_mm']:.2f}mm "
          f"(spec: {params['outer_radius_mm']}mm)"
          f"{'' if geom['fits_in_spec'] else ' *** EXCEEDS SPEC ***'}")
    print(f"  Wire length: {geom['wire_length_mm']:.1f}mm = {geom['wire_length_mm']/1000:.2f}m")
    print(f"  Wire mass: {geom['wire_mass_g']:.2f}g")
    print(f"  Mean radius: {geom['mean_radius_mm']:.2f}mm")

    # --- DC Resistance ---
    temp = params.get("ambient_temperature_C", 20.0)
    R_dc = compute_dc_resistance(geom["wire_length_mm"], geom["wire_cross_section_mm2"], temp)

    # --- Inductance (multilayer Wheeler's) ---
    L_uH = compute_multilayer_inductance(
        params["num_turns"],
        geom["mean_radius_mm"],
        params["length_mm"],
        geom["winding_depth_mm"],
    )
    L_H = L_uH * 1e-6

    # --- Total DC resistance ---
    R_esr = params.get("esr_ohm", 0.01)
    R_wiring = params.get("wiring_resistance_ohm", 0.02)
    R_total_dc = R_dc + R_esr + R_wiring

    # --- RLC parameters (needed for AC frequency) ---
    C = params.get("capacitance_uF", 1000.0) * 1e-6
    V0 = params.get("charge_voltage_V", 400.0)
    alpha = R_total_dc / (2 * L_H)
    omega_0 = 1.0 / math.sqrt(L_H * C)
    zeta = alpha / omega_0

    # Discharge frequency for AC resistance calculation
    if zeta < 1.0:
        omega_d = math.sqrt(omega_0 ** 2 - alpha ** 2)
        freq_Hz = omega_d / (2 * math.pi)
    else:
        freq_Hz = omega_0 / (2 * math.pi)  # approximate

    # --- AC Resistance ---
    ac_info = compute_ac_resistance(
        R_dc, params["wire_diameter_mm"], geom["num_layers"], freq_Hz, temp
    )
    R_coil_ac = ac_info["R_ac_ohm"]
    R_total_ac = R_coil_ac + R_esr + R_wiring

    print(f"\n  --- Electrical Parameters ---")
    print(f"  DC resistance (coil): {R_dc:.5f} ohm")
    print(f"  ESR: {R_esr} ohm, Wiring: {R_wiring} ohm")
    print(f"  Total DC resistance: {R_total_dc:.5f} ohm")
    print(f"  Discharge frequency: {freq_Hz:.0f} Hz")
    print(f"  Skin depth: {ac_info['skin_depth_mm']:.3f}mm")
    print(f"  AC resistance factor: {ac_info['total_ac_factor']:.3f} "
          f"(skin={ac_info['ac_resistance_factor']:.3f}, "
          f"prox={ac_info['proximity_factor']:.3f})")
    print(f"  Total AC resistance: {R_total_ac:.5f} ohm")
    print(f"  Inductance: {L_uH:.2f} uH")

    # --- Full RLC derivation ---
    rlc_input = {
        "capacitance_uF": params.get("capacitance_uF", 1000.0),
        "charge_voltage_V": V0,
        "inductance_uH": L_uH,
        "total_resistance_ohm": R_total_ac,
    }
    rlc = compute_rlc_params(rlc_input)

    print(f"\n  --- RLC Discharge Parameters ---")
    print(f"  Regime: {rlc['regime']}, zeta={rlc['zeta']:.4f}")
    print(f"  omega_0: {rlc['omega_0']:.1f} rad/s")
    if "omega_d" in rlc:
        print(f"  omega_d: {rlc['omega_d']:.1f} rad/s")
    print(f"  Peak current: {rlc['peak_current_A']:.1f} A")
    print(f"  Time to peak: {rlc['time_to_peak_s']*1e6:.1f} us")
    if "zero_crossing_s" in rlc:
        print(f"  Zero crossing (diode clamp): {rlc['zero_crossing_s']*1e3:.3f} ms")
    print(f"  Effective pulse duration: {rlc['effective_pulse_duration_s']*1e3:.3f} ms")
    print(f"  Stored energy: {rlc['stored_energy_J']:.2f} J")
    print(f"  Switch type: {params.get('switch_type', 'MOSFET')}")
    print(f"  Flyback diode: {params.get('has_flyback_diode', True)}")

    # --- Store derived values ---
    params["resistance_ohm"] = R_dc
    params["inductance_uH"] = L_uH
    params["_total_resistance_dc_ohm"] = R_total_dc
    params["_total_resistance_ac_ohm"] = R_total_ac
    params["_peak_current_A"] = rlc["peak_current_A"]
    params["_time_to_peak_us"] = rlc["time_to_peak_s"] * 1e6
    params["_damping_ratio"] = rlc["zeta"]
    params["_regime"] = rlc["regime"]
    params["_effective_pulse_duration_us"] = rlc["effective_pulse_duration_s"] * 1e6
    params["_stored_energy_J"] = rlc["stored_energy_J"]
    params["_num_layers"] = geom["num_layers"]
    params["_turns_per_layer"] = geom["turns_per_layer"]
    params["_actual_outer_radius_mm"] = geom["actual_outer_radius_mm"]
    params["_skin_depth_mm"] = ac_info["skin_depth_mm"]
    params["_ac_resistance_factor"] = ac_info["total_ac_factor"]
    params["_wire_length_mm"] = geom["wire_length_mm"]
    params["_wire_cross_section_mm2"] = geom["wire_cross_section_mm2"]
    params["_wire_mass_g"] = geom["wire_mass_g"]
    params["_discharge_frequency_Hz"] = freq_Hz

    return params


def _wire_gauge(diameter_mm: float) -> str:
    """Approximate AWG from wire diameter."""
    if diameter_mm <= 0:
        return "?"
    awg = 36 - 39 * math.log(diameter_mm / 0.127) / math.log(92)
    return f"{round(awg)}"


def generate_hollow_cylinder_mesh(inner_r, outer_r, length, num_seg=NUM_SEGMENTS):
    """Generate a hollow cylinder (coil shell) mesh.

    4 vertex rings: inner-bottom, inner-top, outer-bottom, outer-top
    Faces: outer wall, inner wall, bottom cap, top cap
    """
    angles = np.linspace(0, 2 * math.pi, num_seg, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    half_l = length / 2

    inner_bottom = np.column_stack([np.full(num_seg, -half_l), inner_r * cos_a, inner_r * sin_a])
    inner_top = np.column_stack([np.full(num_seg, half_l), inner_r * cos_a, inner_r * sin_a])
    outer_bottom = np.column_stack([np.full(num_seg, -half_l), outer_r * cos_a, outer_r * sin_a])
    outer_top = np.column_stack([np.full(num_seg, half_l), outer_r * cos_a, outer_r * sin_a])

    vertices = np.vstack([inner_bottom, inner_top, outer_bottom, outer_top])
    ib, it, ob, ot = 0, num_seg, 2 * num_seg, 3 * num_seg

    faces = []

    for i in range(num_seg):
        j = (i + 1) % num_seg

        # Outer wall
        faces.append([ob + i, ob + j, ot + j])
        faces.append([ob + i, ot + j, ot + i])

        # Inner wall (reversed winding)
        faces.append([ib + i, it + i, it + j])
        faces.append([ib + i, it + j, ib + j])

        # Bottom cap
        faces.append([ib + i, ob + i, ob + j])
        faces.append([ib + i, ob + j, ib + j])

        # Top cap
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
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
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
        "coilLauncher:resistance": ("resistance_ohm", Sdf.ValueTypeNames.Float),
        "coilLauncher:inductance": ("inductance_uH", Sdf.ValueTypeNames.Float),
        "coilLauncher:capacitance": ("capacitance_uF", Sdf.ValueTypeNames.Float),
        "coilLauncher:chargeVoltage": ("charge_voltage_V", Sdf.ValueTypeNames.Float),
        "coilLauncher:esr": ("esr_ohm", Sdf.ValueTypeNames.Float),
        "coilLauncher:wiringResistance": ("wiring_resistance_ohm", Sdf.ValueTypeNames.Float),
        "coilLauncher:marbleSaturation": ("marble_saturation_T", Sdf.ValueTypeNames.Float),
        "coilLauncher:marbleConductivity": ("marble_conductivity_S_per_m", Sdf.ValueTypeNames.Float),
    }

    for attr_name, (key, type_name) in attr_map.items():
        val = params.get(key)
        if val is None:
            continue
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
    params = derive_electrical_params(params)

    # Save derived values back to config (exclude internal _ fields)
    save_params = {k: v for k, v in params.items() if not k.startswith("_")}
    CONFIG_PATH.write_text(json.dumps(save_params, indent=2) + "\n")
    print(f"  Updated: {CONFIG_PATH}")

    create_coil_geometry(params)
    create_coil_properties(params)


if __name__ == "__main__":
    main()
