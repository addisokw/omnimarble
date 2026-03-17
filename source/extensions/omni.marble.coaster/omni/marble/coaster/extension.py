"""Marble Coaster Extension — loads scene, injects EM forces, controls PhysX simulation."""

import math
from pathlib import Path

import carb
import carb.settings
import omni.ext
import omni.kit.commands
import omni.physx
import omni.timeline
import omni.ui as ui
import omni.usd
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

# Path to the omnimarble project's USD files
# Resolves relative: extension.py -> coaster -> marble -> omni -> omni.marble.coaster -> extensions -> source -> PROJECT
_EXT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXT_DIR.parent.parent.parent.parent.parent.parent
USD_DIR = _PROJECT_ROOT / "usd"
SCENE_PATH = USD_DIR / "marble_coaster_scene.usda"
if not SCENE_PATH.exists():
    # Fallback for kit-app-template deployment where USD lives in the omnimarble project
    for _candidate in [Path(r"C:\Users\kaddi\Documents\Projects\omnimarble")]:
        if (_candidate / "usd" / "marble_coaster_scene.usda").exists():
            USD_DIR = _candidate / "usd"
            SCENE_PATH = USD_DIR / "marble_coaster_scene.usda"
            break

MU_0_MM = 4 * math.pi * 1e-4  # T*mm/A


# ---- Pure-Python elliptic integrals (no scipy dependency) ----

def _agm_ellipk(m):
    """Complete elliptic integral K(m) via arithmetic-geometric mean."""
    if m >= 1.0:
        return float('inf')
    if m < 0:
        m = 0.0
    a, b = 1.0, math.sqrt(1.0 - m)
    for _ in range(50):
        a, b = (a + b) / 2.0, math.sqrt(a * b)
        if abs(a - b) < 1e-15:
            break
    return math.pi / (2.0 * a)


def _agm_ellipe(m):
    """Complete elliptic integral E(m) via the AGM method."""
    if m >= 1.0:
        return 1.0
    if m < 0:
        m = 0.0
    a, b = 1.0, math.sqrt(1.0 - m)
    c = math.sqrt(m)
    s = m
    power_of_two = 1.0
    for _ in range(50):
        a_new = (a + b) / 2.0
        b_new = math.sqrt(a * b)
        c = (a - b) / 2.0
        power_of_two *= 2.0
        s += power_of_two * c * c
        a, b = a_new, b_new
        if abs(c) < 1e-15:
            break
    return (math.pi / (2.0 * a)) * (1.0 - s / 2.0)


class CoilParams:
    """Coil parameters read from extension settings."""

    def __init__(self):
        settings = carb.settings.get_settings()
        base = "/exts/omni.marble.coaster"
        self.inner_radius = settings.get_as_float(f"{base}/coil/innerRadius") or 8.0
        self.outer_radius = settings.get_as_float(f"{base}/coil/outerRadius") or 14.0
        self.length = settings.get_as_float(f"{base}/coil/length") or 30.0
        self.num_turns = int(settings.get_as_float(f"{base}/coil/numTurns") or 30)
        self.max_current = settings.get_as_float(f"{base}/coil/maxCurrent") or 10.0
        self.pulse_width_s = (settings.get_as_float(f"{base}/coil/pulseWidth") or 5.0) * 1e-3
        self.resistance = settings.get_as_float(f"{base}/coil/resistance") or 1.2
        self.inductance_H = (settings.get_as_float(f"{base}/coil/inductance") or 150.0) * 1e-6
        self.supply_voltage = settings.get_as_float(f"{base}/coil/supplyVoltage") or 24.0
        self.marble_radius = settings.get_as_float(f"{base}/marble/radius") or 5.0
        self.chi_eff = settings.get_as_float(f"{base}/marble/chiEff") or 100.0

        self.tau = self.inductance_H / self.resistance
        self.I_max = self.supply_voltage / self.resistance
        self.R_mean = (self.inner_radius + self.outer_radius) / 2
        self.V_marble = (4 / 3) * math.pi * self.marble_radius ** 3


class EMForceComputer:
    """Computes electromagnetic force on the marble using analytical Biot-Savart model."""

    def __init__(self, params: CoilParams):
        self.params = params

    def single_loop_field(self, r, z, R, I):
        """B-field from a single current loop using elliptic integrals (pure Python)."""
        r = abs(r)
        if r < 1e-10:
            denom = R * R + z * z
            if denom < 1e-20:
                return 0.0, 0.0
            Bz = MU_0_MM * I * R ** 2 / (2 * denom ** 1.5)
            return 0.0, Bz

        alpha_sq = R ** 2 + r ** 2 + z ** 2 - 2 * R * r
        beta_sq = R ** 2 + r ** 2 + z ** 2 + 2 * R * r
        if beta_sq < 1e-20:
            return 0.0, 0.0

        m = max(0.0, min(1.0 - 1e-12, 1.0 - alpha_sq / beta_sq))
        K = _agm_ellipk(m)
        E = _agm_ellipe(m)
        beta = math.sqrt(beta_sq)
        pf = MU_0_MM * I / (2 * math.pi)

        Bz = pf / beta * (K + (R ** 2 - r ** 2 - z ** 2) / alpha_sq * E)
        Br = pf * z / (r * beta) * (-K + (R ** 2 + r ** 2 + z ** 2) / alpha_sq * E)
        return float(Br), float(Bz)

    def solenoid_field(self, r, z, current):
        """Sum loop contributions for the full solenoid."""
        p = self.params
        Br_total, Bz_total = 0.0, 0.0
        for i in range(p.num_turns):
            z_loop = -p.length / 2 + (p.length * i / max(p.num_turns - 1, 1))
            br, bz = self.single_loop_field(r, z - z_loop, p.R_mean, current)
            Br_total += br
            Bz_total += bz
        return Br_total, Bz_total

    def compute_force(self, r, z, current):
        """Compute (F_r, F_z) in mN on the marble."""
        p = self.params
        dr, dz = 0.1, 0.1

        Br, Bz = self.solenoid_field(r, z, current)

        Br_rp, _ = self.solenoid_field(r + dr, z, current)
        Br_rm, _ = self.solenoid_field(max(r - dr, 0), z, current)
        Br_zp, _ = self.solenoid_field(r, z + dz, current)
        Br_zm, _ = self.solenoid_field(r, z - dz, current)
        _, Bz_rp = self.solenoid_field(r + dr, z, current)
        _, Bz_rm = self.solenoid_field(max(r - dr, 0), z, current)
        _, Bz_zp = self.solenoid_field(r, z + dz, current)
        _, Bz_zm = self.solenoid_field(r, z - dz, current)

        dBr_dr = (Br_rp - Br_rm) / (2 * dr) if r > dr else (Br_rp - Br_rm) / (r + dr)
        dBr_dz = (Br_zp - Br_zm) / (2 * dz)
        dBz_dr = (Bz_rp - Bz_rm) / (2 * dr) if r > dr else (Bz_rp - Bz_rm) / (r + dr)
        dBz_dz = (Bz_zp - Bz_zm) / (2 * dz)

        prefactor = p.chi_eff * p.V_marble / MU_0_MM
        F_r = prefactor * (Br * dBr_dr + Bz * dBr_dz)
        F_z = prefactor * (Br * dBz_dr + Bz * dBz_dz)
        return float(F_r), float(F_z)


class MarbleCoasterExtension(omni.ext.IExt):
    """Main extension class — manages scene, UI, and physics stepping."""

    def on_startup(self, ext_id):
        carb.log_info("[omni.marble.coaster] Starting up")

        self._params = CoilParams()
        self._em = EMForceComputer(self._params)

        # Simulation state
        self._triggered = False
        self._trigger_time = 0.0
        self._sim_time = 0.0
        self._coil_position = [0, 30, 0]  # Default, updated from USD
        self._coil_axis = [0, 1, 0]

        # PhysX callback
        self._physx_sub = None

        # Build UI
        self._build_ui()

    def on_shutdown(self):
        carb.log_info("[omni.marble.coaster] Shutting down")
        self._unsubscribe_physics()
        if self._window:
            self._window.destroy()
            self._window = None

    def _build_ui(self):
        """Build the control panel UI."""
        self._window = ui.Window("Marble Coaster Control", width=350, height=500)

        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Scene", style={"font_size": 18})
                ui.Button("Load Scene", clicked_fn=self._load_scene, height=30)
                ui.Button("Configure PhysX", clicked_fn=self._configure_physx, height=30)

                ui.Spacer(height=10)
                ui.Label("Coil Parameters", style={"font_size": 18})

                with ui.HStack(spacing=4):
                    ui.Label("Supply Voltage (V):", width=150)
                    self._voltage_field = ui.FloatField(width=100)
                    self._voltage_field.model.set_value(self._params.supply_voltage)

                with ui.HStack(spacing=4):
                    ui.Label("Pulse Width (ms):", width=150)
                    self._pulse_field = ui.FloatField(width=100)
                    self._pulse_field.model.set_value(self._params.pulse_width_s * 1e3)

                with ui.HStack(spacing=4):
                    ui.Label("Num Turns:", width=150)
                    self._turns_field = ui.IntField(width=100)
                    self._turns_field.model.set_value(self._params.num_turns)

                with ui.HStack(spacing=4):
                    ui.Label("Max Current (A):", width=150)
                    self._current_field = ui.FloatField(width=100)
                    self._current_field.model.set_value(self._params.max_current)

                ui.Button("Update Parameters", clicked_fn=self._update_params, height=30)

                ui.Spacer(height=10)
                ui.Label("Simulation", style={"font_size": 18})
                ui.Button("Start Simulation", clicked_fn=self._start_simulation, height=30)
                ui.Button("Stop Simulation", clicked_fn=self._stop_simulation, height=30)
                ui.Button("Reset Marble", clicked_fn=self._reset_marble, height=30)

                ui.Spacer(height=10)
                ui.Label("Status", style={"font_size": 18})
                self._status_label = ui.Label("Ready", word_wrap=True)

    def _load_scene(self):
        """Load the marble coaster USD scene."""
        scene_path = str(SCENE_PATH).replace("\\", "/")
        if not SCENE_PATH.exists():
            self._status_label.text = f"Scene not found: {scene_path}"
            carb.log_warn(f"Scene not found: {scene_path}")
            return

        omni.usd.get_context().open_stage(scene_path)
        self._status_label.text = f"Loaded: {SCENE_PATH.name}"
        carb.log_info(f"Loaded scene: {scene_path}")

        # Read coil position from USD
        self._read_coil_from_usd()

    def _read_coil_from_usd(self):
        """Read coil position and properties from the loaded USD stage."""
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        coil_prim = stage.GetPrimAtPath("/World/Coil")
        if coil_prim:
            xform = UsdGeom.Xformable(coil_prim)
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    pos = op.Get()
                    self._coil_position = [pos[0], pos[1], pos[2]]

            # Read custom attributes
            attr = coil_prim.GetAttribute("coilLauncher:innerRadius")
            if attr and attr.Get() is not None:
                self._params.inner_radius = attr.Get()
            attr = coil_prim.GetAttribute("coilLauncher:outerRadius")
            if attr and attr.Get() is not None:
                self._params.outer_radius = attr.Get()
            self._params.R_mean = (self._params.inner_radius + self._params.outer_radius) / 2

            carb.log_info(f"Coil position: {self._coil_position}")

    def _configure_physx(self):
        """Apply PhysxSchema settings that aren't available in usd-core."""
        stage = omni.usd.get_context().get_stage()
        if not stage:
            self._status_label.text = "No stage loaded"
            return

        # Physics scene settings
        physics_scene = stage.GetPrimAtPath("/World/PhysicsScene")
        if physics_scene:
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene)
            physx_scene.CreateEnableCCDAttr(True)
            physx_scene.CreateTimeStepsPerSecondAttr(500)  # 2ms timestep
            physx_scene.CreateEnableStabilizationAttr(True)

        # Marble rigid body settings
        marble_prim = stage.GetPrimAtPath("/World/Marble")
        if marble_prim:
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(marble_prim)
            physx_rb.CreateEnableCCDAttr(True)
            physx_rb.CreateLinearDampingAttr(0.01)
            physx_rb.CreateAngularDampingAttr(0.05)

        # Collision contact settings on the marble geometry
        marble_geom = stage.GetPrimAtPath("/World/Marble/Geom")
        if marble_geom:
            try:
                physx_col = PhysxSchema.PhysxCollisionAPI.Apply(marble_geom)
                physx_col.CreateContactOffsetAttr(0.5)
                physx_col.CreateRestOffsetAttr(0.1)
            except Exception as e:
                carb.log_warn(f"Could not set contact offsets: {e}")

        self._status_label.text = "PhysX configured: CCD on, 500 Hz, damping applied"
        carb.log_info("PhysX settings configured")

    def _update_params(self):
        """Read UI fields and update coil parameters."""
        self._params.supply_voltage = self._voltage_field.model.get_value_as_float()
        self._params.pulse_width_s = self._pulse_field.model.get_value_as_float() * 1e-3
        self._params.num_turns = self._turns_field.model.get_value_as_int()
        self._params.max_current = self._current_field.model.get_value_as_float()
        self._params.I_max = self._params.supply_voltage / self._params.resistance
        self._params.tau = self._params.inductance_H / self._params.resistance

        self._em = EMForceComputer(self._params)
        self._status_label.text = (
            f"Updated: V={self._params.supply_voltage}V, "
            f"pw={self._params.pulse_width_s*1e3:.1f}ms, "
            f"N={self._params.num_turns}, I_max={self._params.I_max:.1f}A"
        )

    def _start_simulation(self):
        """Subscribe to PhysX step callback and start timeline."""
        self._triggered = False
        self._trigger_time = 0.0
        self._sim_time = 0.0
        self._pulse_cut = False
        self._log_counter = 0
        self._subscribe_physics()

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        self._status_label.text = "Simulation running..."

    def _stop_simulation(self):
        """Stop simulation."""
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        self._unsubscribe_physics()
        self._status_label.text = "Simulation stopped"

    def _reset_marble(self):
        """Reset marble to initial position."""
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        marble_prim = stage.GetPrimAtPath("/World/Marble")
        if marble_prim:
            xform = UsdGeom.Xformable(marble_prim)
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(op.Get(0))  # Reset to time=0 value
                    break

        self._triggered = False
        self._trigger_time = 0.0
        self._sim_time = 0.0
        self._pulse_cut = False
        self._log_counter = 0
        self._status_label.text = "Marble reset"

    def _subscribe_physics(self):
        """Subscribe to the PhysX simulation step."""
        if self._physx_sub is not None:
            return
        physx = omni.physx.get_physx_interface()
        self._physx_sub = physx.subscribe_physics_step_events(self._on_physics_step)

    def _unsubscribe_physics(self):
        """Unsubscribe from physics steps."""
        self._physx_sub = None

    def _on_physics_step(self, dt):
        """Called each PhysX timestep — compute and apply EM force."""
        self._sim_time += dt
        self._log_counter = getattr(self, '_log_counter', 0) + 1

        stage = omni.usd.get_context().get_stage()
        if not stage:
            if self._log_counter % 100 == 1:
                carb.log_error("[EM] No stage")
            return

        marble_prim = stage.GetPrimAtPath("/World/Marble")
        if not marble_prim:
            if self._log_counter % 100 == 1:
                carb.log_error("[EM] No marble prim")
            return

        # --- DEBUG: Read marble state from PhysX ---
        physx_iface = omni.physx.get_physx_interface()
        rb_data = physx_iface.get_rigidbody_transformation("/World/Marble")

        # Also read from USD
        xform = UsdGeom.Xformable(marble_prim)
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos = world_transform.ExtractTranslation()
        marble_pos = [pos[0], pos[1], pos[2]]

        rb_api = UsdPhysics.RigidBodyAPI(marble_prim)
        vel_attr = rb_api.GetVelocityAttr()
        cur_vel = vel_attr.Get() if vel_attr else None

        if self._log_counter % 25 == 1:
            physx_pos = rb_data.get('position', None) if rb_data.get('ret_val', False) else None
            carb.log_warn(
                f"[DBG] t={self._sim_time:.3f}s dt={dt:.4f}s "
                f"USD_pos=({marble_pos[0]:.1f},{marble_pos[1]:.1f},{marble_pos[2]:.1f}) "
                f"PhysX_pos={physx_pos} "
                f"USD_vel={cur_vel} "
                f"rb_valid={rb_data.get('ret_val', False)}"
            )

        # Coil-local coordinates
        coil_pos = self._coil_position
        coil_axis = self._coil_axis
        ax_len = math.sqrt(sum(a * a for a in coil_axis))
        coil_axis_n = [a / ax_len for a in coil_axis]

        relative = [marble_pos[i] - coil_pos[i] for i in range(3)]
        z_along = sum(relative[i] * coil_axis_n[i] for i in range(3))
        radial_vec = [relative[i] - z_along * coil_axis_n[i] for i in range(3)]
        r = math.sqrt(sum(v * v for v in radial_vec))

        # Trigger when marble is approaching the coil (negative z_along = before center)
        if not self._triggered and z_along < 0:
            self._triggered = True
            self._trigger_time = self._sim_time
            carb.log_warn(f"[EM] Coil triggered at t={self._sim_time:.4f}s, z_along={z_along:.1f}mm")

        if not self._triggered:
            return

        # Cut pulse when marble passes coil center
        if z_along > 0 and not getattr(self, '_pulse_cut', False):
            self._pulse_cut = True
            self._pulse_cut_time = self._sim_time
            carb.log_warn(f"[EM] Pulse cut at t={self._sim_time:.4f}s, marble past center")

        # LR circuit current — stays ON until marble passes coil center (pulse_cut)
        p = self._params
        dt_trigger = self._sim_time - self._trigger_time

        if getattr(self, '_pulse_cut', False):
            dt_cut = self._sim_time - self._pulse_cut_time
            current = p.max_current * math.exp(-dt_cut / p.tau)
        elif dt_trigger < 1.0:  # Keep current on until pulse_cut (1s safety max)
            current = p.max_current * (1 - math.exp(-dt_trigger / p.tau))
        else:
            current = 0.0

        if abs(current) < 1e-6:
            return

        # Compute EM force
        _, Bz = self._em.solenoid_field(r, z_along, current)
        _, Bz_p = self._em.solenoid_field(r, z_along + 0.1, current)
        _, Bz_m = self._em.solenoid_field(r, z_along - 0.1, current)
        dBz_dz = (Bz_p - Bz_m) / 0.2

        prefactor = p.chi_eff * p.V_marble / MU_0_MM
        F_z_mN = prefactor * Bz * dBz_dz  # mN
        force_N = F_z_mN * 1e-3
        force_world = [force_N * coil_axis_n[i] for i in range(3)]

        # Compute expected acceleration for logging
        marble_mass_kg = p.V_marble * 7.8e-3 * 1e-3  # mm^3 * g/mm^3 * kg/g = kg
        accel_ms2 = force_N / marble_mass_kg
        dv_mm_s = accel_ms2 * 1000.0 * dt

        # Log EVERY step during active pulse, then every 25th after
        if not getattr(self, '_pulse_cut', False) or self._log_counter % 25 == 1:
            carb.log_warn(
                f"[EM] step={self._log_counter} t={self._sim_time:.4f}s I={current:.2f}A "
                f"Bz={Bz*1e3:.3f}mT F={F_z_mN:.2f}mN "
                f"accel={accel_ms2:.1f}m/s2 dv={dv_mm_s:.2f}mm/s "
                f"z={z_along:.1f}mm r={r:.1f}mm"
            )

        # --- Try ALL methods of applying force ---

        # NOTE: apply_force_at_pos was confirmed non-functional in this Kit version.
        # Using direct velocity modification below instead.

        # apply_force_at_pos doesn't actually move the marble (confirmed by debug).
        # Use direct velocity modification instead.
        if vel_attr and cur_vel is not None:
            new_vel = Gf.Vec3f(
                cur_vel[0] + dv_mm_s * coil_axis_n[0],
                cur_vel[1] + dv_mm_s * coil_axis_n[1],
                cur_vel[2] + dv_mm_s * coil_axis_n[2],
            )
            vel_attr.Set(new_vel)
            speed = math.sqrt(new_vel[0]**2 + new_vel[1]**2 + new_vel[2]**2)
            if not getattr(self, '_pulse_cut', False) or self._log_counter % 25 == 1:
                carb.log_warn(f"[VEL] step={self._log_counter} "
                              f"cur=({cur_vel[0]:.1f},{cur_vel[1]:.1f},{cur_vel[2]:.1f}) "
                              f"dv={dv_mm_s:.2f} "
                              f"new=({new_vel[0]:.1f},{new_vel[1]:.1f},{new_vel[2]:.1f}) "
                              f"speed={speed:.1f}mm/s")
