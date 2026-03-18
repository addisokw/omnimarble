"""Marble Coaster Extension — RLC capacitor discharge coilgun physics.

PhysX-native force application via apply_force_at_pos + PINN B-field inference.

torch and physicsnemo are installed into Kit's Python at startup via
omni.kit.pipapi.  They are imported lazily in on_startup(), not at module level.
"""

import json
import math
from pathlib import Path

import carb
import carb._carb
import carb.settings
import omni.ext
import omni.kit.commands
import omni.kit.pipapi
import omni.physx
import omni.timeline
import omni.ui as ui
import omni.usd
from omni.physx.scripts.utils import get_physx_simulation_interface
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

# Late-bound after pipapi install in on_startup()
torch = None
NeMoFC = None

OMNIMARBLE_PROJECT = Path(r"C:\Users\kaddi\Documents\Projects\omnimarble")
USD_DIR = OMNIMARBLE_PROJECT / "usd"
SCENE_PATH = USD_DIR / "marble_coaster_scene.usda"
PINN_CHECKPOINT = OMNIMARBLE_PROJECT / "models" / "pinn_checkpoint" / "pinn_best.pt"

MU_0_MM = 4 * math.pi * 1e-4  # T*mm/A
COPPER_TEMP_COEFF = 0.00393  # /C
COPPER_SPECIFIC_HEAT = 0.385  # J/(g*C)
COPPER_RESISTIVITY_OHM_MM = 1.72e-5  # ohm*mm
COPPER_DENSITY_G_MM3 = 8.96e-3


class CoilParams:
    """Coil parameters read from config/coil_params.json, overridable via extension settings."""

    def __init__(self):
        settings = carb.settings.get_settings()
        base = "/exts/omni.marble.coaster"

        # Load JSON config as primary source of truth
        config_path = OMNIMARBLE_PROJECT / "config" / "coil_params.json"
        cfg = {}
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text())
            except Exception:
                carb.log_warn(f"[CoilParams] Failed to read {config_path}, using extension defaults")

        def _get(setting_path, json_key, default):
            """Read from JSON config first, fall back to carb settings, then default.

            JSON is the source of truth. Carb settings only matter for fields
            not present in JSON (or when overridden at runtime via UI).
            """
            if json_key in cfg:
                return cfg[json_key]
            val = settings.get_as_float(f"{base}/{setting_path}")
            if val is not None:
                return val
            return default

        # Geometry
        self.inner_radius = _get("coil/innerRadius", "inner_radius_mm", 12.0)
        self.outer_radius = _get("coil/outerRadius", "outer_radius_mm", 18.0)
        self.length = _get("coil/length", "length_mm", 30.0)
        self.num_turns = int(_get("coil/numTurns", "num_turns", 30))
        self.wire_diameter = _get("coil/wireDiameter", "wire_diameter_mm", 0.8)
        self.insulation_thickness = _get("coil/insulationThickness", "insulation_thickness_mm", 0.035)

        # RLC circuit
        self.capacitance_uF = _get("coil/capacitance", "capacitance_uF", 470.0)
        self.charge_voltage = _get("coil/chargeVoltage", "charge_voltage_V", 50.0)
        self.esr = _get("coil/esr", "esr_ohm", 0.01)
        self.wiring_resistance = _get("coil/wiringResistance", "wiring_resistance_ohm", 0.02)
        self.switch_type = settings.get_as_string(f"{base}/coil/switchType") or cfg.get("switch_type", "MOSFET")
        self.has_flyback_diode = settings.get_as_bool(f"{base}/coil/hasFlybackDiode")
        if self.has_flyback_diode is None:
            self.has_flyback_diode = cfg.get("has_flyback_diode", True)

        # Marble
        self.marble_radius = _get("marble/radius", "marble_radius_mm", 5.0)
        self.chi_eff = _get("marble/chiEff", "marble_chi_eff", 3.0)
        self.B_sat = _get("marble/saturationT", "marble_saturation_T", 1.8)
        self.conductivity = _get("marble/conductivity", "marble_conductivity_S_per_m", 6e6)

        # Environment
        self.ambient_temp = _get("environment/ambientTemperature", "ambient_temperature_C", 20.0)

        # IR break-beam gate positions along coil axis, relative to coil center (mm).
        json_gates = cfg.get("gate_positions", {})

        self.gates = {
            "vel_in_1":  json_gates.get("vel_in_1",  settings.get_as_float(f"{base}/gates/velIn1") or -60.0),
            "vel_in_2":  json_gates.get("vel_in_2",  settings.get_as_float(f"{base}/gates/velIn2") or -40.0),
            "entry":     json_gates.get("entry",      settings.get_as_float(f"{base}/gates/entry") or -20.0),
            "cutoff":    json_gates.get("cutoff",     settings.get_as_float(f"{base}/gates/cutoff") or 5.0),
            "vel_out_1": json_gates.get("vel_out_1",  settings.get_as_float(f"{base}/gates/velOut1") or 60.0),
            "vel_out_2": json_gates.get("vel_out_2",  settings.get_as_float(f"{base}/gates/velOut2") or 120.0),
        }
        carb.log_info(f"[CoilParams] Gates: {self.gates}")
        self.sensor_entry_offset = self.gates["entry"]
        self.sensor_cutoff_offset = self.gates["cutoff"]

        # Derived geometry
        self.R_mean = (self.inner_radius + self.outer_radius) / 2
        self.V_marble = (4 / 3) * math.pi * self.marble_radius ** 3
        self.marble_mass_kg = self.V_marble * 7.8e-3 * 1e-3

        # Winding geometry
        wire_pitch = self.wire_diameter + 2 * self.insulation_thickness
        self.turns_per_layer = max(1, int(self.length / wire_pitch))
        self.num_layers = math.ceil(self.num_turns / self.turns_per_layer)
        winding_depth = self.num_layers * wire_pitch
        self.mean_radius = self.inner_radius + winding_depth / 2

        # Wire length and resistance
        wire_length = 0.0
        turns_remaining = self.num_turns
        for layer in range(self.num_layers):
            r_layer = self.inner_radius + (layer + 0.5) * wire_pitch
            n_this = min(self.turns_per_layer, turns_remaining)
            wire_length += n_this * 2 * math.pi * r_layer
            turns_remaining -= n_this
        self.wire_length_mm = wire_length
        wire_cross = math.pi * (self.wire_diameter / 2) ** 2
        self.wire_mass_g = wire_cross * wire_length * COPPER_DENSITY_G_MM3
        self.R_dc = COPPER_RESISTIVITY_OHM_MM * wire_length / wire_cross

        # Inductance (Wheeler's multilayer)
        a_in = self.mean_radius / 25.4
        l_in = self.length / 25.4
        c_in = winding_depth / 25.4
        denom = 6 * a_in + 9 * l_in + 10 * c_in
        self.inductance_uH = 0.8 * a_in ** 2 * self.num_turns ** 2 / denom if denom > 0 else 0
        self.inductance_H = self.inductance_uH * 1e-6

        # Total resistance
        self.R_total = self.R_dc + self.esr + self.wiring_resistance

        # RLC parameters
        C = self.capacitance_uF * 1e-6
        L = self.inductance_H
        self.alpha = self.R_total / (2 * L) if L > 0 else 0
        self.omega_0 = 1.0 / math.sqrt(L * C) if L > 0 and C > 0 else 0
        self.zeta = self.alpha / self.omega_0 if self.omega_0 > 0 else 999

        if self.zeta < 1.0:
            self.omega_d = math.sqrt(self.omega_0 ** 2 - self.alpha ** 2)
            self.regime = "underdamped"
            self.peak_current = (self.charge_voltage / (self.omega_d * L)) if L > 0 else 0
            self.t_peak = math.atan2(self.omega_d, self.alpha) / self.omega_d
            self.t_zero_crossing = math.pi / self.omega_d
        elif abs(self.zeta - 1.0) < 0.01:
            self.omega_d = 0
            self.regime = "critically_damped"
            self.peak_current = (self.charge_voltage / L) / (self.alpha * math.e) if L > 0 else 0
            self.t_peak = 1.0 / self.alpha if self.alpha > 0 else 0
            self.t_zero_crossing = 5.0 / self.alpha if self.alpha > 0 else 0
        else:
            self.omega_d = 0
            self.regime = "overdamped"
            self.peak_current = 0
            self.t_peak = 0
            self.t_zero_crossing = 5.0 / self.alpha if self.alpha > 0 else 0

        self.stored_energy = 0.5 * C * self.charge_voltage ** 2

        carb.log_info(f"[CoilParams] RLC: {self.regime}, zeta={self.zeta:.4f}, "
                      f"I_peak={self.peak_current:.1f}A, E={self.stored_energy:.1f}J")

    def log_specs(self):
        """Log full coil specs to console."""
        carb.log_warn(
            f"[COIL SPECS] "
            f"Turns={self.num_turns} | "
            f"R_inner={self.inner_radius}mm R_outer={self.outer_radius}mm L_coil={self.length}mm | "
            f"Wire={self.wire_diameter}mm ({self.num_layers}layer) | "
            f"Cap={self.capacitance_uF}uF @ {self.charge_voltage}V | "
            f"E={self.stored_energy:.2f}J | "
            f"R_dc={self.R_dc:.4f}ohm R_total={self.R_total:.4f}ohm | "
            f"L_ind={self.inductance_uH:.2f}uH | "
            f"RLC: {self.regime} zeta={self.zeta:.4f} | "
            f"I_peak={self.peak_current:.1f}A | "
            f"chi_eff={self.chi_eff} B_sat={self.B_sat}T | "
            f"Gates: {self.gates}"
        )


class PINNForceComputer:
    """Computes EM force using PINN B-field inference with saturation and eddy currents."""

    def __init__(self, params: CoilParams, pinn_model, device):
        self.params = params
        self.model = pinn_model
        self.device = device
        # Check if model was trained with B/I normalization
        self._current_normalized = bool(
            getattr(pinn_model, 'current_normalized', torch.tensor(False)).item()
        )

    def _pinn_field(self, r: float, z: float, current: float) -> tuple:
        """Single PINN forward pass returning (B_r, B_z).

        Input features: (r, z, I, N, R_mean, L)
        Output: (A_phi, B_r, B_z)
        """
        p = self.params
        inp = torch.tensor(
            [[r, z, current, float(p.num_turns), p.R_mean, p.length]],
            dtype=torch.float32, device=self.device,
        )
        with torch.no_grad():
            out = self.model(inp)
        # If model outputs B/I, multiply by I to get actual B
        scale = current if self._current_normalized else 1.0
        return float(out[0, 1]) * scale, float(out[0, 2]) * scale

    def _pinn_field_with_grad(self, r: float, z: float, current: float) -> tuple:
        """PINN forward pass with autograd for dBz/dz and dBr/dr.

        Returns (B_r, B_z, dBr_dr, dBr_dz, dBz_dr, dBz_dz).
        """
        p = self.params
        inp = torch.tensor(
            [[r, z, current, float(p.num_turns), p.R_mean, p.length]],
            dtype=torch.float32, device=self.device, requires_grad=True,
        )
        out = self.model(inp)
        B_r = out[0, 1]
        B_z = out[0, 2]

        # Gradients of B_r w.r.t. input (r=0, z=1)
        grad_Br = torch.autograd.grad(
            B_r, inp, create_graph=False, retain_graph=True,
        )[0][0]
        grad_Bz = torch.autograd.grad(
            B_z, inp, create_graph=False, retain_graph=False,
        )[0][0]

        # If model outputs B/I, scale everything by I
        scale = current if self._current_normalized else 1.0
        return (
            B_r.detach().item() * scale, B_z.detach().item() * scale,
            grad_Br[0].item() * scale, grad_Br[1].item() * scale,  # dBr/dr, dBr/dz
            grad_Bz[0].item() * scale, grad_Bz[1].item() * scale,  # dBz/dr, dBz/dz
        )

    def compute_force(self, r: float, z: float, current: float,
                      dBdt: float = 0.0, vel_axial: float = 0.0) -> tuple:
        """Compute (F_r, F_z) in mN using PINN B-field with saturation and eddy currents."""
        p = self.params

        Br, Bz, dBr_dr, dBr_dz, dBz_dr, dBz_dz = self._pinn_field_with_grad(r, z, current)

        # Saturation check
        B_internal = (1 + p.chi_eff / 3) * abs(Bz)
        if B_internal < p.B_sat:
            prefactor = p.chi_eff * p.V_marble / MU_0_MM
            F_r = prefactor * (Br * dBr_dr + Bz * dBr_dz)
            F_z = prefactor * (Br * dBz_dr + Bz * dBz_dz)
        else:
            M_sat = p.B_sat / MU_0_MM
            F_z = M_sat * p.V_marble * dBz_dz * MU_0_MM
            prefactor = p.chi_eff * p.V_marble / MU_0_MM
            F_r = prefactor * (Br * dBr_dr + Bz * dBr_dz)

        # Eddy current braking
        if abs(dBdt) > 1e-10 and abs(vel_axial) > 1e-6:
            r_m = p.marble_radius * 1e-3
            V_m3 = p.V_marble * 1e-9
            F_eddy_N = -p.conductivity * V_m3 * r_m ** 2 * dBdt ** 2 / 20.0
            F_eddy_mN = F_eddy_N * 1000
            if vel_axial > 0:
                F_z += F_eddy_mN
            else:
                F_z -= F_eddy_mN

        return float(F_r), float(F_z)

    def get_Bz(self, r: float, z: float, current: float) -> float:
        """Return axial B-field component for logging."""
        _, Bz = self._pinn_field(r, z, current)
        return Bz


class MarbleCoasterExtension(omni.ext.IExt):
    """Main extension class — manages scene, UI, and physics stepping."""

    def on_startup(self, ext_id):
        carb.log_info("[omni.marble.coaster] Starting up (PhysX-native + PINN)")

        # Install torch + physicsnemo into Kit's Python via pipapi (cached after first run)
        self._install_ml_deps()

        self._params = CoilParams()

        # Load PINN model (required — fail loudly if missing)
        self._pinn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._pinn_model = self._load_pinn()
        self._em = PINNForceComputer(self._params, self._pinn_model, self._pinn_device)
        carb.log_warn(f"[PINN] Loaded on {self._pinn_device}, checkpoint: {PINN_CHECKPOINT}")

        # Simulation state
        self._triggered = False
        self._trigger_time = 0.0
        self._sim_time = 0.0
        self._coil_position = [0, 30, 0]
        self._coil_axis = [0, 1, 0]

        # RLC circuit state (coupled ODE)
        self._circuit_I = 0.0
        self._circuit_Q_cap = self._params.capacitance_uF * 1e-6 * self._params.charge_voltage
        self._wire_temp = self._params.ambient_temp
        self._prev_B = 0.0
        self._pulse_cut = False
        self._pulse_cut_time = 0.0

        # PhysX force API (initialized on simulation start when stage is available)
        self._physx_sim = None
        self._stage_id = None
        self._marble_prim_id = None

        # IR gate state
        self._init_gate_state()

        # PhysX callback
        self._physx_sub = None

        self._build_ui()

    @staticmethod
    def _install_ml_deps():
        """Install torch and physicsnemo into Kit Python via omni.kit.pipapi.

        Packages are cached after the first install — subsequent launches skip
        the download.  We bind the module-level ``torch`` and ``NeMoFC``
        sentinels so the rest of the extension can use them normally.
        """
        global torch, NeMoFC

        carb.log_warn("[PINN] Installing torch (first launch may take a few minutes)…")
        omni.kit.pipapi.install(
            "torch",
            extra_args=[
                "--extra-index-url", "https://download.pytorch.org/whl/cu128",
            ],
        )
        carb.log_warn("[PINN] Installing nvidia-physicsnemo…")
        omni.kit.pipapi.install("nvidia-physicsnemo", module="physicsnemo")

        import torch as _torch
        from physicsnemo.models.mlp import FullyConnected as _NeMoFC

        torch = _torch
        NeMoFC = _NeMoFC
        carb.log_warn(f"[PINN] torch {torch.__version__} (CUDA {torch.version.cuda}), physicsnemo ready")

    def _load_pinn(self) -> "torch.nn.Module":
        """Load the trained PINN checkpoint. Fails if checkpoint is missing."""
        model = NeMoFC(
            in_features=6,
            layer_size=256,
            out_features=3,
            num_layers=6,
            activation_fn="silu",
            skip_connections=True,
        )

        # Wrap with input normalization buffers matching BFieldPINN from train_pinn.py
        class PINNWrapper(torch.nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                self.register_buffer("input_mean", torch.zeros(6))
                self.register_buffer("input_std", torch.ones(6))
                self.register_buffer("output_scale", torch.ones(3))
                self.register_buffer("current_normalized", torch.tensor(False))

            def forward(self, x):
                x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
                return self.backbone(x_norm) * self.output_scale

        wrapper = PINNWrapper(model)
        checkpoint = torch.load(str(PINN_CHECKPOINT), map_location=self._pinn_device, weights_only=False)
        if "model_state_dict" in checkpoint:
            wrapper.load_state_dict(checkpoint["model_state_dict"])
        else:
            wrapper.load_state_dict(checkpoint)
        wrapper.to(self._pinn_device)
        wrapper.eval()
        return wrapper

    def _init_gate_state(self):
        """Initialize / reset all IR gate tracking state."""
        self._prev_z_along = None
        self._gate_times = {}
        self._gate_triggered = {}
        for name in self._params.gates:
            self._gate_times[name] = None
            self._gate_triggered[name] = False
        self._approach_velocity = None
        self._exit_velocity = None
        self._gate_exit_measured = False
        self._predicted_fire_time = None

    def on_shutdown(self):
        carb.log_info("[omni.marble.coaster] Shutting down")
        self._unsubscribe_physics()
        if self._window:
            self._window.destroy()
            self._window = None

    def _build_ui(self):
        """Build the control panel UI."""
        self._window = ui.Window("Marble Coaster Control", width=380, height=650)

        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Scene", style={"font_size": 18})
                ui.Button("Load Scene", clicked_fn=self._load_scene, height=30)
                ui.Button("Configure PhysX", clicked_fn=self._configure_physx, height=30)

                ui.Spacer(height=10)
                ui.Label("RLC Circuit", style={"font_size": 18})

                with ui.HStack(spacing=4):
                    ui.Label("Charge Voltage (V):", width=160)
                    self._voltage_field = ui.FloatField(width=100)
                    self._voltage_field.model.set_value(self._params.charge_voltage)

                with ui.HStack(spacing=4):
                    ui.Label("Capacitance (uF):", width=160)
                    self._cap_field = ui.FloatField(width=100)
                    self._cap_field.model.set_value(self._params.capacitance_uF)

                with ui.HStack(spacing=4):
                    ui.Label("Num Turns:", width=160)
                    self._turns_field = ui.IntField(width=100)
                    self._turns_field.model.set_value(self._params.num_turns)

                with ui.HStack(spacing=4):
                    ui.Label("Chi Effective:", width=160)
                    self._chi_field = ui.FloatField(width=100)
                    self._chi_field.model.set_value(self._params.chi_eff)

                ui.Button("Update Parameters", clicked_fn=self._update_params, height=30)

                ui.Spacer(height=10)
                ui.Label("Derived Values", style={"font_size": 18})
                self._derived_label = ui.Label(
                    f"Peak I: {self._params.peak_current:.0f}A | "
                    f"Regime: {self._params.regime} | "
                    f"zeta: {self._params.zeta:.3f}\n"
                    f"L: {self._params.inductance_uH:.1f}uH | "
                    f"R: {self._params.R_total:.4f}ohm | "
                    f"E: {self._params.stored_energy:.1f}J",
                    word_wrap=True,
                )

                ui.Spacer(height=10)
                ui.Label("Simulation", style={"font_size": 18})
                ui.Button("Start Simulation", clicked_fn=self._start_simulation, height=30)
                ui.Button("Stop Simulation", clicked_fn=self._stop_simulation, height=30)
                ui.Button("Reset Marble", clicked_fn=self._reset_marble, height=30)

                ui.Spacer(height=10)
                ui.Label("Status", style={"font_size": 18})
                self._status_label = ui.Label("Ready", word_wrap=True)
                self._realtime_label = ui.Label(
                    "I: -- | V_cap: -- | T_wire: -- | F: --",
                    word_wrap=True,
                )

    def _load_scene(self):
        """Load the marble coaster USD scene."""
        scene_path = str(SCENE_PATH).replace("\\", "/")
        if not SCENE_PATH.exists():
            self._status_label.text = f"Scene not found: {scene_path}"
            return

        omni.usd.get_context().open_stage(scene_path)
        self._status_label.text = f"Loaded: {SCENE_PATH.name}"
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

            for attr_name, field in [
                ("coilLauncher:innerRadius", "inner_radius"),
                ("coilLauncher:outerRadius", "outer_radius"),
                ("coilLauncher:capacitance", "capacitance_uF"),
                ("coilLauncher:chargeVoltage", "charge_voltage"),
            ]:
                attr = coil_prim.GetAttribute(attr_name)
                if attr and attr.Get() is not None:
                    setattr(self._params, field, attr.Get())

            self._params.R_mean = (self._params.inner_radius + self._params.outer_radius) / 2
            carb.log_info(f"Coil position: {self._coil_position}")

    def _configure_physx(self):
        """Apply PhysxSchema settings."""
        stage = omni.usd.get_context().get_stage()
        if not stage:
            self._status_label.text = "No stage loaded"
            return

        physics_scene = stage.GetPrimAtPath("/World/PhysicsScene")
        if physics_scene:
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene)
            physx_scene.CreateEnableCCDAttr(True)
            physx_scene.CreateTimeStepsPerSecondAttr(500)
            physx_scene.CreateEnableStabilizationAttr(True)

        marble_prim = stage.GetPrimAtPath("/World/Marble")
        if marble_prim:
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(marble_prim)
            physx_rb.CreateEnableCCDAttr(True)
            physx_rb.CreateLinearDampingAttr(0.01)
            physx_rb.CreateAngularDampingAttr(0.05)

        marble_geom = stage.GetPrimAtPath("/World/Marble/Geom")
        if marble_geom:
            try:
                physx_col = PhysxSchema.PhysxCollisionAPI.Apply(marble_geom)
                physx_col.CreateContactOffsetAttr(0.5)
                physx_col.CreateRestOffsetAttr(0.1)
            except Exception as e:
                carb.log_warn(f"Could not set contact offsets: {e}")

        self._status_label.text = "PhysX configured: CCD on, 500 Hz"

    def _update_params(self):
        """Read UI fields and recompute all RLC parameters from scratch.

        CoilParams.__init__ reads from JSON config as source of truth.
        UI fields override only the 4 user-adjustable values.
        """
        self._params = CoilParams()
        # Override with UI field values (user-adjustable subset)
        self._params.charge_voltage = self._voltage_field.model.get_value_as_float()
        self._params.capacitance_uF = self._cap_field.model.get_value_as_float()
        self._params.num_turns = int(self._turns_field.model.get_value_as_int())
        self._params.chi_eff = self._chi_field.model.get_value_as_float()

        # Recompute RLC with the UI-overridden values
        C = self._params.capacitance_uF * 1e-6
        L = self._params.inductance_H
        self._params.alpha = self._params.R_total / (2 * L) if L > 0 else 0
        self._params.omega_0 = 1.0 / math.sqrt(L * C) if L > 0 and C > 0 else 0
        self._params.zeta = self._params.alpha / self._params.omega_0 if self._params.omega_0 > 0 else 999
        if self._params.zeta < 1.0:
            self._params.omega_d = math.sqrt(self._params.omega_0 ** 2 - self._params.alpha ** 2)
            self._params.regime = "underdamped"
            self._params.peak_current = self._params.charge_voltage / (self._params.omega_d * L) if L > 0 else 0
        self._params.stored_energy = 0.5 * C * self._params.charge_voltage ** 2

        self._em = PINNForceComputer(self._params, self._pinn_model, self._pinn_device)

        self._derived_label.text = (
            f"Peak I: {self._params.peak_current:.0f}A | "
            f"Regime: {self._params.regime} | "
            f"zeta: {self._params.zeta:.3f}\n"
            f"L: {self._params.inductance_uH:.1f}uH | "
            f"R: {self._params.R_total:.4f}ohm | "
            f"E: {self._params.stored_energy:.1f}J"
        )
        self._status_label.text = (
            f"Updated: V={self._params.charge_voltage}V, "
            f"C={self._params.capacitance_uF}uF, "
            f"I_peak={self._params.peak_current:.0f}A"
        )

    def _start_simulation(self):
        """Subscribe to PhysX step callback and start timeline."""
        self._triggered = False
        self._trigger_time = 0.0
        self._sim_time = 0.0
        self._pulse_cut = False
        self._pulse_cut_time = 0.0
        self._log_counter = 0

        # Reset circuit state
        p = self._params
        self._circuit_I = 0.0
        self._circuit_Q_cap = p.capacitance_uF * 1e-6 * p.charge_voltage
        self._wire_temp = p.ambient_temp
        self._prev_B = 0.0

        # Cache PhysX integer IDs for apply_force_at_pos
        stage = omni.usd.get_context().get_stage()
        self._physx_sim = get_physx_simulation_interface()
        self._stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        self._marble_prim_id = PhysicsSchemaTools.sdfPathToInt("/World/Marble")

        self._subscribe_physics()
        self._params.log_specs()

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        self._status_label.text = "Simulation running (PhysX-native + PINN)..."

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
                    op.Set(op.Get(0))
                    break

        self._triggered = False
        self._trigger_time = 0.0
        self._sim_time = 0.0
        self._pulse_cut = False
        self._pulse_cut_time = 0.0
        self._circuit_I = 0.0
        p = self._params
        self._circuit_Q_cap = p.capacitance_uF * 1e-6 * p.charge_voltage
        self._circuit_I_rms = 0.0
        self._circuit_I_peak = 0.0
        self._wire_temp = p.ambient_temp
        self._prev_B = 0.0
        self._log_counter = 0
        self._init_gate_state()
        self._status_label.text = "Marble reset"
        self._realtime_label.text = "I: -- | V_cap: -- | T_wire: -- | F: --"

    def _subscribe_physics(self):
        if self._physx_sub is not None:
            return
        physx = omni.physx.get_physx_interface()
        self._physx_sub = physx.subscribe_physics_step_events(self._on_physics_step)

    def _unsubscribe_physics(self):
        self._physx_sub = None

    def _rlc_current(self, t_since_trigger):
        """Compute RLC discharge current (pure Python)."""
        p = self._params
        if t_since_trigger < 0:
            return 0.0

        if p.regime == "underdamped":
            I = (p.charge_voltage / (p.omega_d * p.inductance_H)) * \
                math.exp(-p.alpha * t_since_trigger) * \
                math.sin(p.omega_d * t_since_trigger)
        elif p.regime == "critically_damped":
            I = (p.charge_voltage / p.inductance_H) * \
                t_since_trigger * math.exp(-p.alpha * t_since_trigger)
        else:
            s1 = -p.alpha + math.sqrt(p.alpha ** 2 - p.omega_0 ** 2)
            s2 = -p.alpha - math.sqrt(p.alpha ** 2 - p.omega_0 ** 2)
            if abs(s1 - s2) > 1e-12:
                I = (p.charge_voltage / (p.inductance_H * (s1 - s2))) * \
                    (math.exp(s1 * t_since_trigger) - math.exp(s2 * t_since_trigger))
            else:
                I = 0.0

        return max(I, 0.0) if p.has_flyback_diode else I

    def _coupled_rlc_step(self, dt, z_along, vel_axial):
        """Step the coupled electromechanical RLC ODE with sub-stepping."""
        max_substep = 1e-5
        n_substeps = max(1, int(math.ceil(dt / max_substep)))
        sub_dt = dt / n_substeps

        I_sq_sum = 0.0
        I_peak = 0.0
        for _ in range(n_substeps):
            self._coupled_rlc_single_step(sub_dt, z_along, vel_axial)
            I_sq_sum += self._circuit_I ** 2
            I_peak = max(I_peak, abs(self._circuit_I))

        self._circuit_I_rms = math.sqrt(I_sq_sum / n_substeps)
        self._circuit_I_peak = I_peak

    def _coupled_rlc_single_step(self, dt, z_along, vel_axial):
        """Single RK4 step of the coupled RLC ODE."""
        p = self._params
        R = p.R_total
        C = p.capacitance_uF * 1e-6
        L = p.inductance_H

        # Dynamic inductance coupling
        coil_half = p.length / 2
        marble_front = z_along + p.marble_radius
        marble_back = z_along - p.marble_radius
        overlap_start = max(marble_back, -coil_half)
        overlap_end = min(marble_front, coil_half)
        overlap = max(0.0, overlap_end - overlap_start) / (2 * p.marble_radius)

        k = (p.marble_radius / p.inner_radius) ** 2 * (1 + p.chi_eff) * 0.01
        L_eff = L * (1.0 + k * overlap)

        # dL/dx (numerical)
        dx = 0.1
        overlap_p = max(0.0, min(z_along + dx + p.marble_radius, coil_half) -
                        max(z_along + dx - p.marble_radius, -coil_half)) / (2 * p.marble_radius)
        overlap_m = max(0.0, min(z_along - dx + p.marble_radius, coil_half) -
                        max(z_along - dx - p.marble_radius, -coil_half)) / (2 * p.marble_radius)
        dLdx = L * k * (overlap_p - overlap_m) / (2 * dx)

        I = self._circuit_I
        Q_cap = self._circuit_Q_cap

        def derivs(I_v, Q_v):
            V_cap = Q_v / C
            back_emf = dLdx * (vel_axial * 1e-3) * I_v
            dI = (V_cap - R * I_v - back_emf) / L_eff
            dQ = -I_v
            return dI, dQ

        k1_I, k1_Q = derivs(I, Q_cap)
        k2_I, k2_Q = derivs(I + 0.5 * dt * k1_I, Q_cap + 0.5 * dt * k1_Q)
        k3_I, k3_Q = derivs(I + 0.5 * dt * k2_I, Q_cap + 0.5 * dt * k2_Q)
        k4_I, k4_Q = derivs(I + dt * k3_I, Q_cap + dt * k3_Q)

        self._circuit_I = I + (dt / 6) * (k1_I + 2 * k2_I + 2 * k3_I + k4_I)
        self._circuit_Q_cap = Q_cap + (dt / 6) * (k1_Q + 2 * k2_Q + 2 * k3_Q + k4_Q)

        if p.has_flyback_diode and self._circuit_I < 0:
            self._circuit_I = 0.0

    def _on_physics_step(self, dt):
        """Called each PhysX timestep — compute and apply EM force via PhysX API."""
        self._sim_time += dt
        self._log_counter = getattr(self, '_log_counter', 0) + 1

        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        marble_prim = stage.GetPrimAtPath("/World/Marble")
        if not marble_prim:
            return

        # Read marble state
        xform = UsdGeom.Xformable(marble_prim)
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos = world_transform.ExtractTranslation()
        marble_pos = [pos[0], pos[1], pos[2]]

        rb_api = UsdPhysics.RigidBodyAPI(marble_prim)
        vel_attr = rb_api.GetVelocityAttr()
        cur_vel = vel_attr.Get() if vel_attr else None

        # Coil-local coordinates
        p = self._params
        coil_pos = self._coil_position
        coil_axis = self._coil_axis
        ax_len = math.sqrt(sum(a * a for a in coil_axis))
        coil_axis_n = [a / ax_len for a in coil_axis]

        relative = [marble_pos[i] - coil_pos[i] for i in range(3)]
        z_along = sum(relative[i] * coil_axis_n[i] for i in range(3))
        radial_vec = [relative[i] - z_along * coil_axis_n[i] for i in range(3)]
        r = math.sqrt(sum(v * v for v in radial_vec))

        # --- Multi-Gate IR Sensor System ---
        prev_z = self._prev_z_along
        self._prev_z_along = z_along

        if prev_z is not None:
            for gate_name, gate_pos in p.gates.items():
                if self._gate_triggered.get(gate_name):
                    continue
                crossed = (prev_z < gate_pos and z_along >= gate_pos) or \
                          (prev_z > gate_pos and z_along <= gate_pos)
                if crossed:
                    self._gate_triggered[gate_name] = True
                    self._gate_times[gate_name] = self._sim_time
                    carb.log_warn(f"[IR] Gate '{gate_name}' crossed at z_along={z_along:.1f}mm "
                                  f"t={self._sim_time:.4f}s")

        # Compute approach velocity from vel_in pair
        t1 = self._gate_times.get("vel_in_1")
        t2 = self._gate_times.get("vel_in_2")
        if t1 is not None and t2 is not None and self._approach_velocity is None:
            dt_gates = t2 - t1
            if abs(dt_gates) > 1e-6:
                dist = p.gates["vel_in_2"] - p.gates["vel_in_1"]
                self._approach_velocity = dist / dt_gates
                carb.log_warn(f"[IR] Approach velocity: {self._approach_velocity:.1f} mm/s "
                              f"(dt={dt_gates*1000:.2f}ms over {dist:.0f}mm)")

                dist_to_entry = p.gates["entry"] - p.gates["vel_in_2"]
                eta = dist_to_entry / self._approach_velocity if self._approach_velocity > 0 else 0
                self._predicted_fire_time = self._sim_time + eta
                carb.log_warn(f"[IR] Predicted entry in {eta*1000:.1f}ms "
                              f"(fire at t={self._predicted_fire_time:.4f}s)")

        # Entry gate: trigger the coil
        if not self._triggered and self._gate_triggered.get("entry"):
            self._triggered = True
            self._trigger_time = self._sim_time
            vel_str = f", v_approach={self._approach_velocity:.0f}mm/s" if self._approach_velocity else ""
            carb.log_warn(f"[EM] COIL FIRED at z_along={z_along:.1f}mm "
                          f"t={self._sim_time:.4f}s{vel_str}")

        if not self._triggered:
            return

        # Cutoff gate: kill the pulse
        if p.switch_type == "MOSFET" and not self._pulse_cut:
            if self._gate_triggered.get("cutoff"):
                self._pulse_cut = True
                self._pulse_cut_time = self._sim_time
                carb.log_warn(f"[EM] PULSE CUT at z_along={z_along:.1f}mm "
                              f"t={self._sim_time:.4f}s")

        # Compute gate-measured exit velocity from vel_out pair
        t_o1 = self._gate_times.get("vel_out_1")
        t_o2 = self._gate_times.get("vel_out_2")
        if t_o1 is not None and t_o2 is not None and not getattr(self, '_gate_exit_measured', False):
            dt_out = t_o2 - t_o1
            if abs(dt_out) > 1e-6:
                self._gate_exit_measured = True
                dist_out = p.gates["vel_out_2"] - p.gates["vel_out_1"]
                gate_exit_vel = dist_out / dt_out
                carb.log_warn(f"[IR] EXIT VELOCITY (gate-measured): {gate_exit_vel:.1f} mm/s "
                              f"= {gate_exit_vel/1000:.2f} m/s "
                              f"(dt={dt_out*1000:.2f}ms over {dist_out:.0f}mm)")
                if self._approach_velocity and self._approach_velocity > 0:
                    boost = gate_exit_vel / self._approach_velocity
                    carb.log_warn(f"[IR] Speed boost: {boost:.1f}x "
                                  f"({self._approach_velocity:.0f} -> {gate_exit_vel:.0f} mm/s)")
                self._exit_velocity = gate_exit_vel

        # Compute current
        vel_axial = 0.0
        if cur_vel is not None:
            vel_axial = sum(cur_vel[i] * coil_axis_n[i] for i in range(3))

        if not self._pulse_cut:
            self._coupled_rlc_step(dt, z_along, vel_axial)
            current = getattr(self, '_circuit_I_rms', abs(self._circuit_I))
        else:
            dt_cut = self._sim_time - self._pulse_cut_time
            I_at_cut = self._rlc_current(self._pulse_cut_time - self._trigger_time)
            current = I_at_cut * math.exp(-(p.R_total / p.inductance_H) * dt_cut)
            self._circuit_I = current

        if abs(current) < 1e-6:
            return

        # Compute EM force via PINN B-field
        Bz = self._em.get_Bz(r, z_along, current)
        B_now = abs(Bz)
        dBdt = (B_now - self._prev_B) / dt if dt > 0 else 0
        self._prev_B = B_now

        F_r, F_z_mN = self._em.compute_force(r, z_along, current, dBdt=dBdt, vel_axial=vel_axial)

        # Apply force via PhysX-native API (mN -> force in stage units)
        # PhysX in Kit: stage units = mm, mass in kg, force in mN gives correct acceleration
        F_x_mN = float(F_z_mN * coil_axis_n[0] + F_r * (radial_vec[0] / max(r, 1e-6)))
        F_y_mN = float(F_z_mN * coil_axis_n[1] + F_r * (radial_vec[1] / max(r, 1e-6)))
        F_z_world_mN = float(F_z_mN * coil_axis_n[2] + F_r * (radial_vec[2] / max(r, 1e-6)))

        force_vec = carb._carb.Float3(F_x_mN, F_y_mN, F_z_world_mN)
        pos_vec = carb._carb.Float3(marble_pos[0], marble_pos[1], marble_pos[2])
        self._physx_sim.apply_force_at_pos(self._stage_id, self._marble_prim_id, force_vec, pos_vec)

        # Compute dv for logging/exit velocity estimates only (informational)
        force_N = F_z_mN * 1e-3
        accel_ms2 = force_N / p.marble_mass_kg
        dv_mm_s = accel_ms2 * 1000.0 * dt

        # Compute exit velocity directly from impulse (gates can't catch sub-step motion)
        if self._exit_velocity is None and abs(dv_mm_s) > 1.0:
            v_approach = self._approach_velocity or 0.0
            self._exit_velocity = v_approach + abs(dv_mm_s)
            boost = self._exit_velocity / v_approach if v_approach > 0 else float('inf')
            carb.log_warn(
                f"[EM] LAUNCH: dv={dv_mm_s:.0f}mm/s, "
                f"v_approach={v_approach:.0f} -> v_exit={self._exit_velocity:.0f}mm/s "
                f"= {self._exit_velocity/1000:.2f}m/s (boost: {boost:.1f}x)"
            )

        # Wire temperature update
        R_now = p.R_dc * (1 + COPPER_TEMP_COEFF * (self._wire_temp - 20.0))
        R_total = R_now + p.esr + p.wiring_resistance
        if p.wire_mass_g > 0:
            self._wire_temp += current ** 2 * R_total * dt / (p.wire_mass_g * COPPER_SPECIFIC_HEAT)

        # Capacitor voltage
        V_cap = self._circuit_Q_cap / (p.capacitance_uF * 1e-6)

        # Update real-time display
        if self._log_counter % 10 == 1:
            v_in = f"{self._approach_velocity:.0f}" if self._approach_velocity else "--"
            v_out = f"{self._exit_velocity:.0f}" if self._exit_velocity else "--"
            self._realtime_label.text = (
                f"I: {current:.1f}A | F: {F_z_mN:.0f}mN | "
                f"V_cap: {V_cap:.0f}V | T: {self._wire_temp:.0f}C\n"
                f"v_in: {v_in} | v_out: {v_out} mm/s"
            )

        # Log
        if not self._pulse_cut or self._log_counter % 25 == 1:
            carb.log_warn(
                f"[EM] t={self._sim_time:.4f}s I={current:.1f}A "
                f"Bz={Bz*1e3:.2f}mT F={F_z_mN:.1f}mN dv={dv_mm_s:.1f}mm/s "
                f"V_cap={V_cap:.0f}V T={self._wire_temp:.1f}C "
                f"z={z_along:.1f}mm"
            )
