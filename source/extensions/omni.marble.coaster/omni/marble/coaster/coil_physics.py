"""Pure-Python coil physics — no carb/omni dependencies.

Holds the derived-quantity computation (winding geometry, resistance,
Wheeler inductance, RLC regime) and the closed-form RLC current so they
can be unit-tested outside Kit. The Kit extension's CoilParams subclasses
CoilPhysics and layers JSON/carb-settings/UI handling on top.

Note: scripts/rlc_circuit.py implements the same physics for the headless
pipeline. The two are cross-checked against each other in tests
(tests/test_coil_physics.py) rather than unified — tracked as tech debt.

All units: mm, A, T, s, mN.
"""

import math

MU_0_MM = 4 * math.pi * 1e-4  # T*mm/A
COPPER_TEMP_COEFF = 0.00393  # /C
COPPER_SPECIFIC_HEAT = 0.385  # J/(g*C)
COPPER_RESISTIVITY_OHM_MM = 1.72e-5  # ohm*mm
COPPER_DENSITY_G_MM3 = 8.96e-3

STEEL_DENSITY_G_MM3 = 7.8e-3

# How close zeta must be to 1.0 to be called critically damped. Must match
# rlc_circuit.compute_rlc_params or the two models classify the same circuit
# differently and the cross-check test fails intermittently -- the bench rig's
# 2-can point lands at zeta ~= 1.02, near enough to that seam to matter.
CRITICAL_ZETA_TOL = 1e-6

# Below this separation the overdamped two-exponential form is numerically 0/0
# and the critically-damped limit is used instead.
SPLIT_ROOT_TOL = 1e-12

DEFAULT_GATES = {
    "vel_in_1": -60.0,
    "vel_in_2": -40.0,
    "entry": -20.0,
    "cutoff": 5.0,
    "vel_out_1": 60.0,
    "vel_out_2": 120.0,
}


def gate_crossed(prev_z: float, z: float, gate_pos: float) -> bool:
    """True if the marble crossed the gate plane between two positions."""
    return (prev_z < gate_pos and z >= gate_pos) or \
           (prev_z > gate_pos and z <= gate_pos)


class CoilPhysics:
    """Coil + RLC circuit parameters with derived quantities.

    Construct with raw values, then all derived attributes (winding
    geometry, R_dc, Wheeler inductance, RLC regime, peak current, stored
    energy) are available. Call recompute_derived() after mutating raw
    attributes (e.g. UI overrides).
    """

    def __init__(self,
                 inner_radius=12.0, outer_radius=18.0, length=30.0,
                 num_turns=30, wire_diameter=0.8, insulation_thickness=0.035,
                 capacitance_uF=470.0, charge_voltage=50.0,
                 esr=0.01, wiring_resistance=0.02,
                 switch_type="MOSFET", has_flyback_diode=True,
                 marble_radius=5.0, chi_eff=3.0, B_sat=1.8, conductivity=6e6,
                 ambient_temp=20.0, gates=None):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.length = length
        self.num_turns = int(num_turns)
        self.wire_diameter = wire_diameter
        self.insulation_thickness = insulation_thickness

        self.capacitance_uF = capacitance_uF
        self.charge_voltage = charge_voltage
        self.esr = esr
        self.wiring_resistance = wiring_resistance
        self.switch_type = switch_type
        self.has_flyback_diode = has_flyback_diode

        self.marble_radius = marble_radius
        self.chi_eff = chi_eff
        self.B_sat = B_sat
        self.conductivity = conductivity
        self.ambient_temp = ambient_temp

        self.gates = dict(DEFAULT_GATES if gates is None else gates)
        self.sensor_entry_offset = self.gates["entry"]
        self.sensor_cutoff_offset = self.gates["cutoff"]

        self.recompute_derived()

    def recompute_derived(self):
        """Recompute all derived quantities from the raw attributes."""
        # Derived geometry
        self.R_mean = (self.inner_radius + self.outer_radius) / 2
        self.V_marble = (4 / 3) * math.pi * self.marble_radius ** 3
        self.marble_mass_kg = self.V_marble * STEEL_DENSITY_G_MM3 * 1e-3

        # Winding geometry
        wire_pitch = self.wire_diameter + 2 * self.insulation_thickness
        self.turns_per_layer = max(1, int(self.length / wire_pitch))
        self.num_layers = math.ceil(self.num_turns / self.turns_per_layer)
        winding_depth = self.num_layers * wire_pitch
        self.winding_depth = winding_depth
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

        # Provenance, so a caller can tell a measurement from an estimate.
        # apply_measured() flips these and keeps the derived value alongside.
        self.L_source = "derived"
        self.R_source = "derived"
        self.inductance_uH_derived = self.inductance_uH
        self.R_total_derived = self.R_total

        self._recompute_rlc()

    def apply_measured(self, inductance_uH=None, total_resistance_ohm=None):
        """Override the geometry-derived L and R with measured values.

        The bench is the arbiter. The rig's coil measures 17.9 uH where the
        geometry model predicts 17.3, and its LOOP resistance (0.164 ohm, the
        whole discharge path) is not the same quantity as coil DC resistance
        plus a guessed ESR -- so it is substituted wholesale rather than added
        to. Only the RLC block is recomputed; the winding geometry is untouched
        so the derived values stay inspectable alongside.
        """
        if inductance_uH is not None:
            self.inductance_uH_derived = self.inductance_uH
            self.inductance_uH = float(inductance_uH)
            self.inductance_H = self.inductance_uH * 1e-6
            self.L_source = "measured"
        if total_resistance_ohm is not None:
            self.R_total_derived = self.R_total
            self.R_total = float(total_resistance_ohm)
            self.R_source = "measured"
        self._recompute_rlc()

    def _recompute_rlc(self):
        """Damping, regime, peak current and stored energy from C, L, R."""
        # RLC parameters
        C = self.capacitance_uF * 1e-6
        L = self.inductance_H
        self.alpha = self.R_total / (2 * L) if L > 0 else 0
        self.omega_0 = 1.0 / math.sqrt(L * C) if L > 0 and C > 0 else 0
        self.zeta = self.alpha / self.omega_0 if self.omega_0 > 0 else 999

        V0 = self.charge_voltage
        if self.zeta < 1.0:
            self.omega_d = math.sqrt(self.omega_0 ** 2 - self.alpha ** 2)
            self.regime = "underdamped"
            self.peak_current_undamped = (V0 / (self.omega_d * L)) if L > 0 else 0
            self.t_peak = math.atan2(self.omega_d, self.alpha) / self.omega_d
            self.peak_current = (
                self.peak_current_undamped
                * math.exp(-self.alpha * self.t_peak)
                * math.sin(self.omega_d * self.t_peak)
            )
            self.t_zero_crossing = math.pi / self.omega_d
        elif abs(self.zeta - 1.0) < CRITICAL_ZETA_TOL:
            self.omega_d = 0
            self.regime = "critically_damped"
            self.t_peak = 1.0 / self.alpha if self.alpha > 0 else 0
            self.peak_current = (V0 / L) * self.t_peak * math.exp(-1.0) if L > 0 else 0
            self.peak_current_undamped = self.peak_current
            self.t_zero_crossing = 5.0 / self.alpha if self.alpha > 0 else 0
        else:
            self.omega_d = 0
            self.regime = "overdamped"
            # dI/dt = 0 -> s1*exp(s1*t) = s2*exp(s2*t) -> t = ln(s2/s1)/(s1-s2).
            # Both roots are negative with s1 > s2, so s2/s1 > 1 and t_peak > 0.
            # This branch used to hardcode peak_current = 0, which reported 0 A
            # for every overdamped bank -- i.e. most of the rig's C-sweep.
            s1 = -self.alpha + math.sqrt(self.alpha ** 2 - self.omega_0 ** 2)
            s2 = -self.alpha - math.sqrt(self.alpha ** 2 - self.omega_0 ** 2)
            self.t_peak = 0
            self.peak_current = 0
            if abs(s1 - s2) > SPLIT_ROOT_TOL and s1 != 0 and s2 != 0 and L > 0:
                t_peak = math.log(s2 / s1) / (s1 - s2)
                if t_peak > 0:
                    self.t_peak = t_peak
                    self.peak_current = abs(
                        (V0 / (L * (s1 - s2)))
                        * (math.exp(s1 * t_peak) - math.exp(s2 * t_peak))
                    )
            self.peak_current_undamped = self.peak_current
            self.t_zero_crossing = 5.0 / self.alpha if self.alpha > 0 else 0

        self.stored_energy = 0.5 * C * self.charge_voltage ** 2

    def rlc_current(self, t_since_trigger: float) -> float:
        """Closed-form RLC discharge current with flyback diode clamp."""
        if t_since_trigger < 0:
            return 0.0

        if self.regime == "underdamped":
            I = (self.charge_voltage / (self.omega_d * self.inductance_H)) * \
                math.exp(-self.alpha * t_since_trigger) * \
                math.sin(self.omega_d * t_since_trigger)
        elif self.regime == "critically_damped":
            I = (self.charge_voltage / self.inductance_H) * \
                t_since_trigger * math.exp(-self.alpha * t_since_trigger)
        else:
            s1 = -self.alpha + math.sqrt(self.alpha ** 2 - self.omega_0 ** 2)
            s2 = -self.alpha - math.sqrt(self.alpha ** 2 - self.omega_0 ** 2)
            if abs(s1 - s2) > SPLIT_ROOT_TOL:
                I = (self.charge_voltage / (self.inductance_H * (s1 - s2))) * \
                    (math.exp(s1 * t_since_trigger) - math.exp(s2 * t_since_trigger))
            else:
                # s1 -> s2: the two-exponential form is 0/0. Its limit is the
                # critically-damped waveform, so use that rather than returning
                # a silent zero -- the rig's 2-can point sits at zeta ~= 1.02,
                # close enough to land here.
                I = (self.charge_voltage / self.inductance_H) * \
                    t_since_trigger * math.exp(-self.alpha * t_since_trigger)

        return max(I, 0.0) if self.has_flyback_diode else I
