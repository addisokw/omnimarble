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
            if abs(s1 - s2) > 1e-12:
                I = (self.charge_voltage / (self.inductance_H * (s1 - s2))) * \
                    (math.exp(s1 * t_since_trigger) - math.exp(s2 * t_since_trigger))
            else:
                I = 0.0

        return max(I, 0.0) if self.has_flyback_diode else I
