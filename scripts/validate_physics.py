"""Validate the electromagnetic launch physics against known values.

Checks:
1. B-field magnitude at coil center vs textbook formula
2. Force magnitude vs order-of-magnitude estimate
3. Expected launch velocity vs energy conservation
4. Comparison with real-world coilgun data
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "coil_params.json"

sys.path.insert(0, str(ROOT / "scripts"))
from analytical_bfield import solenoid_field, ferromagnetic_force, MU_0_MM


def main():
    params = json.loads(CONFIG_PATH.read_text())
    print("=" * 60)
    print("PHYSICS VALIDATION REPORT")
    print("=" * 60)

    # Coil parameters
    N = params["num_turns"]
    L = params["length_mm"]
    R_inner = params["inner_radius_mm"]
    R_outer = params["outer_radius_mm"]
    R_mean = (R_inner + R_outer) / 2
    I = params["max_current_A"]
    V = params["supply_voltage_V"]
    R_ohm = params["resistance_ohm"]
    L_H = params["inductance_uH"] * 1e-6
    pulse_ms = params["pulse_width_ms"]
    marble_radius = 5.0  # mm
    marble_volume_mm3 = (4/3) * math.pi * marble_radius**3
    marble_density_kg_m3 = 7800  # steel
    marble_volume_m3 = marble_volume_mm3 * 1e-9
    marble_mass_kg = marble_density_kg_m3 * marble_volume_m3
    marble_mass_g = marble_mass_kg * 1000
    chi_eff = 100.0

    print(f"\n--- Coil Parameters ---")
    print(f"  Turns: {N}")
    print(f"  Length: {L} mm")
    print(f"  Mean radius: {R_mean} mm")
    print(f"  Max current: {I} A (V/R = {V}/{R_ohm})")
    print(f"  Inductance: {L_H*1e6:.0f} uH")
    print(f"  Pulse width: {pulse_ms} ms")
    print(f"  LR time constant: {L_H/R_ohm*1e6:.1f} us")

    print(f"\n--- Marble Parameters ---")
    print(f"  Radius: {marble_radius} mm")
    print(f"  Volume: {marble_volume_mm3:.1f} mm^3 = {marble_volume_m3:.2e} m^3")
    print(f"  Mass: {marble_mass_g:.2f} g = {marble_mass_kg:.4f} kg")
    print(f"  Susceptibility: {chi_eff}")

    # 1. B-field at center
    print(f"\n--- B-field Validation ---")
    Br_0, Bz_0 = solenoid_field(0, 0, params)
    # Textbook: B_center = mu0 * n * I where n = N/L (turns per unit length)
    n = N / (L * 1e-3)  # turns per meter
    B_textbook_infinite = 4 * math.pi * 1e-7 * n * I
    print(f"  B_z at center (analytical): {Bz_0*1e3:.3f} mT")
    print(f"  B_z infinite solenoid:      {B_textbook_infinite*1e3:.3f} mT")
    print(f"  Ratio (finite/infinite):    {Bz_0/B_textbook_infinite:.3f}")
    print(f"  (Should be < 1.0 for a short solenoid - OK)" if Bz_0 < B_textbook_infinite else "  WARNING: exceeds infinite solenoid value")

    # 2. Force at various positions
    print(f"\n--- Force Profile ---")
    print(f"  Position (z)   |  B_z (mT)  | dB/dz (T/m) |  F_z (mN)  |  F_z (N)")
    print(f"  " + "-" * 65)
    for z in [-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30]:
        Fr, Fz = ferromagnetic_force(0, z, marble_radius, params)
        _, Bz = solenoid_field(0, z, params)
        _, Bz_p = solenoid_field(0, z + 0.1, params)
        _, Bz_m = solenoid_field(0, z - 0.1, params)
        dBz_dz = (Bz_p - Bz_m) / 0.2  # T/mm
        print(f"  z={z:+4d} mm      | {Bz*1e3:8.3f}  | {dBz_dz*1e3:10.4f}   | {Fz:9.3f}  | {Fz*1e-3:8.5f}")

    # 3. Peak force
    peak_F_mN = 0
    peak_z = 0
    for z_test in range(-40, 41):
        _, Fz = ferromagnetic_force(0, z_test, marble_radius, params)
        if abs(Fz) > abs(peak_F_mN):
            peak_F_mN = Fz
            peak_z = z_test
    print(f"\n  Peak force: {peak_F_mN:.2f} mN at z={peak_z} mm")
    print(f"  Peak force: {peak_F_mN*1e-3:.4f} N")
    print(f"  Peak acceleration: {peak_F_mN*1e-3/marble_mass_kg:.1f} m/s^2 = {peak_F_mN*1e-3/marble_mass_kg/9.81:.1f} g")

    # 4. Energy analysis
    print(f"\n--- Energy Analysis ---")
    # Electrical energy in pulse: E = V * I_avg * t_pulse
    # For LR circuit, I_avg ~ I_max * (1 - tau/t_pulse * (1 - e^(-t_pulse/tau)))
    tau = L_H / R_ohm
    t_pulse = pulse_ms * 1e-3
    I_max = V / R_ohm
    # Average current during pulse
    if t_pulse > 10 * tau:
        I_avg = I_max  # Fully ramped
    else:
        I_avg = I_max * (1 - tau / t_pulse * (1 - math.exp(-t_pulse / tau)))
    E_electrical = V * I_avg * t_pulse
    E_dissipated = I_avg**2 * R_ohm * t_pulse
    print(f"  Pulse energy: {E_electrical*1e3:.2f} mJ")
    print(f"  Resistive loss: {E_dissipated*1e3:.2f} mJ")

    # Integrate force over distance to get kinetic energy
    # Simplified: marble traverses from z=-20 to z=+20 (40mm through coil)
    dz_step = 0.5  # mm
    KE_gained = 0
    z = -30
    while z < 30:
        _, Fz = ferromagnetic_force(0, z, marble_radius, params)
        KE_gained += Fz * dz_step * 1e-6  # mN * mm = uJ -> convert to J
        z += dz_step

    # Net KE depends on whether force is pulling in then pushing out
    # For ferromagnetic attraction, force pulls marble IN toward center
    # but also resists exit - need to time pulse cutoff
    print(f"  Work done by EM force (z=-30 to +30, steady current): {KE_gained*1e3:.4f} mJ")

    # If all work converts to KE: v = sqrt(2*KE/m)
    if KE_gained > 0:
        v_exit = math.sqrt(2 * KE_gained / marble_mass_kg)
        print(f"  Theoretical exit velocity (upper bound): {v_exit:.3f} m/s = {v_exit*1e3:.1f} mm/s")
    else:
        print(f"  Net work is negative (marble decelerates through center)")
        print(f"  This is expected - the pulse must be CUT before marble reaches center!")

    # With properly timed pulse (cut at center):
    KE_half = 0
    z = -30
    while z < 0:
        _, Fz = ferromagnetic_force(0, z, marble_radius, params)
        KE_half += Fz * dz_step * 1e-6
        z += dz_step
    if KE_half > 0:
        v_half = math.sqrt(2 * KE_half / marble_mass_kg)
        print(f"\n  With pulse cut at center (z=0):")
        print(f"    Work done: {KE_half*1e3:.4f} mJ")
        print(f"    Exit velocity: {v_half:.3f} m/s = {v_half*1e3:.1f} mm/s")
        print(f"    Height marble could reach: {KE_half / (marble_mass_kg * 9.81) * 1e3:.1f} mm")

    # 5. Sanity checks
    print(f"\n--- Sanity Checks ---")
    # Real hobbyist coilguns: ~1-5 m/s with similar parameters
    print(f"  Typical hobbyist coilgun velocity: 1-10 m/s")
    print(f"  Our predicted velocity: {v_half:.2f} m/s" if KE_half > 0 else "  Could not compute")

    # Check if force makes sense: F ~ chi * V * B * dB/dx / mu0
    B_typical = Bz_0
    dB_typical = abs(peak_F_mN * MU_0_MM / (chi_eff * marble_volume_mm3 * B_typical))
    print(f"  B at center: {B_typical*1e3:.2f} mT")
    print(f"  Marble weight: {marble_mass_kg * 9.81 * 1e3:.2f} mN")
    print(f"  Peak EM force / weight ratio: {abs(peak_F_mN) / (marble_mass_kg * 9.81 * 1e3):.1f}x")

    # Current in simulation
    print(f"\n--- Simulation Parameters ---")
    print(f"  PhysX timestep: 2ms (500 Hz)")
    print(f"  dv per step at peak force: {abs(peak_F_mN)*1e-3/marble_mass_kg * 0.002 * 1000:.2f} mm/s")
    print(f"  Steps during pulse: {int(t_pulse / 0.002)}")
    print(f"  Current reaches {I_max*(1-math.exp(-t_pulse/tau)):.2f}A / {I_max:.1f}A during pulse")
    print(f"  (tau={tau*1e6:.1f}us << pulse={t_pulse*1e3:.1f}ms, so current is ~constant)")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
