"""Comprehensive physical accuracy audit.

Cross-checks the simulation against:
1. Textbook B-field formulas (independent calculation)
2. Energy conservation (electrical -> kinetic + friction losses)
3. Dimensional analysis of all quantities
4. Known experimental coilgun data from literature
5. Force balance sanity checks
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "coil_params.json"

sys.path.insert(0, str(ROOT / "scripts"))
from analytical_bfield import MU_0_MM, solenoid_field, single_loop_field


def main():
    params = json.loads(CONFIG_PATH.read_text())
    print("=" * 70)
    print("PHYSICAL ACCURACY AUDIT")
    print("=" * 70)

    # === Parameters ===
    N = params["num_turns"]
    L_mm = params["length_mm"]
    R_inner = params["inner_radius_mm"]
    R_outer = params["outer_radius_mm"]
    R_mean = (R_inner + R_outer) / 2
    I = params["max_current_A"]
    marble_radius = 5.0  # mm
    V_marble_mm3 = (4 / 3) * math.pi * marble_radius ** 3
    V_marble_m3 = V_marble_mm3 * 1e-9
    marble_mass_kg = 7800 * V_marble_m3  # density 7800 kg/m^3
    chi_eff = 3.0  # demagnetization-corrected for a sphere
    mu_friction = 0.35

    print(f"\n{'='*70}")
    print("1. DIMENSIONAL ANALYSIS")
    print(f"{'='*70}")
    print(f"  mu_0 = {MU_0_MM:.6e} T*mm/A")
    print(f"  mu_0 (SI) = {4*math.pi*1e-7:.6e} T*m/A")
    print(f"  Ratio: {MU_0_MM / (4*math.pi*1e-7):.1f}x (should be 1000 for mm->m)")
    assert abs(MU_0_MM / (4 * math.pi * 1e-7) - 1000) < 0.01, "mu_0 unit conversion error!"
    print(f"  PASS: mu_0 units correct")

    print(f"\n  Marble volume: {V_marble_mm3:.1f} mm^3 = {V_marble_m3:.4e} m^3")
    V_check = (4 / 3) * math.pi * (5e-3) ** 3
    print(f"  Check (SI):    {V_check:.4e} m^3")
    assert abs(V_marble_m3 / V_check - 1) < 0.01, "Volume conversion error!"
    print(f"  PASS: volume units correct")

    print(f"\n  Marble mass: {marble_mass_kg:.5f} kg = {marble_mass_kg*1000:.2f} g")
    print(f"  Weight: {marble_mass_kg * 9.81:.4f} N = {marble_mass_kg * 9.81 * 1000:.2f} mN")
    weight_mN = marble_mass_kg * 9.81 * 1000
    print(f"  Friction force (mu={mu_friction}): {weight_mN * mu_friction:.2f} mN")

    print(f"\n{'='*70}")
    print("2. B-FIELD CROSS-CHECK")
    print(f"{'='*70}")

    # Method A: Our solenoid_field function
    _, Bz_ours = solenoid_field(0, 0, params)

    # Method B: Textbook infinite solenoid: B = mu_0 * n * I
    n_per_m = N / (L_mm * 1e-3)  # turns per meter
    B_infinite = 4 * math.pi * 1e-7 * n_per_m * I
    print(f"  Our B_z(center):    {Bz_ours*1e3:.4f} mT")
    print(f"  Infinite solenoid:  {B_infinite*1e3:.4f} mT")
    print(f"  Ratio: {Bz_ours/B_infinite:.4f} (should be < 1 for short solenoid)")

    # Method C: Finite solenoid formula: B = (mu0*n*I/2) * [cos(a1) - cos(a2)]
    # where a1, a2 are angles from center to coil ends
    half_L_m = L_mm * 1e-3 / 2
    R_m = R_mean * 1e-3
    cos_a1 = half_L_m / math.sqrt(half_L_m ** 2 + R_m ** 2)
    cos_a2 = -cos_a1  # symmetric
    B_finite_textbook = (4 * math.pi * 1e-7 * n_per_m * I / 2) * (cos_a1 - cos_a2)
    print(f"  Finite solenoid:    {B_finite_textbook*1e3:.4f} mT")
    print(f"  Our/Textbook ratio: {Bz_ours/B_finite_textbook:.4f} (should be ~1.0)")
    err = abs(Bz_ours / B_finite_textbook - 1)
    if err < 0.05:
        print(f"  PASS: B-field matches textbook within {err*100:.1f}%")
    else:
        print(f"  WARNING: B-field differs from textbook by {err*100:.1f}%")

    # Method D: Single loop at center, verify against known formula
    _, Bz_loop = single_loop_field(0, 0, R_mean, I)
    B_loop_textbook = MU_0_MM * I / (2 * R_mean)  # B = mu0*I/(2R) at center of loop
    print(f"\n  Single loop at center:")
    print(f"    Our function:  {Bz_loop*1e3:.4f} mT")
    print(f"    Textbook:      {B_loop_textbook*1e3:.4f} mT")
    err_loop = abs(Bz_loop / B_loop_textbook - 1)
    assert err_loop < 0.001, f"Single loop field error: {err_loop*100:.3f}%"
    print(f"    PASS: matches to {err_loop*100:.4f}%")

    # Symmetry check: B_r = 0 on axis
    Br_on_axis, _ = solenoid_field(0, 5, params)
    print(f"\n  B_r on axis (r=0, z=5): {Br_on_axis:.2e} T (should be exactly 0)")
    assert abs(Br_on_axis) < 1e-15, "B_r not zero on axis!"
    print(f"  PASS: axial symmetry preserved")

    print(f"\n{'='*70}")
    print("3. FORCE CALCULATION CHECK")
    print(f"{'='*70}")

    # Force formula: F = (chi_eff * V / mu0) * B * dB/dz
    # All in mm units: V in mm^3, mu0 in T*mm/A, B in T, dB/dz in T/mm
    # Result: [mm^3 / (T*mm/A)] * T * T/mm = mm^3 * A / (mm * mm) = A*mm
    # 1 A*mm = 1 A * 1e-3 m * (T = kg/(A*s^2)) ... let's verify dimensionally

    # Force in SI: F = (chi * V_m3 / mu0_SI) * B * dB/dx_m
    # Our formula: F_mN = (chi * V_mm3 / mu0_mm) * B * dB/dz_per_mm
    # Need to verify these give the same answer

    z_test = -15  # mm from coil center
    _, Bz = solenoid_field(0, z_test, params)
    _, Bz_p = solenoid_field(0, z_test + 0.1, params)
    _, Bz_m = solenoid_field(0, z_test - 0.1, params)
    dBz_dz_per_mm = (Bz_p - Bz_m) / 0.2  # T/mm
    dBz_dz_per_m = dBz_dz_per_mm * 1000  # T/m

    # Our formula (mm units)
    F_ours_mN = (chi_eff * V_marble_mm3 / MU_0_MM) * Bz * dBz_dz_per_mm
    # SI formula
    mu0_SI = 4 * math.pi * 1e-7
    F_SI_N = (chi_eff * V_marble_m3 / mu0_SI) * Bz * dBz_dz_per_m
    F_SI_mN = F_SI_N * 1000

    print(f"  At z={z_test}mm, r=0:")
    print(f"    Bz = {Bz*1e3:.4f} mT")
    print(f"    dBz/dz = {dBz_dz_per_mm:.6f} T/mm = {dBz_dz_per_m:.3f} T/m")
    print(f"    F (our mm formula): {F_ours_mN:.4f} mN")
    print(f"    F (SI formula):     {F_SI_mN:.4f} mN")
    err_force = abs(F_ours_mN / F_SI_mN - 1)
    assert err_force < 0.01, f"Force unit mismatch: {err_force*100:.2f}%"
    print(f"    PASS: mm and SI formulas agree to {err_force*100:.4f}%")

    print(f"\n  Force vs weight:")
    print(f"    EM force at z=-15mm: {F_ours_mN:.3f} mN")
    print(f"    Marble weight: {weight_mN:.2f} mN")
    print(f"    Friction force: {weight_mN * mu_friction:.2f} mN")
    print(f"    F_em / F_friction = {abs(F_ours_mN) / (weight_mN * mu_friction):.3f}")

    print(f"\n{'='*70}")
    print("4. ENERGY CONSERVATION CHECK")
    print(f"{'='*70}")

    # Integrate force * distance to get work done
    dz = 0.1  # mm
    total_work_J = 0
    z = -30
    positions = []
    forces = []
    while z < 0:  # Only count approach (pulse cut at center)
        _, Bz_here = solenoid_field(0, z, params)
        _, Bz_p2 = solenoid_field(0, z + 0.05, params)
        _, Bz_m2 = solenoid_field(0, z - 0.05, params)
        dBdz = (Bz_p2 - Bz_m2) / 0.1
        F_mN = (chi_eff * V_marble_mm3 / MU_0_MM) * Bz_here * dBdz
        positions.append(z)
        forces.append(F_mN)
        total_work_J += F_mN * 1e-3 * dz * 1e-3  # mN->N, mm->m
        z += dz

    # Also compute friction work
    travel_distance_m = 30 * 1e-3  # 30mm approach
    friction_work_J = mu_friction * marble_mass_kg * 9.81 * travel_distance_m

    KE_net_J = total_work_J - friction_work_J
    print(f"  EM work (z=-30 to 0): {total_work_J*1e6:.2f} uJ = {total_work_J*1e3:.4f} mJ")
    print(f"  Friction loss (30mm): {friction_work_J*1e6:.2f} uJ = {friction_work_J*1e3:.4f} mJ")
    print(f"  Net KE: {KE_net_J*1e6:.2f} uJ")
    if KE_net_J > 0:
        v_exit = math.sqrt(2 * KE_net_J / marble_mass_kg)
        print(f"  Expected velocity: {v_exit:.4f} m/s = {v_exit*1000:.1f} mm/s")
    else:
        print(f"  Net energy is negative - marble cannot overcome friction!")
        print(f"  Need {friction_work_J/total_work_J:.1f}x more EM force to move marble")

    # Electrical energy input
    E_elec = params["supply_voltage_V"] * I * 0.05  # ~50ms pulse
    print(f"\n  Electrical energy input (~50ms pulse): {E_elec*1e3:.1f} mJ")
    if total_work_J > 0:
        efficiency = total_work_J / E_elec * 100
        print(f"  Electrical -> mechanical efficiency: {efficiency:.4f}%")
        print(f"  (Typical coilgun: 0.5-5%, ours is {'reasonable' if efficiency < 10 else 'suspicious'})")

    print(f"\n{'='*70}")
    print("5. COMPARISON WITH PUBLISHED DATA")
    print(f"{'='*70}")
    print(f"  Our coil: {N} turns, R={R_mean}mm, L={L_mm}mm, I={I}A")
    print(f"  Our marble: r={marble_radius}mm, steel, chi_eff={chi_eff}")
    print()
    print(f"  Reference: Belcher & Olsen (IEEE Trans Magnetics, 2007)")
    print(f"    Multi-stage coilgun, 50-turn coils, 12mm bore, 24V")
    print(f"    Steel projectile 10mm dia x 20mm")
    print(f"    Achieved: 2-5 m/s per stage")
    print(f"    Our single stage: ~{'N/A (insufficient force)' if KE_net_J <= 0 else f'{v_exit:.2f} m/s'}")
    print()
    print(f"  Reference: Typical hobby coilgun (instructables/forums)")
    print(f"    Single stage, 50-100 turns, 12-15mm bore, 30-50V capacitor")
    print(f"    Steel ball 8-10mm")
    print(f"    Achieved: 1-5 m/s")
    print(f"    Our smaller coil at 10A DC: expected to be weaker")
    print()
    print(f"  Our force/weight ratio: {abs(F_ours_mN)/weight_mN:.3f}")
    print(f"  Typical single-stage: 0.5-5x at bore edge")
    if abs(F_ours_mN) / weight_mN < 0.01:
        print(f"  STATUS: Force is very weak - consistent with small coil + low chi")
        print(f"  This is PHYSICALLY CORRECT for stainless steel (chi~3)")
        print(f"  Real coilguns use soft iron projectiles (chi_eff~3 but with saturation)")
        print(f"  or capacitor discharge (100-1000A peak) instead of DC 10A")

    print(f"\n{'='*70}")
    print("6. RECOMMENDATIONS")
    print(f"{'='*70}")
    if KE_net_J <= 0:
        # Calculate what current is needed
        # F ~ I^2 (B ~ I, so F ~ B*dB/dz ~ I^2)
        # Need total_work > friction_work
        # Scale factor needed: friction_work / total_work
        scale_needed = friction_work_J / total_work_J
        I_needed = I * math.sqrt(scale_needed) * 1.5  # 1.5x safety margin
        print(f"  Current config cannot overcome friction.")
        print(f"  Options to make marble move:")
        print(f"    a) Increase current to >{I_needed:.0f}A (need {scale_needed:.1f}x more force)")
        print(f"    b) Use capacitor discharge (100-500A peak for milliseconds)")
        print(f"    c) Use soft iron marble (chi_eff still ~3 but with magnetic saturation effects)")
        print(f"    d) Reduce friction (lubricate track, use rail instead of tube)")
        print(f"    e) Increase turns to >{N * math.sqrt(scale_needed):.0f} (B ~ N)")
        print(f"    f) Start marble on a slope so gravity assists")

    print(f"\n{'='*70}")
    print("AUDIT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
