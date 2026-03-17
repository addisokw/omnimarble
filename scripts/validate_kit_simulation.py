"""Headless simulation that replicates the Kit extension's exact physics.

Runs the same force computation + velocity integration as the Kit extension,
but in a standalone script so we can validate without launching the GUI.
Uses the track mesh for collision detection.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "coil_params.json"
PLOTS_DIR = ROOT / "results" / "plots"

sys.path.insert(0, str(ROOT / "scripts"))
from analytical_bfield import MU_0_MM, solenoid_field
from run_physics_test import load_track_mesh, check_collision, MARBLE_RADIUS, GRAVITY


def main():
    params = json.loads(CONFIG_PATH.read_text())

    # Match Kit extension parameters exactly
    chi_eff = 100.0  # soft iron/steel marble
    marble_radius = 5.0  # mm
    V_marble = (4 / 3) * math.pi * marble_radius ** 3
    marble_mass_kg = V_marble * 7.8e-3 * 1e-3  # THE FIXED MASS
    marble_mass_g = marble_mass_kg * 1000

    coil_pos = np.array(params["position_mm"], dtype=float)
    coil_axis = np.array(params.get("axis", [0, 1, 0]), dtype=float)
    coil_axis = coil_axis / np.linalg.norm(coil_axis)

    max_current = params["max_current_A"]
    R_ohm = params["resistance_ohm"]
    L_H = params["inductance_uH"] * 1e-6
    tau = L_H / R_ohm
    pulse_width_s = params["pulse_width_ms"] * 1e-3

    R_mean = (params["inner_radius_mm"] + params["outer_radius_mm"]) / 2

    print(f"=== Headless Kit Simulation Validation ===")
    print(f"Marble mass: {marble_mass_g:.2f} g = {marble_mass_kg:.5f} kg")
    print(f"Chi_eff: {chi_eff}")
    print(f"Coil: pos={coil_pos}, axis={coil_axis}")
    print(f"Max current: {max_current} A, tau={tau*1e6:.0f} us")
    print(f"Pulse width: {pulse_width_s*1e3:.1f} ms")

    # Load track
    mesh = load_track_mesh()
    print(f"Track: {len(mesh.faces)} faces")

    # Initial marble position (same as create_marble.py)
    verts = mesh.vertices
    start_mask = verts[:, 1] < 20
    start_verts = verts[start_mask]
    cx = start_verts[:, 0].mean()
    track_top_z = start_verts[:, 2].max()
    pos = np.array([cx, 10.0, track_top_z + marble_radius + 1])
    vel = np.zeros(3)

    print(f"Start pos: {pos}")

    # Simulation
    dt = 0.002  # 500Hz like Kit with PhysX configured
    total_time = 3.0
    n_steps = int(total_time / dt)

    triggered = False
    trigger_time = 0.0
    pulse_cut = False
    pulse_cut_time = 0.0

    times = []
    positions = []
    velocities = []
    forces_log = []
    currents_log = []

    for step in range(n_steps):
        t = step * dt
        times.append(t)
        positions.append(pos.copy())
        velocities.append(vel.copy())

        # Gravity (let collision handle track contact)
        accel_gravity = GRAVITY  # mm/s^2

        # Coil-local coords
        relative = pos - coil_pos
        z_along = np.dot(relative, coil_axis)
        radial_vec = relative - z_along * coil_axis
        r = np.linalg.norm(radial_vec)

        # Trigger
        if not triggered and z_along < 0:
            triggered = True
            trigger_time = t
            print(f"  t={t:.4f}s: TRIGGERED, z_along={z_along:.1f}mm")

        # EM force
        em_accel = np.zeros(3)
        current = 0.0
        F_z_mN = 0.0

        if triggered:
            # Pulse cut
            if z_along > 0 and not pulse_cut:
                pulse_cut = True
                pulse_cut_time = t
                print(f"  t={t:.4f}s: PULSE CUT, z_along={z_along:.1f}mm")

            # Current — pulse stays ON until marble passes coil center (z_along > 0)
            # The pulse_width_ms in config is a max safety limit, not the primary cutoff
            dt_trigger = t - trigger_time
            if pulse_cut:
                dt_cut = t - pulse_cut_time
                current = max_current * math.exp(-dt_cut / tau)
            elif dt_trigger < 1.0:  # Keep current on until pulse_cut (up to 1s safety)
                current = max_current * (1 - math.exp(-dt_trigger / tau))
            else:
                current = 0.0

            if step < 10:
                print(f"    [current debug] step={step} dt_trig={dt_trigger:.6f} pulse_cut={pulse_cut} "
                      f"current={current:.6f} z_along={z_along:.2f}")
            if abs(current) > 1e-8:
                # B-field and force (same as Kit extension)
                params_now = params.copy()
                params_now["current_A"] = current
                _, Bz = solenoid_field(r, z_along, params_now)
                _, Bz_p = solenoid_field(r, z_along + 0.1, params_now)
                _, Bz_m = solenoid_field(r, z_along - 0.1, params_now)
                dBz_dz = (Bz_p - Bz_m) / 0.2

                prefactor = chi_eff * V_marble / MU_0_MM
                F_z_mN = prefactor * Bz * dBz_dz
                force_N = F_z_mN * 1e-3

                # a = F/m, convert to mm/s^2
                accel_ms2 = force_N / marble_mass_kg
                dv_mm_s = accel_ms2 * 1000.0  # mm/s^2
                em_accel = dv_mm_s * coil_axis  # Along coil axis

        currents_log.append(current)
        forces_log.append(F_z_mN)

        # Total acceleration
        total_accel = accel_gravity + em_accel

        # Semi-implicit Euler
        vel = vel + total_accel * dt
        pos = pos + vel * dt

        # Collision
        _, pos, vel = check_collision(pos, vel, marble_radius, mesh)

        # Floor check
        if pos[2] < mesh.bounds[0][2] - 50:
            print(f"  t={t:.3f}s: Fell below floor")
            break

        # Periodic log — every step during first 0.05s after trigger, then every 50
        if step % 500 == 0 or (triggered and t < trigger_time + 0.05) or (triggered and step % 50 == 0 and t < trigger_time + 0.5):
            speed = np.linalg.norm(vel)
            print(f"  t={t:.4f}s: pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) "
                  f"vel_Y={vel[1]:.1f}mm/s speed={speed:.1f}mm/s "
                  f"I={current:.2f}A F={F_z_mN:.2f}mN")

    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)
    forces_log = np.array(forces_log)
    currents_log = np.array(currents_log)

    # Summary
    speed = np.linalg.norm(velocities, axis=1)
    max_speed = speed.max()
    max_y = positions[:, 1].max()
    max_z = positions[:, 2].max()
    print(f"\n=== Results ===")
    print(f"  Max speed: {max_speed:.1f} mm/s = {max_speed/1000:.3f} m/s")
    print(f"  Max Y (along track): {max_y:.1f} mm")
    print(f"  Max Z (height): {max_z:.1f} mm")
    print(f"  Final pos: ({positions[-1,0]:.1f}, {positions[-1,1]:.1f}, {positions[-1,2]:.1f})")

    # Expected from validation: ~869 mm/s with chi=100, so ~435 mm/s with chi=25 (force ~ chi)
    # Actually force ~ chi, KE ~ chi, v ~ sqrt(chi), so v ~ sqrt(25/100) * 869 = 434 mm/s
    print(f"\n  Expected exit velocity (from validate_physics.py, scaled): "
          f"~{869 * math.sqrt(25/100):.0f} mm/s")

    # Plot
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(times, positions[:, 1], 'b-')
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Y position (mm)")
    axes[0, 0].set_title("Y (along track) vs time")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(times, positions[:, 2], 'r-')
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Z position (mm)")
    axes[0, 1].set_title("Z (height) vs time")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(times, speed, 'g-')
    axes[0, 2].set_xlabel("Time (s)")
    axes[0, 2].set_ylabel("Speed (mm/s)")
    axes[0, 2].set_title("Speed vs time")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(times, currents_log, 'm-')
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Current (A)")
    axes[1, 0].set_title("Coil current")
    axes[1, 0].set_xlim(0, 0.2)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(times, forces_log, 'r-')
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Force (mN)")
    axes[1, 1].set_title("EM force")
    axes[1, 1].set_xlim(0, 0.2)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(positions[:, 1], positions[:, 2], 'b-', linewidth=0.5)
    axes[1, 2].plot(positions[0, 1], positions[0, 2], 'go', markersize=8, label='start')
    axes[1, 2].set_xlabel("Y (mm)")
    axes[1, 2].set_ylabel("Z (mm)")
    axes[1, 2].set_title("YZ trajectory (side view)")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f"Kit Simulation Validation (chi={chi_eff}, mass={marble_mass_g:.2f}g)", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kit_validation.png", dpi=150)
    plt.close()
    print(f"\nSaved: {PLOTS_DIR / 'kit_validation.png'}")


if __name__ == "__main__":
    main()
