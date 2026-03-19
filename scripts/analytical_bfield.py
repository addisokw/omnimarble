"""Analytical B-field computation for a finite solenoid using Biot-Savart with elliptic integrals.

All units in mm, A, T (Tesla). Forces in mN.
μ₀ = 4π × 10⁻⁷ T·m/A = 4π × 10⁻⁴ T·mm/A
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipe, ellipk

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "coil_params.json"
PLOTS_DIR = ROOT / "results" / "plots"

MU_0_MM = 4 * math.pi * 1e-4  # T·mm/A


def single_loop_field(r: float, z: float, R: float, I: float) -> tuple[float, float]:
    """Compute B_r and B_z from a single circular current loop.

    Args:
        r: radial distance from axis (mm)
        z: axial distance from loop plane (mm)
        R: loop radius (mm)
        I: current (A)

    Returns:
        (B_r, B_z) in Tesla
    """
    r = abs(r)

    if r < 1e-10:
        # On axis — simplified formula
        B_z = MU_0_MM * I * R**2 / (2 * (R**2 + z**2) ** 1.5)
        return 0.0, B_z

    # Compute parameter m = k² for scipy's ellipk/ellipe
    alpha_sq = R**2 + r**2 + z**2 - 2 * R * r
    beta_sq = R**2 + r**2 + z**2 + 2 * R * r

    if beta_sq < 1e-20:
        return 0.0, 0.0

    m = 1.0 - alpha_sq / beta_sq  # m = 4Rr / beta²
    m = np.clip(m, 0, 1 - 1e-12)

    K = ellipk(m)
    E = ellipe(m)

    beta = math.sqrt(beta_sq)
    prefactor = MU_0_MM * I / (2 * math.pi)

    # B_z
    B_z = prefactor / beta * (K + (R**2 - r**2 - z**2) / alpha_sq * E)

    # B_r
    B_r = prefactor * z / (r * beta) * (-K + (R**2 + r**2 + z**2) / alpha_sq * E)

    return float(B_r), float(B_z)


def solenoid_field(r: float, z: float, coil_params: dict) -> tuple[float, float]:
    """Compute B-field of a finite solenoid by summing loop contributions.

    The solenoid is centered at origin, extending from -L/2 to L/2 along z-axis.
    Uses the mean coil radius = (inner + outer) / 2.

    Args:
        r: radial distance (mm)
        z: axial position (mm)
        coil_params: dict with coil parameters

    Returns:
        (B_r, B_z) in Tesla
    """
    N = coil_params["num_turns"]
    L = coil_params["length_mm"]
    R_mean = (coil_params["inner_radius_mm"] + coil_params["outer_radius_mm"]) / 2
    I = coil_params.get("current_A", coil_params.get("max_current_A", 10.0))

    # Each turn contributes current I, distributed over length L
    # Position each loop evenly along the solenoid
    z_positions = np.linspace(-L / 2, L / 2, N)

    B_r_total = 0.0
    B_z_total = 0.0

    for z_loop in z_positions:
        dz = z - z_loop
        b_r, b_z = single_loop_field(r, dz, R_mean, I)
        B_r_total += b_r
        B_z_total += b_z

    return B_r_total, B_z_total


def solenoid_field_gradient(
    r: float, z: float, coil_params: dict, dr: float = 0.1, dz: float = 0.1
) -> tuple[float, float, float, float]:
    """Compute B-field gradients via central finite differences.

    Returns:
        (dBr_dr, dBr_dz, dBz_dr, dBz_dz)
    """
    # dBr/dr
    Br_p, _ = solenoid_field(r + dr, z, coil_params)
    Br_m, _ = solenoid_field(max(r - dr, 0), z, coil_params)
    dBr_dr = (Br_p - Br_m) / (2 * dr) if r > dr else (Br_p - Br_m) / (r + dr)

    # dBr/dz
    Br_zp, _ = solenoid_field(r, z + dz, coil_params)
    Br_zm, _ = solenoid_field(r, z - dz, coil_params)
    dBr_dz = (Br_zp - Br_zm) / (2 * dz)

    # dBz/dr
    _, Bz_p = solenoid_field(r + dr, z, coil_params)
    _, Bz_m = solenoid_field(max(r - dr, 0), z, coil_params)
    dBz_dr = (Bz_p - Bz_m) / (2 * dr) if r > dr else (Bz_p - Bz_m) / (r + dr)

    # dBz/dz
    _, Bz_zp = solenoid_field(r, z + dz, coil_params)
    _, Bz_zm = solenoid_field(r, z - dz, coil_params)
    dBz_dz = (Bz_zp - Bz_zm) / (2 * dz)

    return dBr_dr, dBr_dz, dBz_dr, dBz_dz


def ferromagnetic_force(
    r: float, z: float, marble_radius: float, coil_params: dict,
    chi_eff: float = 3.0,
) -> tuple[float, float]:
    """Compute force on a ferromagnetic sphere in the B-field.

    F = (χ_eff * V / μ₀) * (B·∇)B

    Args:
        r: radial position (mm)
        z: axial position (mm)
        marble_radius: sphere radius (mm)
        coil_params: coil parameters
        chi_eff: effective susceptibility (dimensionless, ~3 for steel bearing, demagnetization-corrected)

    Returns:
        (F_r, F_z) in mN
    """
    V = (4 / 3) * math.pi * marble_radius**3  # mm³

    B_r, B_z = solenoid_field(r, z, coil_params)
    dBr_dr, dBr_dz, dBz_dr, dBz_dz = solenoid_field_gradient(r, z, coil_params)

    # (B·∇)B components in cylindrical coords
    # F_r = (χV/μ₀) * (B_r * dB_r/dr + B_z * dB_r/dz)
    # F_z = (χV/μ₀) * (B_r * dB_z/dr + B_z * dB_z/dz)
    prefactor = chi_eff * V / MU_0_MM  # mm² · A (gives force in T²·mm²/T·mm·A⁻¹ → needs unit conversion)

    # Convert: T² · mm² / (T·mm/A) = T·A·mm = 1e-3 N·mm⁻¹ · mm = mN... let's be careful
    # B is in T, gradient is T/mm, V in mm³, μ₀ in T·mm/A
    # F = χ·V/μ₀ · B · dB/dx → [mm³/(T·mm/A)] · T · T/mm = mm³·A·T/(T·mm²) = A·T·mm
    # 1 A·T·mm = 1 A · (kg/(A·s²)) · (1e-3 m) = 1e-3 kg·m/s² = 1e-3 N = 1 mN ✓

    F_r = prefactor * (B_r * dBr_dr + B_z * dBr_dz)  # mN
    F_z = prefactor * (B_r * dBz_dr + B_z * dBz_dz)  # mN

    return float(F_r), float(F_z)


def solenoid_field_batch(r_arr, z_arr, coil_params):
    """Vectorized B-field computation for arrays of (r, z) points.

    ~100x faster than calling solenoid_field() in a Python loop.
    Computes all N_turns loop contributions for all points simultaneously
    using numpy broadcasting.

    Args:
        r_arr: (M,) array of radial positions (mm)
        z_arr: (M,) array of axial positions (mm)
        coil_params: dict with coil parameters

    Returns:
        (Br, Bz) arrays of shape (M,) in Tesla
    """
    N = coil_params["num_turns"]
    L = coil_params["length_mm"]
    R = (coil_params["inner_radius_mm"] + coil_params["outer_radius_mm"]) / 2
    I = coil_params.get("current_A", coil_params.get("max_current_A", 10.0))

    r = np.abs(np.asarray(r_arr, dtype=np.float64))  # (M,)
    z = np.asarray(z_arr, dtype=np.float64)            # (M,)
    z_loops = np.linspace(-L / 2, L / 2, N)           # (N,)

    # dz[i, j] = z[i] - z_loops[j], shape (M, N)
    dz = z[:, None] - z_loops[None, :]

    # Broadcast r to (M, N)
    r_2d = r[:, None] * np.ones(N)[None, :]

    # Elliptic integral parameters
    alpha_sq = R**2 + r_2d**2 + dz**2 - 2 * R * r_2d  # (M, N)
    beta_sq = R**2 + r_2d**2 + dz**2 + 2 * R * r_2d   # (M, N)
    beta_sq = np.maximum(beta_sq, 1e-30)

    m = 1.0 - alpha_sq / beta_sq
    m = np.clip(m, 0, 1 - 1e-12)

    K = ellipk(m)
    E = ellipe(m)

    beta = np.sqrt(beta_sq)
    alpha_sq_safe = np.where(np.abs(alpha_sq) < 1e-30, 1e-30, alpha_sq)
    prefactor = MU_0_MM * I / (2 * math.pi)

    # Bz from each loop
    Bz_loops = prefactor / beta * (K + (R**2 - r_2d**2 - dz**2) / alpha_sq_safe * E)

    # Br from each loop (handle r=0 case)
    r_safe = np.where(r_2d < 1e-10, 1e-10, r_2d)
    Br_loops = prefactor * dz / (r_safe * beta) * (-K + (R**2 + r_2d**2 + dz**2) / alpha_sq_safe * E)

    # On-axis: Br=0, Bz uses simplified formula
    on_axis = r_2d < 1e-10
    Bz_on_axis = MU_0_MM * I * R**2 / (2 * (R**2 + dz**2) ** 1.5)
    Bz_loops = np.where(on_axis, Bz_on_axis, Bz_loops)
    Br_loops = np.where(on_axis, 0.0, Br_loops)

    # Sum over loops
    Br_total = Br_loops.sum(axis=1)  # (M,)
    Bz_total = Bz_loops.sum(axis=1)  # (M,)

    return Br_total.astype(np.float64), Bz_total.astype(np.float64)


def solenoid_field_gradient_batch(r_arr, z_arr, coil_params, dr=0.1, dz_step=0.1):
    """Vectorized B-field gradient computation via central finite differences.

    Returns:
        (dBr_dr, dBr_dz, dBz_dr, dBz_dz) arrays of shape (M,)
    """
    r = np.asarray(r_arr, dtype=np.float64)
    z = np.asarray(z_arr, dtype=np.float64)

    # dBz/dz
    _, Bz_zp = solenoid_field_batch(r, z + dz_step, coil_params)
    _, Bz_zm = solenoid_field_batch(r, z - dz_step, coil_params)
    dBz_dz = (Bz_zp - Bz_zm) / (2 * dz_step)

    # dBz/dr
    _, Bz_rp = solenoid_field_batch(r + dr, z, coil_params)
    _, Bz_rm = solenoid_field_batch(np.maximum(r - dr, 0), z, coil_params)
    # Use one-sided diff near r=0
    denom_r = np.where(r > dr, 2 * dr, r + dr)
    dBz_dr = (Bz_rp - Bz_rm) / denom_r

    # dBr/dr
    Br_rp, _ = solenoid_field_batch(r + dr, z, coil_params)
    Br_rm, _ = solenoid_field_batch(np.maximum(r - dr, 0), z, coil_params)
    dBr_dr = (Br_rp - Br_rm) / denom_r

    # dBr/dz
    Br_zp, _ = solenoid_field_batch(r, z + dz_step, coil_params)
    Br_zm, _ = solenoid_field_batch(r, z - dz_step, coil_params)
    dBr_dz = (Br_zp - Br_zm) / (2 * dz_step)

    return dBr_dr, dBr_dz, dBz_dr, dBz_dz


def validate_and_plot(coil_params: dict):
    """Generate validation plots for the B-field computation."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. B-field contour map
    r_grid = np.linspace(0, 60, 100)
    z_grid = np.linspace(-80, 80, 200)
    R, Z = np.meshgrid(r_grid, z_grid)

    Br = np.zeros_like(R)
    Bz = np.zeros_like(R)
    Bmag = np.zeros_like(R)

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            br, bz = solenoid_field(R[i, j], Z[i, j], coil_params)
            Br[i, j] = br
            Bz[i, j] = bz
            Bmag[i, j] = math.sqrt(br**2 + bz**2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # |B| contour
    c0 = axes[0].contourf(Z, R, np.log10(Bmag + 1e-10), levels=30, cmap="viridis")
    plt.colorbar(c0, ax=axes[0], label="log₁₀(|B| / T)")
    axes[0].set_xlabel("z (mm)")
    axes[0].set_ylabel("r (mm)")
    axes[0].set_title("|B| field magnitude")

    # B_z on axis
    z_axis = np.linspace(-80, 80, 500)
    bz_axis = np.array([solenoid_field(0, z, coil_params)[1] for z in z_axis])
    axes[1].plot(z_axis, bz_axis * 1e3, "b-", linewidth=2)
    axes[1].set_xlabel("z (mm)")
    axes[1].set_ylabel("B_z (mT)")
    axes[1].set_title("B_z on axis (r=0)")
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    L = coil_params["length_mm"]
    axes[1].axvspan(-L / 2, L / 2, alpha=0.1, color="red", label="coil region")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # B_r off axis — should be 0 at r=0
    r_test = [0, 2, 5, 10, 20]
    for r_val in r_test:
        br_line = np.array([solenoid_field(r_val, z, coil_params)[0] for z in z_axis])
        axes[2].plot(z_axis, br_line * 1e3, label=f"r={r_val} mm")
    axes[2].set_xlabel("z (mm)")
    axes[2].set_ylabel("B_r (mT)")
    axes[2].set_title("B_r vs z at various r")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "bfield_validation.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'bfield_validation.png'}")

    # 2. Force plot
    z_force = np.linspace(-60, 60, 300)
    forces_r = np.zeros_like(z_force)
    forces_z = np.zeros_like(z_force)
    for i, z in enumerate(z_force):
        fr, fz = ferromagnetic_force(0, z, 5.0, coil_params)
        forces_r[i] = fr
        forces_z[i] = fz

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z_force, forces_z, "r-", linewidth=2, label="F_z (axial)")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Force (mN)")
    ax.set_title("Axial force on marble along coil axis (r=0)")
    ax.axvspan(-L / 2, L / 2, alpha=0.1, color="blue", label="coil region")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "em_force_on_axis.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'em_force_on_axis.png'}")


def main():
    params = json.loads(CONFIG_PATH.read_text())

    # Quick validation: on-axis center field
    Br_0, Bz_0 = solenoid_field(0, 0, params)
    print(f"B-field at center (r=0, z=0): B_r={Br_0:.6f} T, B_z={Bz_0:.6f} T ({Bz_0*1e3:.3f} mT)")

    # Symmetry check: B_r should be 0 on axis
    Br_check, _ = solenoid_field(0, 10, params)
    print(f"B_r at (r=0, z=10): {Br_check:.2e} T (should be ~0)")

    # Force at entry
    Fr, Fz = ferromagnetic_force(0, -20, 5.0, params)
    print(f"Force at (r=0, z=-20mm): F_r={Fr:.4f} mN, F_z={Fz:.4f} mN")

    validate_and_plot(params)


if __name__ == "__main__":
    main()
