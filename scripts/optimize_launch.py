"""Differentiable optimization of coil launch parameters.

Stretch goal — uses Warp's autodiff (wp.Tape) to optimize:
- Pulse timing, width, voltage
- Coil position, num turns

Loss: (exit_vel - target)² + penalty_derailed + energy_cost
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

try:
    import warp as wp
    HAS_WARP = True
except ImportError:
    HAS_WARP = False

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def optimize_with_torch(coil_params: dict, target_velocity: float = 2000.0):
    """Optimize launch parameters using PyTorch autodiff as fallback.

    Simplified 1D trajectory along coil axis for differentiability.

    Args:
        coil_params: base coil parameters
        target_velocity: desired exit velocity in mm/s
    """
    if not HAS_TORCH:
        print("ERROR: PyTorch required for optimization")
        return

    from analytical_bfield import MU_0_MM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Optimizing on: {device}")

    # Optimizable parameters
    supply_voltage = torch.tensor(coil_params["supply_voltage_V"], device=device, requires_grad=True)
    pulse_width = torch.tensor(coil_params["pulse_width_ms"] * 1e-3, device=device, requires_grad=True)
    coil_z_offset = torch.tensor(0.0, device=device, requires_grad=True)  # Offset from default position

    # Fixed parameters
    R_ohm = coil_params["resistance_ohm"]
    L_H = coil_params["inductance_uH"] * 1e-6
    tau = L_H / R_ohm
    marble_radius = 5.0  # mm
    V_marble = (4 / 3) * math.pi * marble_radius**3
    chi_eff = 100.0
    marble_mass_g = V_marble * 7.8e-3
    N_turns = coil_params["num_turns"]
    R_mean = (coil_params["inner_radius_mm"] + coil_params["outer_radius_mm"]) / 2
    L_coil = coil_params["length_mm"]

    # Simplified 1D force model along axis
    # Precompute force profile shape from analytical solution
    z_samples = torch.linspace(-60, 60, 200, device=device)

    from analytical_bfield import solenoid_field, solenoid_field_gradient

    force_profile_np = np.zeros(200)
    for i, z in enumerate(z_samples.cpu().numpy()):
        Br, Bz = solenoid_field(0, z, coil_params)
        _, _, dBz_dr, dBz_dz = solenoid_field_gradient(0, z, coil_params)
        # On axis, F_z = (χV/μ₀) * Bz * dBz/dz (Br=0 on axis)
        force_profile_np[i] = Bz * dBz_dz

    force_profile = torch.tensor(force_profile_np, device=device, dtype=torch.float32)
    # Normalize so we can scale by current²
    I_ref = coil_params["max_current_A"]
    force_scale = chi_eff * V_marble / MU_0_MM  # mN per (T * T/mm)

    optimizer = torch.optim.Adam([supply_voltage, pulse_width, coil_z_offset], lr=0.5)

    dt = 0.0001  # 0.1ms
    n_steps = 2000  # 200ms simulation
    target_vel_t = torch.tensor(target_velocity, device=device)

    best_loss = float("inf")
    best_params = {}
    loss_history = []

    print(f"\nOptimizing for target exit velocity: {target_velocity} mm/s")
    print(f"Initial: V={supply_voltage.item():.1f}V, "
          f"pulse={pulse_width.item()*1e3:.2f}ms")

    for epoch in range(200):
        optimizer.zero_grad()

        # Forward simulation
        z = torch.tensor(-40.0, device=device)  # Start 40mm before coil center
        vz = torch.tensor(0.0, device=device)
        triggered = False
        trigger_step = 0

        # Simple Euler integration
        max_vel = torch.tensor(0.0, device=device)
        total_energy = torch.tensor(0.0, device=device)

        for step in range(n_steps):
            t = step * dt

            # Gravity
            a_grav = -9810.0  # mm/s² downward (but we're in 1D along coil axis)

            # EM force
            if not triggered and z > -20:
                triggered = True
                trigger_step = step

            a_em = torch.tensor(0.0, device=device)
            if triggered:
                dt_trigger = (step - trigger_step) * dt
                # LR current
                I_max = supply_voltage / R_ohm
                if dt_trigger < pulse_width:
                    I_t = I_max * (1 - torch.exp(torch.tensor(-dt_trigger / tau, device=device)))
                else:
                    I_end = I_max * (1 - torch.exp(torch.tensor(-pulse_width.item() / tau, device=device)))
                    I_t = I_end * torch.exp(torch.tensor(-(dt_trigger - pulse_width.item()) / tau, device=device))

                # Interpolate force from profile
                z_shifted = z - coil_z_offset
                z_idx = (z_shifted + 60) / 120 * 199
                z_idx = torch.clamp(z_idx, 0, 198)
                idx_low = z_idx.long()
                idx_high = torch.clamp(idx_low + 1, max=199)
                frac = z_idx - idx_low.float()
                f_interp = force_profile[idx_low] * (1 - frac) + force_profile[idx_high] * frac

                # Scale by (I/I_ref)² since B ∝ I and force ∝ B²
                force_mN = force_scale * f_interp * (I_t / I_ref) ** 2
                a_em = force_mN * 1000.0 / marble_mass_g

                # Energy tracking
                total_energy = total_energy + supply_voltage * I_t * dt

            # Integration (1D along coil axis, ignoring gravity for simplicity)
            vz = vz + a_em * dt
            z = z + vz * dt

            max_vel = torch.maximum(max_vel, torch.abs(vz))

        # Loss
        vel_loss = (vz - target_vel_t) ** 2 / target_velocity**2
        energy_loss = 0.01 * total_energy
        # Penalize extreme voltages
        voltage_penalty = 0.1 * torch.relu(supply_voltage - 48) ** 2 + 0.1 * torch.relu(5 - supply_voltage) ** 2
        # Penalize extreme pulse widths
        pw_penalty = 0.1 * torch.relu(pulse_width - 0.05) ** 2 + 0.1 * torch.relu(0.001 - pulse_width) ** 2

        loss = vel_loss + energy_loss + voltage_penalty + pw_penalty

        loss.backward()
        optimizer.step()

        # Clamp to physical bounds
        with torch.no_grad():
            supply_voltage.clamp_(5, 48)
            pulse_width.clamp_(0.5e-3, 50e-3)
            coil_z_offset.clamp_(-20, 20)

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_params = {
                "supply_voltage_V": supply_voltage.item(),
                "pulse_width_ms": pulse_width.item() * 1e3,
                "coil_z_offset_mm": coil_z_offset.item(),
                "exit_velocity_mm_s": vz.item(),
                "max_velocity_mm_s": max_vel.item(),
            }

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | loss={loss_val:.4f} | "
                  f"V={supply_voltage.item():.1f}V, pw={pulse_width.item()*1e3:.2f}ms, "
                  f"v_exit={vz.item():.1f} mm/s")

    print(f"\n=== Optimization Results ===")
    print(f"  Target velocity: {target_velocity} mm/s")
    print(f"  Best exit velocity: {best_params['exit_velocity_mm_s']:.1f} mm/s")
    print(f"  Supply voltage: {best_params['supply_voltage_V']:.1f} V")
    print(f"  Pulse width: {best_params['pulse_width_ms']:.2f} ms")
    print(f"  Coil Z offset: {best_params['coil_z_offset_mm']:.1f} mm")

    # Plot convergence
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(loss_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Launch Parameter Optimization Convergence")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "optimization_convergence.png", dpi=150)
    plt.close()
    print(f"\nSaved: {PLOTS_DIR / 'optimization_convergence.png'}")

    return best_params


def main():
    print("=== Coil Launch Parameter Optimization ===\n")
    params = json.loads(CONFIG_PATH.read_text())
    optimize_with_torch(params, target_velocity=2000.0)


if __name__ == "__main__":
    main()
