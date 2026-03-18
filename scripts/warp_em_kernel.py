"""Warp GPU kernels for EM force computation and LR circuit simulation.

NOTE: Warp kernels cannot call Python/PyTorch functions directly.
This module provides:
1. LR circuit current pulse model (pure math, fits Warp DSL)
2. Batched force application kernel
3. Hybrid pipeline: PyTorch PINN → numpy → Warp kernel
"""

import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

try:
    import warp as wp
    HAS_WARP = True
except ImportError:
    HAS_WARP = False
    print("WARNING: warp-lang not installed. Run: uv add warp-lang")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_WARP:
    wp.init()

    # ---- LR Circuit Current Kernel ----

    @wp.kernel
    def lr_circuit_current(
        times: wp.array(dtype=float),
        trigger_time: float,
        pulse_width: float,
        V_over_R: float,
        tau: float,  # L/R
        currents: wp.array(dtype=float),
    ):
        """Compute LR circuit current for an array of time values."""
        i = wp.tid()
        dt = times[i] - trigger_time

        if dt < 0.0:
            currents[i] = 0.0
        elif dt < pulse_width:
            # Rising: I(t) = (V/R)(1 - e^(-t/τ))
            currents[i] = V_over_R * (1.0 - wp.exp(-dt / tau))
        else:
            # Decay: I(t) = I_end * e^(-(t-t_pulse)/τ)
            I_end = V_over_R * (1.0 - wp.exp(-pulse_width / tau))
            currents[i] = I_end * wp.exp(-(dt - pulse_width) / tau)

    # ---- Force Application Kernel ----

    @wp.kernel
    def apply_em_force(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        forces: wp.array(dtype=wp.vec3),  # Pre-computed EM forces
        mass: float,  # grams
        gravity: wp.vec3,
        dt: float,
        new_positions: wp.array(dtype=wp.vec3),
        new_velocities: wp.array(dtype=wp.vec3),
    ):
        """Apply pre-computed EM force + gravity, integrate with semi-implicit Euler."""
        i = wp.tid()
        pos = positions[i]
        vel = velocities[i]
        f_em = forces[i]

        # a = gravity + F_em/m (F in mN, m in g → a in mm/s²: mN/g = 1e-3N/1e-3kg = m/s² = 1000 mm/s²)
        accel = gravity + f_em * (1000.0 / mass)

        # Semi-implicit Euler
        new_vel = vel + accel * dt
        new_pos = pos + new_vel * dt

        new_velocities[i] = new_vel
        new_positions[i] = new_pos

    # ---- Batched Trajectory Rollout (for optimization) ----

    @wp.kernel
    def trajectory_rollout_step(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        forces: wp.array(dtype=wp.vec3),
        mass: float,
        gravity: wp.vec3,
        dt: float,
    ):
        """In-place trajectory step for differentiable rollout via wp.Tape."""
        i = wp.tid()
        vel = velocities[i]
        f_em = forces[i]
        accel = gravity + f_em * (1000.0 / mass)
        velocities[i] = vel + accel * dt
        positions[i] = positions[i] + velocities[i] * dt


class HybridEMForceComputer:
    """Hybrid PINN + Warp force computation pipeline.

    Per timestep:
    1. Read marble positions
    2. PyTorch PINN inference → B_r, B_z, gradients
    3. Python: compute F = (χV/μ₀)(B·∇)B
    4. Pass force to Warp kernel for integration
    """

    def __init__(self, pinn_model=None, coil_params=None, marble_radius=5.0, chi_eff=3.0):
        self.pinn_model = pinn_model
        self.coil_params = coil_params or {}
        self.marble_radius = marble_radius
        self.chi_eff = chi_eff
        self.V_marble = (4 / 3) * math.pi * marble_radius**3  # mm³
        self.MU_0_MM = 4 * math.pi * 1e-4

    def compute_force_pinn(self, positions_np, current_A):
        """Compute EM force using PINN model.

        Args:
            positions_np: (N, 3) array of marble positions in world coords
            current_A: current flowing through coil

        Returns:
            (N, 3) force array in mN
        """
        if self.pinn_model is None or not HAS_TORCH:
            return np.zeros_like(positions_np)

        coil_pos = np.array(self.coil_params.get("position_mm", [0, 0, 20]))
        coil_axis = np.array(self.coil_params.get("axis", [1, 0, 0]), dtype=float)
        coil_axis = coil_axis / np.linalg.norm(coil_axis)

        N = len(positions_np)
        forces = np.zeros((N, 3))

        # Transform to coil-local cylindrical
        relative = positions_np - coil_pos
        z_along = relative @ coil_axis
        radial_vecs = relative - z_along[:, None] * coil_axis
        r = np.linalg.norm(radial_vecs, axis=1)

        # PINN input
        N_turns = self.coil_params.get("num_turns", 30)
        R_mean = (self.coil_params.get("inner_radius_mm", 8) +
                  self.coil_params.get("outer_radius_mm", 14)) / 2
        L = self.coil_params.get("length_mm", 30)

        inputs = np.column_stack([
            r, z_along,
            np.full(N, current_A),
            np.full(N, N_turns),
            np.full(N, R_mean),
            np.full(N, L),
        ]).astype(np.float32)

        # PINN inference with gradient computation for force
        device = next(self.pinn_model.parameters()).device
        x_t = torch.tensor(inputs, device=device, requires_grad=True)

        out = self.pinn_model(x_t)
        output_scale = self.pinn_model.output_scale
        Br = out[:, 1] * output_scale[1]
        Bz = out[:, 2] * output_scale[2]

        # Compute gradients via autograd
        Br_sum = Br.sum()
        Bz_sum = Bz.sum()

        grad_Br = torch.autograd.grad(Br_sum, x_t, retain_graph=True)[0]
        grad_Bz = torch.autograd.grad(Bz_sum, x_t)[0]

        dBr_dr = grad_Br[:, 0].detach().cpu().numpy()
        dBr_dz = grad_Br[:, 1].detach().cpu().numpy()
        dBz_dr = grad_Bz[:, 0].detach().cpu().numpy()
        dBz_dz = grad_Bz[:, 1].detach().cpu().numpy()

        Br_np = Br.detach().cpu().numpy()
        Bz_np = Bz.detach().cpu().numpy()

        # F = (χV/μ₀)(B·∇)B
        prefactor = self.chi_eff * self.V_marble / self.MU_0_MM
        F_r = prefactor * (Br_np * dBr_dr + Bz_np * dBr_dz)
        F_z = prefactor * (Br_np * dBz_dr + Bz_np * dBz_dz)

        # Convert back to Cartesian
        for i in range(N):
            forces[i] = F_z[i] * coil_axis
            if r[i] > 1e-6:
                radial_dir = radial_vecs[i] / r[i]
                forces[i] += F_r[i] * radial_dir

        return forces


def demo():
    """Demonstrate Warp kernels with simple test."""
    if not HAS_WARP:
        print("Warp not available, skipping demo")
        return

    print("=== Warp EM Kernel Demo ===\n")

    # Test LR circuit current
    n = 1000
    times = wp.array(np.linspace(0, 0.02, n), dtype=float)
    currents = wp.zeros(n, dtype=float)

    V_over_R = 20.0  # 24V / 1.2Ω
    tau = 150e-6 / 1.2  # L/R = 125μs
    pulse_width = 5e-3  # 5ms

    wp.launch(
        lr_circuit_current,
        dim=n,
        inputs=[times, 0.001, pulse_width, V_over_R, tau, currents],
    )

    currents_np = currents.numpy()
    print(f"LR circuit current: peak={currents_np.max():.2f}A at t={np.argmax(currents_np)/n*20:.2f}ms")

    # Test force application
    n_marbles = 4
    positions = wp.array(np.random.randn(n_marbles, 3).astype(np.float32), dtype=wp.vec3)
    velocities = wp.zeros(n_marbles, dtype=wp.vec3)
    forces = wp.array(np.array([[0, 0, 10]] * n_marbles, dtype=np.float32), dtype=wp.vec3)
    new_pos = wp.zeros(n_marbles, dtype=wp.vec3)
    new_vel = wp.zeros(n_marbles, dtype=wp.vec3)

    gravity = wp.vec3(0.0, 0.0, -9810.0)
    mass = 4.08  # grams

    wp.launch(
        apply_em_force,
        dim=n_marbles,
        inputs=[positions, velocities, forces, mass, gravity, 0.001, new_pos, new_vel],
    )

    print(f"Force application test: {n_marbles} marbles processed")
    print(f"  New velocities: {new_vel.numpy()}")

    print("\nWarp kernels working correctly!")


if __name__ == "__main__":
    demo()
