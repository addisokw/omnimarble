"""PINN-based B-field solver for headless simulation.

Uses trained PhysicsNeMo PINN for B-field inference. Requires torch,
physicsnemo, and the trained checkpoint at models/pinn_checkpoint/pinn_best.pt.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
from physicsnemo.models.mlp import FullyConnected as NeMoFC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from rlc_circuit import eddy_braking_force, saturated_force

MU_0_MM = 4 * math.pi * 1e-4  # T*mm/A
PINN_CHECKPOINT = ROOT / "models" / "pinn_checkpoint" / "pinn_best.pt"


def _load_pinn(device: torch.device) -> torch.nn.Module:
    """Load trained PINN checkpoint. Fails if missing."""
    backbone = NeMoFC(
        in_features=6,
        layer_size=256,
        out_features=3,
        num_layers=6,
        activation_fn="silu",
        skip_connections=True,
    )

    class PINNWrapper(torch.nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.backbone = bb
            self.register_buffer("input_mean", torch.zeros(6))
            self.register_buffer("input_std", torch.ones(6))
            self.register_buffer("output_scale", torch.ones(3))
            self.register_buffer("current_normalized", torch.tensor(False))

        def forward(self, x):
            x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
            return self.backbone(x_norm) * self.output_scale

    wrapper = PINNWrapper(backbone)
    checkpoint = torch.load(str(PINN_CHECKPOINT), map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        wrapper.load_state_dict(checkpoint["model_state_dict"])
    else:
        wrapper.load_state_dict(checkpoint)
    wrapper.to(device)
    wrapper.eval()

    # Cache the flag for fast access
    wrapper._current_normalized = bool(wrapper.current_normalized.item())
    return wrapper


class WarpBFieldSolver:
    """PINN-based B-field solver.

    Usage:
        solver = WarpBFieldSolver(coil_params)
        B, dBdz = solver.solve(current_A=100.0, marble_pos=np.array([0, 0, -10]))
        F_r, F_z = solver.get_force(current_A=100.0, marble_pos=..., marble_vel=...)
    """

    def __init__(self, coil_params: dict, resolution: int = 32, marble_radius: float = 5.0,
                 chi_eff: float = 3.0):
        self.coil_params = coil_params
        self.resolution = resolution
        self.marble_radius = marble_radius
        self.chi_eff = chi_eff
        self.V_marble = (4 / 3) * math.pi * marble_radius ** 3  # mm^3

        # Marble material properties from config
        self.B_sat = coil_params.get("marble_saturation_T", 1.8)
        self.conductivity = coil_params.get("marble_conductivity_S_per_m", 6e6)

        # Coil geometry for PINN inputs
        self._num_turns = float(coil_params.get("num_turns", 30))
        self._R_mean = (coil_params.get("inner_radius_mm", 12.0) +
                        coil_params.get("outer_radius_mm", 18.0)) / 2.0
        self._length = float(coil_params.get("length_mm", 30.0))

        # Load PINN
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _load_pinn(self._device)
        print(f"  [WarpBFieldSolver] PINN loaded on {self._device}")

    def _to_cylindrical(self, marble_pos: np.ndarray) -> tuple:
        """Convert position to (r, z) in coil-local cylindrical coords."""
        if len(marble_pos) == 2:
            return float(marble_pos[0]), float(marble_pos[1])
        coil_axis = np.array(self.coil_params.get("axis", [0, 1, 0]), dtype=float)
        coil_axis = coil_axis / np.linalg.norm(coil_axis)
        z = float(np.dot(marble_pos, coil_axis))
        radial = marble_pos - z * coil_axis
        r = float(np.linalg.norm(radial))
        return r, z

    def _pinn_field(self, r: float, z: float, current_A: float) -> tuple:
        """PINN forward pass returning (B_r, B_z)."""
        inp = torch.tensor(
            [[r, z, current_A, self._num_turns, self._R_mean, self._length]],
            dtype=torch.float32, device=self._device,
        )
        with torch.no_grad():
            out = self._model(inp)
        # If model outputs B/I, multiply by I to get actual B
        scale = current_A if self._model._current_normalized else 1.0
        return float(out[0, 1]) * scale, float(out[0, 2]) * scale

    def _pinn_field_with_grad(self, r: float, z: float, current_A: float) -> tuple:
        """PINN forward pass with autograd for gradients.

        Returns (B_r, B_z, dBr_dr, dBr_dz, dBz_dr, dBz_dz).
        """
        inp = torch.tensor(
            [[r, z, current_A, self._num_turns, self._R_mean, self._length]],
            dtype=torch.float32, device=self._device, requires_grad=True,
        )
        out = self._model(inp)
        B_r = out[0, 1]
        B_z = out[0, 2]

        grad_Br = torch.autograd.grad(
            B_r, inp, create_graph=False, retain_graph=True,
        )[0][0]
        grad_Bz = torch.autograd.grad(
            B_z, inp, create_graph=False, retain_graph=False,
        )[0][0]

        # If model outputs B/I, scale everything by I
        scale = current_A if self._model._current_normalized else 1.0
        return (
            B_r.detach().item() * scale, B_z.detach().item() * scale,
            grad_Br[0].item() * scale, grad_Br[1].item() * scale,
            grad_Bz[0].item() * scale, grad_Bz[1].item() * scale,
        )

    def solve(self, current_A: float, marble_pos: np.ndarray) -> tuple:
        """Solve for B-field at marble position.

        Returns:
            (B_at_marble, dBdz_at_marble) — B magnitude (T) and axial gradient (T/mm)
        """
        r, z = self._to_cylindrical(marble_pos)
        Br, Bz = self._pinn_field(r, z, current_A)
        B_mag = math.sqrt(Br ** 2 + Bz ** 2)

        # Get dBz/dz from autograd
        _, _, _, _, _, dBz_dz = self._pinn_field_with_grad(r, z, current_A)

        return B_mag, dBz_dz

    def get_force(self, current_A: float, marble_pos: np.ndarray,
                  marble_vel: np.ndarray = None, dBdt: float = 0.0) -> tuple:
        """Compute full force on marble including saturation and eddy currents.

        Returns:
            (F_r, F_z) in mN
        """
        r, z = self._to_cylindrical(marble_pos)

        Br, Bz, dBr_dr, dBr_dz, dBz_dr, dBz_dz = self._pinn_field_with_grad(r, z, current_A)
        B_mag = math.sqrt(Br ** 2 + Bz ** 2)

        # Force with saturation
        marble_params = {
            "chi_eff": self.chi_eff,
            "volume_mm3": self.V_marble,
            "saturation_T": self.B_sat,
            "conductivity_S_per_m": self.conductivity,
            "radius_mm": self.marble_radius,
        }

        # Axial force with saturation
        F_z = saturated_force(Bz, dBz_dz, marble_params)

        # Radial force
        prefactor = self.chi_eff * self.V_marble / MU_0_MM
        F_r = prefactor * (Br * dBr_dr + Bz * dBr_dz)

        # Eddy current braking
        if marble_vel is not None:
            vel_axial = 0.0
            if len(marble_vel) == 2:
                vel_axial = marble_vel[1]
            else:
                coil_axis = np.array(self.coil_params.get("axis", [0, 1, 0]), dtype=float)
                coil_axis = coil_axis / np.linalg.norm(coil_axis)
                vel_axial = float(np.dot(marble_vel, coil_axis))

            F_eddy = eddy_braking_force(dBdt, vel_axial, marble_params)
            F_z += F_eddy

        return float(F_r), float(F_z)

    def invalidate_cache(self):
        """No-op — PINN has no cache to invalidate."""
        pass


def demo():
    """Quick demo of the PINN B-field solver."""
    import json
    config = json.loads((ROOT / "config" / "coil_params.json").read_text())

    print("=== PINN B-Field Solver Demo ===\n")

    solver = WarpBFieldSolver(config, chi_eff=3.0)

    print(f"  {'z (mm)':>8} | {'B (mT)':>10} | {'dBz/dz (T/m)':>12} | {'F_z (mN)':>10}")
    print(f"  {'-'*50}")
    for z in [-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30]:
        B, dBdz = solver.solve(current_A=100.0, marble_pos=np.array([0.0, float(z)]))
        Fr, Fz = solver.get_force(current_A=100.0, marble_pos=np.array([0.0, float(z)]))
        print(f"  {z:+8d} | {B*1e3:10.4f} | {dBdz*1e3:12.4f} | {Fz:10.3f}")

    print(f"\n  Solver: PINN (PhysicsNeMo FullyConnected)")


if __name__ == "__main__":
    demo()
