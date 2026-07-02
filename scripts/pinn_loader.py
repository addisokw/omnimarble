"""Shared PINN checkpoint loader and inference helpers.

Single source of truth for:
  - Model architectures (legacy direct-B and v8 derived-B)
  - Checkpoint loading with version detection
  - Batch and scalar field/gradient inference

Two model generations are supported transparently:

  Legacy (v3-v7): backbone outputs (A_phi, B_r, B_z) directly. Physics
  (curl consistency, div-free) enforced only via training losses.

  Derived-B (v8+): backbone outputs a single scalar f, with the ansatz
  A_phi = r * f. The field is derived inside forward() via autograd:
      B_r = -dA/dz          = -r * df/dz
      B_z = (1/r) d(rA)/dr  = 2f + r * df/dr
  which makes div(B)=0, curl consistency, A(0,z)=0, and Br(0,z)=0 exact
  by construction (the r*f parameterization removes the 1/r singularity
  at the axis, no special casing needed). Field gradients used for force
  are then SECOND derivatives of f (double backward), which is why
  forward() builds its first-derivative graph with create_graph=True.

Callers never need to manage requires_grad / no_grad themselves: the
module-level predict functions handle both model generations.

All units: mm, A, T (mm-scaled mu_0 = 4*pi*1e-4 T*mm/A).
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "pinn_checkpoint"
DEFAULT_CHECKPOINT = MODEL_DIR / "pinn_best.pt"

try:
    from physicsnemo.models.mlp import FullyConnected as NeMoFC
    HAS_NEMO = True
except ImportError:
    HAS_NEMO = False


class SinActivation(nn.Module):
    """Sine activation -- better than ReLU for smooth PDE solutions."""
    def forward(self, x):
        return torch.sin(x)


def _build_backbone(in_dim, hidden_dim, num_layers, out_dim, backend):
    """Build the MLP backbone (PhysicsNeMo FullyConnected or torch fallback)."""
    if backend == "physicsnemo":
        if not HAS_NEMO:
            raise ImportError(
                "Checkpoint was trained with physicsnemo backend but "
                "nvidia-physicsnemo is not installed"
            )
        return NeMoFC(
            in_features=in_dim,
            layer_size=hidden_dim,
            out_features=out_dim,
            num_layers=num_layers,
            activation_fn="silu",
            skip_connections=True,
        )
    layers = [nn.Linear(in_dim, hidden_dim), SinActivation()]
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), SinActivation()])
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class BFieldPINNLegacy(nn.Module):
    """v3-v7 PINN: backbone outputs (A_phi, B_r, B_z) directly."""

    def __init__(self, in_dim=6, hidden_dim=256, num_layers=6, out_dim=3,
                 backend="physicsnemo" if HAS_NEMO else "pytorch"):
        super().__init__()
        self.backbone = _build_backbone(in_dim, hidden_dim, num_layers, out_dim, backend)
        self._use_nemo = backend == "physicsnemo"
        self.register_buffer("input_mean", torch.zeros(in_dim))
        self.register_buffer("input_std", torch.ones(in_dim))
        self.register_buffer("output_scale", torch.ones(out_dim))
        # When True, model outputs B/I -- caller must multiply by I
        self.register_buffer("current_normalized", torch.tensor(False))

    def forward(self, x):
        """Returns (A_phi, B_r, B_z). When current_normalized, values are B/I."""
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.backbone(x_norm) * self.output_scale


class BFieldPINNDerived(nn.Module):
    """v8+ PINN: backbone outputs scalar f, field derived from A_phi = r*f.

    forward() returns the same (A_phi, B_r, B_z) column contract as the
    legacy model, so downstream inference code is version-agnostic.
    """

    def __init__(self, in_dim=6, hidden_dim=256, num_layers=6,
                 backend="physicsnemo" if HAS_NEMO else "pytorch"):
        super().__init__()
        self.backbone = _build_backbone(in_dim, hidden_dim, num_layers, 1, backend)
        self._use_nemo = backend == "physicsnemo"
        self.register_buffer("input_mean", torch.zeros(in_dim))
        self.register_buffer("input_std", torch.ones(in_dim))
        # Scale on f (roughly max|B/I|; f ~ Bz/2 near the axis)
        self.register_buffer("a_scale", torch.tensor(1.0))
        self.register_buffer("current_normalized", torch.tensor(False))
        self.register_buffer("derived_b", torch.tensor(True))

    def forward(self, x):
        """Returns (A_phi, B_r, B_z) with B derived from A_phi by autograd.

        Differentiates w.r.t. the raw (unnormalized) input so the chain
        rule through input normalization is automatic. create_graph=True
        unconditionally: needed at train time (data loss backprops through
        the derivative) and at inference when force code takes a second
        derivative of the output.
        """
        with torch.enable_grad():
            if not x.requires_grad:
                x = x.detach().clone().requires_grad_(True)
            x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
            f = self.backbone(x_norm) * self.a_scale
            gf = torch.autograd.grad(
                f, x, grad_outputs=torch.ones_like(f), create_graph=True,
            )[0]
            f_r = gf[:, 0:1]
            f_z = gf[:, 1:2]
            r = x[:, 0:1]
            A = r * f
            B_r = -r * f_z
            B_z = 2.0 * f + r * f_r
        return torch.cat([A, B_r, B_z], dim=1)


def is_derived_b(model) -> bool:
    """True if the model derives B from A_phi (v8+)."""
    flag = getattr(model, "derived_b", None)
    return bool(flag.item()) if flag is not None else False


def load_model_from_checkpoint(ckpt_path, device):
    """Load a checkpoint into the matching architecture.

    Returns:
        (model, current_normalized, metadata) — model is eval-mode on device.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PINN checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Version detection: explicit checkpoint key, else state-dict inspection
    derived = bool(checkpoint.get("derived_b", False)) if isinstance(checkpoint, dict) else False
    if not derived and "a_scale" in state_dict:
        derived = True

    backend = checkpoint.get("backend", "physicsnemo") if isinstance(checkpoint, dict) else "physicsnemo"

    if derived:
        model = BFieldPINNDerived(backend=backend)
    else:
        model = BFieldPINNLegacy(backend=backend)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    current_normalized = bool(model.current_normalized.item())
    metadata = {
        "step": checkpoint.get("step", "?") if isinstance(checkpoint, dict) else "?",
        "loss": checkpoint.get("loss", "?") if isinstance(checkpoint, dict) else "?",
        "backend": backend,
        "current_normalized": current_normalized,
        "derived_b": derived,
        "path": str(ckpt_path),
    }
    return model, current_normalized, metadata


def _stack_inputs(r, z, I, N, R_mean, L):
    """Stack (r, z, I, N, R_mean, L) into an (n, 6) float32 array."""
    r = np.atleast_1d(np.asarray(r, dtype=np.float32))
    z = np.atleast_1d(np.asarray(z, dtype=np.float32))
    n = len(r)
    return np.column_stack([
        r, z,
        np.full(n, I, dtype=np.float32),
        np.full(n, N, dtype=np.float32),
        np.full(n, R_mean, dtype=np.float32),
        np.full(n, L, dtype=np.float32),
    ]).astype(np.float32)


def predict_field(model, r, z, I, N, R_mean, L, current_normalized, device):
    """Batch inference returning (Br, Bz) numpy arrays in Tesla.

    Works for both legacy and derived-B models (the derived model enables
    grad internally; outputs are detached here).
    """
    inputs = _stack_inputs(r, z, I, N, R_mean, L)
    x = torch.tensor(inputs, device=device)
    out = model(x).detach().cpu().numpy()
    scale = I if current_normalized else 1.0
    return out[:, 1] * scale, out[:, 2] * scale


def predict_field_with_grad(model, r, z, I, N, R_mean, L, current_normalized, device):
    """Batch inference with spatial gradients via autograd.

    For derived-B models this is a double backward (B is already a first
    derivative of f).

    Returns:
        (Br, Bz, dBr_dr, dBr_dz, dBz_dr, dBz_dz) as numpy arrays.
    """
    inputs = _stack_inputs(r, z, I, N, R_mean, L)
    inp_t = torch.tensor(inputs, device=device, requires_grad=True)
    with torch.enable_grad():
        out = model(inp_t)
        B_r = out[:, 1]
        B_z = out[:, 2]
        grad_Br = torch.autograd.grad(
            B_r, inp_t, grad_outputs=torch.ones_like(B_r),
            create_graph=False, retain_graph=True,
        )[0]
        grad_Bz = torch.autograd.grad(
            B_z, inp_t, grad_outputs=torch.ones_like(B_z),
            create_graph=False, retain_graph=False,
        )[0]

    scale = I if current_normalized else 1.0
    return (
        B_r.detach().cpu().numpy() * scale,
        B_z.detach().cpu().numpy() * scale,
        grad_Br[:, 0].detach().cpu().numpy() * scale,
        grad_Br[:, 1].detach().cpu().numpy() * scale,
        grad_Bz[:, 0].detach().cpu().numpy() * scale,
        grad_Bz[:, 1].detach().cpu().numpy() * scale,
    )


def predict_point_with_grad(model, r, z, I, N, R_mean, L, current_normalized, device):
    """Scalar fast path: single-point field + gradients as floats.

    One graph build per call — real-time consumers (Kit extension, warp
    solver) should call this once per step and derive everything (Bz
    logging, force) from the result.

    Returns:
        (Br, Bz, dBr_dr, dBr_dz, dBz_dr, dBz_dz) as floats.
    """
    Br, Bz, dBr_dr, dBr_dz, dBz_dr, dBz_dz = predict_field_with_grad(
        model, [r], [z], I, N, R_mean, L, current_normalized, device,
    )
    return (
        float(Br[0]), float(Bz[0]),
        float(dBr_dr[0]), float(dBr_dz[0]),
        float(dBz_dr[0]), float(dBz_dz[0]),
    )


class PINNField:
    """Convenience wrapper bundling a loaded model with its device and flags."""

    def __init__(self, model, device, metadata):
        self.model = model
        self.device = device
        self.metadata = metadata
        self.current_normalized = bool(model.current_normalized.item())
        self.derived_b = is_derived_b(model)

    def predict_field(self, r, z, I, N, R_mean, L):
        return predict_field(self.model, r, z, I, N, R_mean, L,
                             self.current_normalized, self.device)

    def predict_field_with_grad(self, r, z, I, N, R_mean, L):
        return predict_field_with_grad(self.model, r, z, I, N, R_mean, L,
                                       self.current_normalized, self.device)

    def predict_point_with_grad(self, r, z, I, N, R_mean, L):
        return predict_point_with_grad(self.model, r, z, I, N, R_mean, L,
                                       self.current_normalized, self.device)


def load_pinn(ckpt_path=None, device=None, min_step=None,
              require_current_normalized=False) -> PINNField:
    """Load a PINN checkpoint as a PINNField.

    Args:
        ckpt_path: checkpoint file (default: models/pinn_checkpoint/pinn_best.pt)
        device: torch device (default: cuda if available)
        min_step: if set, reject checkpoints trained for fewer steps
        require_current_normalized: if True, reject non-B/I checkpoints

    Returns:
        PINNField
    """
    if ckpt_path is None:
        ckpt_path = DEFAULT_CHECKPOINT
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, current_normalized, metadata = load_model_from_checkpoint(ckpt_path, device)

    if require_current_normalized and not current_normalized:
        raise RuntimeError(f"{ckpt_path} does not have current_normalized=True")
    if min_step is not None:
        step = metadata.get("step")
        if isinstance(step, int) and step < min_step:
            raise RuntimeError(f"checkpoint step={step} < {min_step} (not fully trained)")

    return PINNField(model, device, metadata)
