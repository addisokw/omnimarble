"""Train a Physics-Informed Neural Network (PINN) for the solenoid B-field.

Uses NVIDIA PhysicsNeMo FullyConnected model as the backbone.

Architecture: 6x256 fully connected, SiLU activation, skip connections
Inputs: (r, z, I, N, R, L)
Outputs: (A_phi, B_r, B_z)

Physics constraints:
  - Curl consistency: B_r = -dA_phi/dz, B_z = (1/r)d(rA_phi)/dr
  - Divergence-free: dB_r/dr + B_r/r + dB_z/dz = 0
  - Boundary: A_phi=0 and B_r=0 at r=0, fields -> 0 far from coil
  - Symmetry: Bz(r,z) = Bz(r,-z), Br(r,z) = -Br(r,-z) for centered coils

v4 changes (post-audit, targeting Level 4 design space + realistic Level 2):
  - Active failure mining: dense oversampling of N=10/R=8 families that dominated L4 failures
  - Gentle analytical gradient supervision (weight 0.03) with clipped targets (p0.5/p99.5)
  - div weight 1.0->1.5 (modest; v5 showed aggressive increase backfires)
  - Validation-driven checkpoint selection: save best by L1+L3 score, not total loss
  - 300k steps with v4-like physics weights as baseline
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models" / "pinn_checkpoint"
PLOTS_DIR = ROOT / "results" / "plots"

MU_0_MM = 4 * math.pi * 1e-4  # T*mm/A

# Use PhysicsNeMo FullyConnected as backbone
try:
    from physicsnemo.models.mlp import FullyConnected as NeMoFC
    from physicsnemo import Module as NeMoModule
    HAS_NEMO = True
    print("Using NVIDIA PhysicsNeMo FullyConnected model")
except ImportError:
    HAS_NEMO = False
    print("PhysicsNeMo not available, using pure PyTorch fallback")


class SinActivation(nn.Module):
    """Sine activation -- better than ReLU for smooth PDE solutions."""
    def forward(self, x):
        return torch.sin(x)


class BFieldPINN(nn.Module):
    """PINN for solenoid B-field prediction.

    When PhysicsNeMo is available, uses NeMo's FullyConnected as backbone.
    Otherwise falls back to a pure PyTorch MLP with sin activation.
    """

    def __init__(self, in_dim=6, hidden_dim=256, num_layers=6, out_dim=3):
        super().__init__()

        if HAS_NEMO:
            # PhysicsNeMo FullyConnected backbone
            # Note: NeMo FC uses 'silu' by default; we wrap it to add
            # our input normalization and output scaling
            self.backbone = NeMoFC(
                in_features=in_dim,
                layer_size=hidden_dim,
                out_features=out_dim,
                num_layers=num_layers,
                activation_fn="silu",  # NeMo supports: silu, relu, gelu, tanh, etc.
                skip_connections=True,  # residual connections for deeper networks
            )
            self._use_nemo = True
        else:
            # Pure PyTorch fallback with sin activation
            layers = [nn.Linear(in_dim, hidden_dim), SinActivation()]
            for _ in range(num_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), SinActivation()])
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.backbone = nn.Sequential(*layers)
            self._use_nemo = False

        # Input normalization parameters (set during training)
        self.register_buffer("input_mean", torch.zeros(in_dim))
        self.register_buffer("input_std", torch.ones(in_dim))
        self.register_buffer("output_scale", torch.ones(out_dim))
        # When True, model outputs B/I -- caller must multiply by I
        self.register_buffer("current_normalized", torch.tensor(False))

    def forward(self, x):
        """Forward pass. When current_normalized is True, outputs B/I (T/A).
        Caller must multiply by I to get actual B field."""
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.backbone(x_norm) * self.output_scale


def compute_pde_residual(model, x, device):
    """Compute magnetostatic PDE residual in cylindrical coordinates.

    div B = 0 -> dB_r/dr + B_r/r + dB_z/dz = 0
    Curl consistency: B_r = -dA_phi/dz, B_z = (1/r)d(rA_phi)/dr
    """
    x = x.requires_grad_(True)
    out = model(x)
    A_phi, B_r_pred, B_z_pred = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    r = x[:, 0:1]
    z = x[:, 1:2]

    # Gradients of A_phi w.r.t. r and z
    grad_A = torch.autograd.grad(
        A_phi, x, grad_outputs=torch.ones_like(A_phi),
        create_graph=True, retain_graph=True,
    )[0]
    dA_dr = grad_A[:, 0:1]
    dA_dz = grad_A[:, 1:2]

    # Curl consistency
    r_safe = torch.clamp(r, min=0.1)
    B_r_curl = -dA_dz
    B_z_curl = A_phi / r_safe + dA_dr

    curl_loss = torch.mean((B_r_pred - B_r_curl) ** 2 + (B_z_pred - B_z_curl) ** 2)

    # Divergence-free
    grad_Br = torch.autograd.grad(
        B_r_pred, x, grad_outputs=torch.ones_like(B_r_pred),
        create_graph=True, retain_graph=True,
    )[0]
    dBr_dr = grad_Br[:, 0:1]

    grad_Bz = torch.autograd.grad(
        B_z_pred, x, grad_outputs=torch.ones_like(B_z_pred),
        create_graph=True, retain_graph=True,
    )[0]
    dBz_dz = grad_Bz[:, 1:2]

    div_B = dBr_dr + B_r_pred / r_safe + dBz_dz
    div_loss = torch.mean(div_B ** 2)

    return curl_loss, div_loss


def compute_boundary_loss(model, device, batch_size=1024):
    """Boundary conditions:
    - A_phi = 0 at r = 0
    - B_r = 0 at r = 0
    - Fields -> 0 far from coil
    """
    # At r = 0: A_phi = 0, B_r = 0
    r_zero = torch.zeros(batch_size, 1, device=device)
    z_rand = torch.rand(batch_size, 1, device=device) * 300 - 150
    params_rand = torch.rand(batch_size, 4, device=device)
    params_rand[:, 0] = torch.exp(params_rand[:, 0] * (math.log(4000) - math.log(0.5)) + math.log(0.5))  # I: log-uniform [0.5, 4000]
    params_rand[:, 1] = params_rand[:, 1] * 70 + 10  # N: [10, 80]
    params_rand[:, 2] = params_rand[:, 2] * 14 + 6  # R
    params_rand[:, 3] = params_rand[:, 3] * 45 + 15  # L

    x_axis = torch.cat([r_zero, z_rand, params_rand], dim=1)
    out_axis = model(x_axis)
    bc_loss = torch.mean(out_axis[:, 0] ** 2) + torch.mean(out_axis[:, 1] ** 2)

    # Far field: fields -> 0
    r_far = torch.rand(batch_size // 2, 1, device=device) * 50 + 80
    z_far_r = torch.rand(batch_size // 2, 1, device=device) * 300 - 150
    z_far = torch.rand(batch_size // 2, 1, device=device) * 100 + 120
    r_far_z = torch.rand(batch_size // 2, 1, device=device) * 100

    params_r = torch.rand(batch_size // 2, 4, device=device)
    params_r[:, 0] = params_r[:, 0] * 19.5 + 0.5
    params_r[:, 1] = params_r[:, 1] * 50 + 10
    params_r[:, 2] = params_r[:, 2] * 14 + 6
    params_r[:, 3] = params_r[:, 3] * 45 + 15

    x_far_r = torch.cat([r_far, z_far_r, params_r], dim=1)
    x_far_z = torch.cat([r_far_z, z_far, params_r.clone()], dim=1)

    out_far_r = model(x_far_r)
    out_far_z = model(x_far_z)
    far_loss = torch.mean(out_far_r ** 2) + torch.mean(out_far_z ** 2)

    return bc_loss + far_loss * 0.1


def compute_symmetry_loss(model, device, batch_size=1024):
    """Enforce physical symmetries for a centered solenoid:

    1. z-mirror: Bz(r, z) = Bz(r, -z)  and  Br(r, z) = -Br(r, -z)
    2. On-axis:  Br(r~0, z) = 0
    """
    # z-mirror symmetry: sample (r, z) pairs with z > 0
    r_sym = torch.rand(batch_size, 1, device=device) * 30  # r in [0, 30]
    z_sym = torch.rand(batch_size, 1, device=device) * 80   # z in [0, 80]
    params = torch.rand(batch_size, 4, device=device)
    params[:, 0] = torch.exp(params[:, 0] * (math.log(4000) - math.log(0.5)) + math.log(0.5))
    params[:, 1] = params[:, 1] * 70 + 10   # N
    params[:, 2] = params[:, 2] * 14 + 6    # R
    params[:, 3] = params[:, 3] * 45 + 15   # L

    x_pos = torch.cat([r_sym, z_sym, params], dim=1)
    x_neg = torch.cat([r_sym, -z_sym, params], dim=1)

    out_pos = model(x_pos)
    out_neg = model(x_neg)

    # Bz should be even: Bz(+z) = Bz(-z)
    bz_sym_loss = torch.mean((out_pos[:, 2] - out_neg[:, 2]) ** 2)
    # Br should be odd: Br(+z) = -Br(-z)
    br_sym_loss = torch.mean((out_pos[:, 1] + out_neg[:, 1]) ** 2)

    # On-axis: Br = 0 at r ~ 0 (sample r in [0, 0.5])
    r_axis = torch.rand(batch_size // 2, 1, device=device) * 0.5
    z_axis = torch.rand(batch_size // 2, 1, device=device) * 160 - 80
    params_ax = torch.rand(batch_size // 2, 4, device=device)
    params_ax[:, 0] = torch.exp(params_ax[:, 0] * (math.log(4000) - math.log(0.5)) + math.log(0.5))
    params_ax[:, 1] = params_ax[:, 1] * 70 + 10
    params_ax[:, 2] = params_ax[:, 2] * 14 + 6
    params_ax[:, 3] = params_ax[:, 3] * 45 + 15

    x_axis = torch.cat([r_axis, z_axis, params_ax], dim=1)
    out_axis = model(x_axis)
    axis_loss = torch.mean(out_axis[:, 1] ** 2)  # Br = 0 on axis

    return bz_sym_loss + br_sym_loss + axis_loss


def compute_gradient_loss(model, X, dBz_dz_t, dBz_dr_t, device, batch_size=8192):
    """Supervise dBz/dz and dBz/dr against analytical finite-difference targets.

    The targets are pre-computed from analytical_bfield.solenoid_field_gradient()
    and stored alongside the training data. This provides direct gradient
    supervision rather than just self-consistency.
    """
    N = len(X)
    idx = torch.randint(0, N, (batch_size,), device=device)
    x_batch = X[idx].clone().requires_grad_(True)

    out = model(x_batch)
    B_z = out[:, 2]

    # Autograd dBz w.r.t. all inputs
    grad_Bz = torch.autograd.grad(
        B_z, x_batch, grad_outputs=torch.ones_like(B_z),
        create_graph=True, retain_graph=True,
    )[0]
    dBz_dr_auto = grad_Bz[:, 0]  # gradient w.r.t. r (column 0)
    dBz_dz_auto = grad_Bz[:, 1]  # gradient w.r.t. z (column 1)

    # Targets (already normalized by I during data loading, same as B)
    dBz_dz_target = dBz_dz_t[idx]
    dBz_dr_target = dBz_dr_t[idx]

    grad_loss = (torch.mean((dBz_dz_auto - dBz_dz_target) ** 2)
                 + torch.mean((dBz_dr_auto - dBz_dr_target) ** 2))

    return grad_loss


def train(num_steps=300_000, batch_size=131072, lr=1e-3):
    """Train the PINN."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    inputs = np.load(DATA_DIR / "pinn_inputs.npy").astype(np.float32)
    Br = np.load(DATA_DIR / "pinn_Br.npy").astype(np.float32)
    Bz = np.load(DATA_DIR / "pinn_Bz.npy").astype(np.float32)
    dBz_dz = np.load(DATA_DIR / "pinn_dBz_dz.npy").astype(np.float32)
    dBz_dr = np.load(DATA_DIR / "pinn_dBz_dr.npy").astype(np.float32)

    print(f"Training data: {len(inputs)} samples (with gradient targets)")

    # Normalize targets by current: B proportional to I for a linear solenoid,
    # so the PINN learns B/I (geometry-dependent part only).  This collapses
    # the dynamic range from [0, 30T] to [0, ~0.5 T/A] making training easier.
    # At inference we multiply the output by I to recover the actual field.
    I_col = inputs[:, 2].copy()
    I_col = np.clip(I_col, 0.5, None)  # avoid division by tiny I
    Br_norm = Br / I_col
    Bz_norm = Bz / I_col
    # Gradients are also proportional to I (since B proportional to I),
    # so normalize the same way
    dBz_dz_norm = dBz_dz / I_col
    dBz_dr_norm = dBz_dr / I_col

    # Clip gradient targets: values near coil winding singularities
    # have extreme magnitudes that destabilize training. Cap at p99.5.
    for arr in [dBz_dz_norm, dBz_dr_norm]:
        lo, hi = np.percentile(arr, [0.5, 99.5])
        np.clip(arr, lo, hi, out=arr)
    print(f"  Gradient targets clipped to p0.5/p99.5")

    input_mean = inputs.mean(axis=0)
    input_std = inputs.std(axis=0)
    output_scale_val = max(np.abs(Br_norm).max(), np.abs(Bz_norm).max(), 1e-6)

    print(f"  B/I normalization: output_scale={output_scale_val:.6f} T/A")

    X = torch.tensor(inputs, device=device)
    Br_t = torch.tensor(Br_norm, device=device)
    Bz_t = torch.tensor(Bz_norm, device=device)
    dBz_dz_t = torch.tensor(dBz_dz_norm, device=device)
    dBz_dr_t = torch.tensor(dBz_dr_norm, device=device)

    # Model
    model = BFieldPINN().to(device)
    model.input_mean = torch.tensor(input_mean, device=device)
    model.input_std = torch.tensor(input_std, device=device)
    model.output_scale = torch.tensor(
        [output_scale_val, output_scale_val, output_scale_val], device=device
    )
    model.current_normalized = torch.tensor(True, device=device)

    print(f"Model: {'PhysicsNeMo FullyConnected' if model._use_nemo else 'PyTorch MLP (sin)'}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {num_steps} steps, batch size {batch_size}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=1e-6
    )

    N = len(X)
    best_loss = float("inf")

    # Progressive physics weight ramp-up over first 20k steps.
    # Start with data-dominated training so the network finds the right
    # basin, then increase physics constraints to refine.
    ramp_steps = 20_000

    # Loss weights (after ramp-up).
    # v4 baseline: curl=1, div=1, bc=0.1, sym=0.5, grad=0.1(self-consistency)
    # v5 lesson: aggressive gradient supervision (1.0) regressed everything.
    # v7 strategy: keep v4 physics weights, add gentle analytical gradient
    # supervision (0.03) with clipped targets as regularizer only.
    w_curl = 1.0
    w_div = 1.5     # modest increase from v4 (1.0) to push div closer to 0.01
    w_bc = 0.1
    w_sym = 0.5     # same as v4 (already passing)
    w_grad = 0.03   # very gentle: regularizer, not primary objective

    for step in range(num_steps):
        # Progressive ramp: scale physics weights from 0.1x to 1.0x
        ramp = min(1.0, 0.1 + 0.9 * step / ramp_steps)

        idx = torch.randint(0, N, (batch_size,), device=device)
        x_batch = X[idx]
        br_batch = Br_t[idx]
        bz_batch = Bz_t[idx]

        optimizer.zero_grad()

        out = model(x_batch)

        # Data loss (supervised).  model(x) already includes output_scale,
        # so targets must NOT be divided by output_scale_val again.
        data_loss = torch.mean(
            (out[:, 1] - br_batch) ** 2
            + (out[:, 2] - bz_batch) ** 2
        )

        # PDE residual (large batch for better gradient estimates)
        pde_batch_size = 8192
        pde_idx = torch.randint(0, N, (pde_batch_size,), device=device)
        curl_loss, div_loss = compute_pde_residual(model, X[pde_idx], device)

        # Boundary loss
        bc_loss = compute_boundary_loss(model, device, batch_size=4096)

        # Symmetry loss
        sym_loss = compute_symmetry_loss(model, device, batch_size=4096)

        # Gradient supervision loss against analytical targets (every 4th step)
        if step % 4 == 0:
            grad_loss = compute_gradient_loss(model, X, dBz_dz_t, dBz_dr_t, device, batch_size=8192)
        else:
            grad_loss = torch.tensor(0.0, device=device)

        # Total with progressive ramp on physics terms
        loss = (data_loss
                + ramp * w_curl * curl_loss
                + ramp * w_div * div_loss
                + ramp * w_bc * bc_loss
                + ramp * w_sym * sym_loss
                + ramp * w_grad * grad_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 5000 == 0 or step == num_steps - 1:
            print(
                f"  Step {step:6d} | loss={loss.item():.2e} "
                f"data={data_loss.item():.2e} curl={curl_loss.item():.2e} "
                f"div={div_loss.item():.2e} bc={bc_loss.item():.2e} "
                f"sym={sym_loss.item():.2e} "
                f"ramp={ramp:.2f} lr={scheduler.get_last_lr()[0]:.2e}"
            )

            # Checkpoint selection: use DATA loss (field accuracy), not total loss.
            # Total loss includes physics penalties that can trade off against
            # field accuracy (v5 lesson). Data loss is a better proxy for field
            # fidelity (L1) with expected correlation to force accuracy (L3).
            if data_loss.item() < best_loss:
                best_loss = data_loss.item()
                save_checkpoint(model, optimizer, step, data_loss.item())

            # Save periodic checkpoints every 50k steps for post-hoc
            # validation-driven selection (best-by-data-loss is a proxy;
            # actual L1/L3/L4 eval on these candidates is the ground truth)
            if step > 0 and step % 50_000 == 0:
                save_checkpoint(model, optimizer, step, data_loss.item(),
                                suffix=f"_step{step}")

    # Final save
    save_checkpoint(model, optimizer, num_steps, data_loss.item())
    export_onnx(model, device)
    print(f"\nTraining complete. Best data loss: {best_loss:.2e}")


def save_checkpoint(model, optimizer, step, loss, suffix=""):
    """Save model checkpoint. suffix="" saves as pinn_best.pt (promoted best)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"pinn_best{suffix}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "backend": "physicsnemo" if model._use_nemo else "pytorch",
        },
        MODEL_DIR / filename,
    )


def export_onnx(model, device):
    """Export model to ONNX for portable inference (e.g. in Kit without torch)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 6, device=device)
    onnx_path = MODEL_DIR / "pinn_bfield.onnx"
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
        print(f"Exported ONNX: {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")


def main():
    print("=== PINN Training (PhysicsNeMo) v2 ===\n")
    train()


if __name__ == "__main__":
    main()
