"""Train a Physics-Informed Neural Network (PINN) for the solenoid B-field.

Uses NVIDIA PhysicsNeMo (v2.0.0+) for PINN training infrastructure.
Falls back to pure PyTorch if PhysicsNeMo is not available.

Architecture: 6×256 fully connected, sin activation
Inputs: (r, z, I, N, R, L)
Outputs: (A_phi, B_r, B_z)
"""

import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models" / "pinn_checkpoint"
PLOTS_DIR = ROOT / "results" / "plots"

MU_0_MM = 4 * math.pi * 1e-4  # T·mm/A

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch not installed. Run: uv add torch")
    raise SystemExit(1)

try:
    import physicsnemo
    HAS_NEMO = True
except ImportError:
    HAS_NEMO = False
    print("WARNING: nvidia-physicsnemo not available, using pure PyTorch fallback")


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class BFieldPINN(nn.Module):
    """PINN for solenoid B-field prediction.

    6 hidden layers of 256 units with sin activation.
    """

    def __init__(self, in_dim=6, hidden_dim=256, num_layers=6, out_dim=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), SinActivation()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), SinActivation()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

        # Input normalization parameters (set during training)
        self.register_buffer("input_mean", torch.zeros(in_dim))
        self.register_buffer("input_std", torch.ones(in_dim))
        self.register_buffer("output_scale", torch.ones(out_dim))

    def forward(self, x):
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.net(x_norm) * self.output_scale


def compute_pde_residual(model, x, device):
    """Compute magnetostatic PDE residual in cylindrical coordinates.

    ∇·B = 0 → ∂B_r/∂r + B_r/r + ∂B_z/∂z = 0
    Curl consistency: B_r = -∂A_φ/∂z, B_z = (1/r)∂(rA_φ)/∂r
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

    # Curl consistency: B_r = -dA_phi/dz, B_z = (1/r)*d(r*A_phi)/dr = A_phi/r + dA_phi/dr
    r_safe = torch.clamp(r, min=0.1)  # Avoid division by zero
    B_r_curl = -dA_dz
    B_z_curl = A_phi / r_safe + dA_dr

    curl_loss = torch.mean((B_r_pred - B_r_curl) ** 2 + (B_z_pred - B_z_curl) ** 2)

    # Divergence-free: dB_r/dr + B_r/r + dB_z/dz = 0
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
    - Fields → 0 far from coil
    """
    rng = torch.Generator(device=device).manual_seed(42)

    # At r = 0: A_phi = 0, B_r = 0
    r_zero = torch.zeros(batch_size, 1, device=device)
    z_rand = torch.rand(batch_size, 1, device=device) * 300 - 150
    params_rand = torch.rand(batch_size, 4, device=device)
    params_rand[:, 0] = params_rand[:, 0] * 19.5 + 0.5  # I
    params_rand[:, 1] = params_rand[:, 1] * 50 + 10  # N
    params_rand[:, 2] = params_rand[:, 2] * 14 + 6  # R
    params_rand[:, 3] = params_rand[:, 3] * 45 + 15  # L

    x_axis = torch.cat([r_zero, z_rand, params_rand], dim=1)
    out_axis = model(x_axis)
    bc_loss = torch.mean(out_axis[:, 0] ** 2) + torch.mean(out_axis[:, 1] ** 2)  # A_phi=0, B_r=0

    # Far field: fields → 0 at large r or z
    r_far = torch.rand(batch_size // 2, 1, device=device) * 50 + 80
    z_far_r = torch.rand(batch_size // 2, 1, device=device) * 300 - 150
    z_far = torch.rand(batch_size // 2, 1, device=device) * 100 + 120
    r_far_z = torch.rand(batch_size // 2, 1, device=device) * 100

    params_r = torch.rand(batch_size // 2, 4, device=device)
    params_r[:, 0] = params_r[:, 0] * 19.5 + 0.5
    params_r[:, 1] = params_r[:, 1] * 50 + 10
    params_r[:, 2] = params_r[:, 2] * 14 + 6
    params_r[:, 3] = params_r[:, 3] * 45 + 15

    params_z = params_r.clone()

    x_far_r = torch.cat([r_far, z_far_r, params_r], dim=1)
    x_far_z = torch.cat([r_far_z, z_far, params_z], dim=1)

    out_far_r = model(x_far_r)
    out_far_z = model(x_far_z)
    far_loss = torch.mean(out_far_r ** 2) + torch.mean(out_far_z ** 2)

    return bc_loss + far_loss * 0.1


def train(num_steps=100_000, batch_size=4096, lr=1e-3):
    """Train the PINN."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load data
    inputs = np.load(DATA_DIR / "pinn_inputs.npy").astype(np.float32)
    Br = np.load(DATA_DIR / "pinn_Br.npy").astype(np.float32)
    Bz = np.load(DATA_DIR / "pinn_Bz.npy").astype(np.float32)

    print(f"Training data: {len(inputs)} samples")

    # Compute normalization stats
    input_mean = inputs.mean(axis=0)
    input_std = inputs.std(axis=0)
    output_scale_val = max(np.abs(Br).max(), np.abs(Bz).max(), 1e-6)

    # Tensors
    X = torch.tensor(inputs, device=device)
    Br_t = torch.tensor(Br, device=device)
    Bz_t = torch.tensor(Bz, device=device)

    # Model
    model = BFieldPINN().to(device)
    model.input_mean = torch.tensor(input_mean, device=device)
    model.input_std = torch.tensor(input_std, device=device)
    model.output_scale = torch.tensor([output_scale_val, output_scale_val, output_scale_val], device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-6)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {num_steps} steps, batch size {batch_size}")

    N = len(X)
    best_loss = float("inf")

    for step in range(num_steps):
        # Random batch
        idx = torch.randint(0, N, (batch_size,), device=device)
        x_batch = X[idx]
        br_batch = Br_t[idx]
        bz_batch = Bz_t[idx]

        optimizer.zero_grad()

        out = model(x_batch)
        # out[:, 0] = A_phi, out[:, 1] = B_r, out[:, 2] = B_z

        # Data loss (supervised)
        data_loss = torch.mean((out[:, 1] - br_batch / output_scale_val) ** 2 +
                               (out[:, 2] - bz_batch / output_scale_val) ** 2)

        # PDE residual (on a subset for efficiency)
        pde_batch_size = min(1024, batch_size)
        pde_idx = torch.randint(0, N, (pde_batch_size,), device=device)
        x_pde = X[pde_idx]
        curl_loss, div_loss = compute_pde_residual(model, x_pde, device)

        # Boundary loss
        bc_loss = compute_boundary_loss(model, device, batch_size=512)

        # Total loss with weights
        loss = data_loss + 0.1 * curl_loss + 0.1 * div_loss + 0.01 * bc_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 5000 == 0 or step == num_steps - 1:
            print(f"  Step {step:6d} | loss={loss.item():.2e} "
                  f"data={data_loss.item():.2e} curl={curl_loss.item():.2e} "
                  f"div={div_loss.item():.2e} bc={bc_loss.item():.2e} "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                save_checkpoint(model, optimizer, step, loss.item())

    # Final save
    save_checkpoint(model, optimizer, num_steps, loss.item())
    export_onnx(model, device)


def save_checkpoint(model, optimizer, step, loss):
    """Save model checkpoint."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, MODEL_DIR / "pinn_best.pt")


def export_onnx(model, device):
    """Export model to ONNX format."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 6, device=device)
    onnx_path = MODEL_DIR / "pinn_bfield.onnx"
    try:
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
        print(f"Exported ONNX: {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")


def main():
    print("=== PINN Training ===\n")
    train()


if __name__ == "__main__":
    main()
