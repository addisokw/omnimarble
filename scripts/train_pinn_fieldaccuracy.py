"""Train the field-accuracy PINN baseline.

This is the v4 recipe: physics losses (curl, div, BC, symmetry) with weights
tuned for strict pointwise field accuracy. No gradient supervision.
Produces pinn_fieldaccuracy.pt.

Use train_pinn_designspace.py for the design-exploration model.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models" / "pinn_checkpoint"

MU_0_MM = 4 * math.pi * 1e-4

try:
    from physicsnemo.models.mlp import FullyConnected as NeMoFC
    from physicsnemo import Module as NeMoModule
    HAS_NEMO = True
    print("Using NVIDIA PhysicsNeMo FullyConnected model")
except ImportError:
    HAS_NEMO = False
    print("PhysicsNeMo not available, using pure PyTorch fallback")

# Import shared model class and physics losses from the main training script
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from train_pinn import (
    BFieldPINN,
    compute_pde_residual,
    compute_boundary_loss,
    compute_symmetry_loss,
    save_checkpoint,
    export_onnx,
)


def train(num_steps=300_000, batch_size=131072, lr=1e-3):
    """Train the field-accuracy PINN baseline (v4 recipe)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    inputs = np.load(DATA_DIR / "pinn_inputs.npy").astype(np.float32)
    Br = np.load(DATA_DIR / "pinn_Br.npy").astype(np.float32)
    Bz = np.load(DATA_DIR / "pinn_Bz.npy").astype(np.float32)

    print(f"Training data: {len(inputs)} samples")

    I_col = inputs[:, 2].copy()
    I_col = np.clip(I_col, 0.5, None)
    Br_norm = Br / I_col
    Bz_norm = Bz / I_col

    input_mean = inputs.mean(axis=0)
    input_std = inputs.std(axis=0)
    output_scale_val = max(np.abs(Br_norm).max(), np.abs(Bz_norm).max(), 1e-6)

    print(f"  B/I normalization: output_scale={output_scale_val:.6f} T/A")

    X = torch.tensor(inputs, device=device)
    Br_t = torch.tensor(Br_norm, device=device)
    Bz_t = torch.tensor(Bz_norm, device=device)

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

    ramp_steps = 20_000

    # v4 loss weights: balanced physics + self-consistency gradient regularizer
    w_curl = 1.0
    w_div = 1.0
    w_bc = 0.1
    w_sym = 0.5
    w_grad = 0.1  # self-consistency smoothness regularizer (original v4 had this)

    for step in range(num_steps):
        ramp = min(1.0, 0.1 + 0.9 * step / ramp_steps)

        idx = torch.randint(0, N, (batch_size,), device=device)
        x_batch = X[idx]
        br_batch = Br_t[idx]
        bz_batch = Bz_t[idx]

        optimizer.zero_grad()

        out = model(x_batch)

        data_loss = torch.mean(
            (out[:, 1] - br_batch) ** 2
            + (out[:, 2] - bz_batch) ** 2
        )

        pde_batch_size = 8192
        pde_idx = torch.randint(0, N, (pde_batch_size,), device=device)
        curl_loss, div_loss = compute_pde_residual(model, X[pde_idx], device)

        bc_loss = compute_boundary_loss(model, device, batch_size=4096)
        sym_loss = compute_symmetry_loss(model, device, batch_size=4096)

        # Self-consistency gradient regularizer: autograd dBz/dz should match
        # the model's own finite-difference. This smooths gradients near
        # boundaries without requiring analytical gradient targets.
        if step % 4 == 0:
            grad_idx = torch.randint(0, N, (8192,), device=device)
            x_grad = X[grad_idx].clone().requires_grad_(True)
            out_grad = model(x_grad)
            bz_grad = out_grad[:, 2]
            auto_grad = torch.autograd.grad(
                bz_grad, x_grad, grad_outputs=torch.ones_like(bz_grad),
                create_graph=True, retain_graph=True,
            )[0][:, 1]  # dBz/dz
            delta = 0.5
            x_p = X[grad_idx].clone(); x_p[:, 1] += delta
            x_m = X[grad_idx].clone(); x_m[:, 1] -= delta
            with torch.no_grad():
                fd_grad = (model(x_p)[:, 2] - model(x_m)[:, 2]) / (2 * delta)
            grad_loss = torch.mean((auto_grad - fd_grad) ** 2)
        else:
            grad_loss = torch.tensor(0.0, device=device)

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

            if loss.item() < best_loss:
                best_loss = loss.item()
                save_checkpoint(model, optimizer, step, loss.item(),
                                suffix="_fieldaccuracy")

    # Final save
    save_checkpoint(model, optimizer, num_steps, loss.item(), suffix="_fieldaccuracy")
    print(f"\nTraining complete. Best loss: {best_loss:.2e}")
    print(f"Checkpoint: {MODEL_DIR / 'pinn_best_fieldaccuracy.pt'}")


def main():
    print("=== PINN Training: Field-Accuracy Baseline ===\n")
    train()


if __name__ == "__main__":
    main()
