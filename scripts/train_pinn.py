"""Train a Physics-Informed Neural Network (PINN) for the solenoid B-field.

Uses NVIDIA PhysicsNeMo FullyConnected as the backbone.

v8 architecture ("physics by construction"):
  The network outputs a single scalar f(r, z, I, N, R, L); the vector
  potential is A_phi = r * f and the field is derived inside forward()
  via autograd:
      B_r = -dA/dz         = -r * df/dz
      B_z = (1/r) d(rA)/dr = 2f + r * df/dr
  This makes div(B)=0, curl consistency, A_phi(0,z)=0, and B_r(0,z)=0
  EXACT identities of the architecture (see pinn_loader.BFieldPINNDerived),
  so the v3-v7 curl/div/on-axis-boundary losses are removed entirely.

Remaining losses:
  - Data: MSE on derived (B_r/I, B_z/I) vs analytical Biot-Savart targets.
    Note the data loss itself now backprops through an autograd derivative
    (create_graph=True in forward), the main per-step cost increase vs v7.
  - Far-field decay (weight 0.1): fields -> 0 far from the coil.
  - z-mirror symmetry (weight 0.5): Bz even, Br odd in z for centered coils.
  - Progressive ramp on physics weights over the first 20k steps (kept).

Removed vs v7: curl (exact), div (exact), on-axis BCs (exact), analytical
gradient supervision (would be a third-order term through the derived
field; v5 showed aggressive gradient supervision regresses accuracy, and
v7 showed weight 0.03 was marginal — re-add only if L2 fails).

Checkpoint selection: best DATA loss (v5 lesson: total loss trades off
against field accuracy), plus periodic saves every 50k steps for post-hoc
validation-driven selection via evaluate_candidates.py.

ONNX export was dropped in v8: autograd inside forward() is not
ONNX-traceable, and nothing consumed the export.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models" / "pinn_checkpoint"

sys.path.insert(0, str(ROOT / "scripts"))

from pinn_loader import HAS_NEMO, BFieldPINNDerived

# Canonical model class (architecture lives in pinn_loader — single source
# of truth shared with all inference consumers)
BFieldPINN = BFieldPINNDerived

MU_0_MM = 4 * math.pi * 1e-4  # T*mm/A

MODEL_VERSION = 8

if HAS_NEMO:
    print("Using NVIDIA PhysicsNeMo FullyConnected model")
else:
    print("PhysicsNeMo not available, using pure PyTorch fallback")


def _sample_coil_params(batch_size, device, log_current=True):
    """Random (I, N, R_mean, L) columns over the training ranges."""
    params = torch.rand(batch_size, 4, device=device)
    if log_current:
        params[:, 0] = torch.exp(
            params[:, 0] * (math.log(4000) - math.log(0.5)) + math.log(0.5)
        )  # I: log-uniform [0.5, 4000]
    else:
        params[:, 0] = params[:, 0] * 19.5 + 0.5
    params[:, 1] = params[:, 1] * 70 + 10   # N: [10, 80]
    params[:, 2] = params[:, 2] * 14 + 6    # R_mean: [6, 20]
    params[:, 3] = params[:, 3] * 45 + 15   # L: [15, 60]
    return params


def compute_farfield_loss(model, device, batch_size=1024):
    """Far-field decay: fields -> 0 far from the coil.

    The r=0 boundary conditions (A_phi=0, B_r=0) of the v3-v7 loss are
    exact by construction in the derived-B model and are no longer needed.
    """
    r_far = torch.rand(batch_size // 2, 1, device=device) * 50 + 80
    z_far_r = torch.rand(batch_size // 2, 1, device=device) * 300 - 150
    z_far = torch.rand(batch_size // 2, 1, device=device) * 100 + 120
    r_far_z = torch.rand(batch_size // 2, 1, device=device) * 100

    params = _sample_coil_params(batch_size // 2, device, log_current=False)

    x_far_r = torch.cat([r_far, z_far_r, params], dim=1)
    x_far_z = torch.cat([r_far_z, z_far, params.clone()], dim=1)

    out_far_r = model(x_far_r)
    out_far_z = model(x_far_z)
    return (torch.mean(out_far_r ** 2) + torch.mean(out_far_z ** 2)) * 0.1


def compute_symmetry_loss(model, device, batch_size=1024):
    """z-mirror symmetry for centered coils: Bz even, Br odd in z.

    The on-axis Br=0 term of the v3-v7 loss is exact by construction.
    """
    r_sym = torch.rand(batch_size, 1, device=device) * 30   # r in [0, 30]
    z_sym = torch.rand(batch_size, 1, device=device) * 80   # z in [0, 80]
    params = _sample_coil_params(batch_size, device)

    x_pos = torch.cat([r_sym, z_sym, params], dim=1)
    x_neg = torch.cat([r_sym, -z_sym, params], dim=1)

    out_pos = model(x_pos)
    out_neg = model(x_neg)

    # Bz even: Bz(+z) = Bz(-z); Br odd: Br(+z) = -Br(-z)
    bz_sym_loss = torch.mean((out_pos[:, 2] - out_neg[:, 2]) ** 2)
    br_sym_loss = torch.mean((out_pos[:, 1] + out_neg[:, 1]) ** 2)
    return bz_sym_loss + br_sym_loss


def train(num_steps=300_000, batch_size=131072, lr=1e-3, out_name="pinn_best"):
    """Train the v8 derived-B PINN."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load data (gradient target files of the v5-v7 experiments are no
    # longer used — gradient supervision is removed in v8)
    inputs = np.load(DATA_DIR / "pinn_inputs.npy").astype(np.float32)
    Br = np.load(DATA_DIR / "pinn_Br.npy").astype(np.float32)
    Bz = np.load(DATA_DIR / "pinn_Bz.npy").astype(np.float32)

    print(f"Training data: {len(inputs)} samples")

    # Normalize targets by current: B proportional to I for a linear solenoid,
    # so the PINN learns B/I (geometry-dependent part only).  This collapses
    # the dynamic range from [0, 30T] to [0, ~0.5 T/A] making training easier.
    # At inference we multiply the output by I to recover the actual field.
    I_col = inputs[:, 2].copy()
    I_col = np.clip(I_col, 0.5, None)  # avoid division by tiny I
    Br_norm = Br / I_col
    Bz_norm = Bz / I_col

    input_mean = inputs.mean(axis=0)
    input_std = inputs.std(axis=0)
    # Scale on the network's scalar head f. Since B_z = 2f + r*f_r, f is of
    # order Bz/2 near the axis — max|B/I| is the right order of magnitude
    # (same role as the v3-v7 output_scale).
    a_scale_val = max(np.abs(Br_norm).max(), np.abs(Bz_norm).max(), 1e-6)

    print(f"  B/I normalization: a_scale={a_scale_val:.6f} T/A")

    X = torch.tensor(inputs, device=device)
    Br_t = torch.tensor(Br_norm, device=device)
    Bz_t = torch.tensor(Bz_norm, device=device)

    # Model
    model = BFieldPINN().to(device)
    model.input_mean = torch.tensor(input_mean, device=device)
    model.input_std = torch.tensor(input_std, device=device)
    model.a_scale = torch.tensor(float(a_scale_val), device=device)
    model.current_normalized = torch.tensor(True, device=device)

    print(f"Model: {'PhysicsNeMo FullyConnected' if model._use_nemo else 'PyTorch MLP (sin)'} "
          f"(v{MODEL_VERSION}, derived-B)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {num_steps} steps, batch size {batch_size}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=1e-6
    )

    N = len(X)
    best_loss = float("inf")

    # Progressive physics weight ramp-up over first 20k steps.
    ramp_steps = 20_000

    # Loss weights (after ramp-up). curl/div/on-axis BC removed: exact by
    # construction. Gradient supervision removed: third-order through the
    # derived field, and marginal in v7.
    w_bc = 0.1   # far-field decay
    w_sym = 0.5  # z-mirror symmetry

    for step in range(num_steps):
        ramp = min(1.0, 0.1 + 0.9 * step / ramp_steps)

        idx = torch.randint(0, N, (batch_size,), device=device)
        x_batch = X[idx]
        br_batch = Br_t[idx]
        bz_batch = Bz_t[idx]

        optimizer.zero_grad()

        # forward() derives (Br, Bz) from A_phi with create_graph=True, so
        # this data loss backprops through the derivative graph.
        out = model(x_batch)
        data_loss = torch.mean(
            (out[:, 1] - br_batch) ** 2
            + (out[:, 2] - bz_batch) ** 2
        )

        bc_loss = compute_farfield_loss(model, device, batch_size=4096)
        sym_loss = compute_symmetry_loss(model, device, batch_size=4096)

        loss = (data_loss
                + ramp * w_bc * bc_loss
                + ramp * w_sym * sym_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 5000 == 0 or step == num_steps - 1:
            print(
                f"  Step {step:6d} | loss={loss.item():.2e} "
                f"data={data_loss.item():.2e} bc={bc_loss.item():.2e} "
                f"sym={sym_loss.item():.2e} "
                f"ramp={ramp:.2f} lr={scheduler.get_last_lr()[0]:.2e}"
            )

            # Checkpoint selection: use DATA loss (field accuracy), not total
            # loss (v5 lesson: physics penalties trade off against accuracy).
            if data_loss.item() < best_loss:
                best_loss = data_loss.item()
                save_checkpoint(model, optimizer, step, data_loss.item(),
                                out_name=out_name)

            # Periodic checkpoints every 50k steps for post-hoc
            # validation-driven selection (evaluate_candidates.py)
            if step > 0 and step % 50_000 == 0:
                save_checkpoint(model, optimizer, step, data_loss.item(),
                                suffix=f"_step{step}", out_name=out_name)

    # Final save
    save_checkpoint(model, optimizer, num_steps, data_loss.item(), out_name=out_name)
    print(f"\nTraining complete. Best data loss: {best_loss:.2e}")


def save_checkpoint(model, optimizer, step, loss, suffix="", out_name="pinn_best"):
    """Save model checkpoint. suffix="" saves as <out_name>.pt (promoted best)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{out_name}{suffix}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "backend": "physicsnemo" if model._use_nemo else "pytorch",
            "model_version": MODEL_VERSION,
            "derived_b": True,
        },
        MODEL_DIR / filename,
    )


def main():
    parser = argparse.ArgumentParser(description="Train the v8 derived-B PINN")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=131072,
                        help="Data batch size (halve if the derivative graph OOMs)")
    parser.add_argument("--out-name", default="pinn_best",
                        help="Checkpoint base name; use e.g. pinn_smoke for "
                             "test runs so the production checkpoint is not overwritten")
    args = parser.parse_args()

    print(f"=== PINN Training v{MODEL_VERSION} (derived-B, PhysicsNeMo) ===\n")
    train(num_steps=args.steps, batch_size=args.batch_size, out_name=args.out_name)


if __name__ == "__main__":
    main()
