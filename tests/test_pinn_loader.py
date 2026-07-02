"""Tests for the shared PINN loader and the derived-B (v8) structural guarantees.

Skipped entirely if torch/physicsnemo are not installed. Checkpoint-dependent
tests skip if models/pinn_checkpoint/pinn_best.pt is absent.
"""

import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("physicsnemo")

import pinn_loader
from pinn_loader import (
    BFieldPINNDerived,
    is_derived_b,
    load_model_from_checkpoint,
    load_pinn,
    predict_field,
    predict_field_with_grad,
)

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "models" / "pinn_checkpoint" / "pinn_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default coil for inference tests
N, R_MEAN, L_COIL = 30, 15.0, 30.0

needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="pinn_best.pt not present"
)


def _random_points(n=64, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.5, 30.0, n).astype(np.float32)
    z = rng.uniform(-40.0, 40.0, n).astype(np.float32)
    return r, z


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


@needs_checkpoint
def test_load_production_checkpoint():
    model, cn, meta = load_model_from_checkpoint(CHECKPOINT, DEVICE)
    assert cn is True, "production checkpoint must be current-normalized (B/I)"
    assert meta["current_normalized"] is True
    assert isinstance(meta["step"], int)


@needs_checkpoint
def test_load_pinn_guards():
    field = load_pinn(device=DEVICE, min_step=200000, require_current_normalized=True)
    assert field.current_normalized
    # Shapes
    r, z = _random_points(16)
    Br, Bz = field.predict_field(r, z, 100.0, N, R_MEAN, L_COIL)
    assert Br.shape == (16,) and Bz.shape == (16,)
    out = field.predict_field_with_grad(r, z, 100.0, N, R_MEAN, L_COIL)
    assert len(out) == 6 and all(a.shape == (16,) for a in out)


@needs_checkpoint
def test_current_normalization_linearity():
    """B/I model: field at 200A is ~2x the field at 100A.

    Not exact: I is also an input feature, so the learned B/I retains a
    weak residual I-dependence (~1%). The 2x scaling itself comes from the
    loader multiplying by I.
    """
    field = load_pinn(device=DEVICE)
    r, z = _random_points(8)
    _, Bz_100 = field.predict_field(r, z, 100.0, N, R_MEAN, L_COIL)
    _, Bz_200 = field.predict_field(r, z, 200.0, N, R_MEAN, L_COIL)
    np.testing.assert_allclose(Bz_200, 2.0 * Bz_100, rtol=0.05)


@needs_checkpoint
def test_point_matches_batch():
    field = load_pinn(device=DEVICE)
    r, z = 3.0, -10.0
    point = field.predict_point_with_grad(r, z, 318.0, N, R_MEAN, L_COIL)
    batch = field.predict_field_with_grad([r], [z], 318.0, N, R_MEAN, L_COIL)
    for p, b in zip(point, batch):
        assert p == pytest.approx(float(b[0]), rel=1e-6)


# ---------------------------------------------------------------------------
# Derived-B (v8) structural guarantees — tested on an untrained model, since
# the properties hold by construction regardless of weights
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def derived_model():
    torch.manual_seed(0)
    model = BFieldPINNDerived().to(DEVICE)
    model.eval()
    return model


def test_derived_flag(derived_model):
    assert is_derived_b(derived_model)


def test_derived_div_b_exact(derived_model):
    """div(B) = dBr/dr + Br/r + dBz/dz must vanish to float32 roundoff."""
    r, z = _random_points(128, seed=1)
    Br, Bz, dBr_dr, dBr_dz, dBz_dr, dBz_dz = predict_field_with_grad(
        derived_model, r, z, 1.0, N, R_MEAN, L_COIL, False, DEVICE,
    )
    div = dBr_dr + Br / r + dBz_dz
    # Normalize by the gradient magnitude scale (the natural scale of the terms)
    scale = np.abs(dBr_dr) + np.abs(Br / r) + np.abs(dBz_dz) + 1e-12
    assert np.max(np.abs(div) / scale) < 1e-4


def test_derived_br_zero_on_axis(derived_model):
    """Br(0, z) = 0 exactly (A = r*f ansatz)."""
    z = np.linspace(-40, 40, 32).astype(np.float32)
    r = np.zeros_like(z)
    Br, _ = predict_field(derived_model, r, z, 1.0, N, R_MEAN, L_COIL, False, DEVICE)
    assert np.max(np.abs(Br)) == 0.0


def test_derived_a_phi_zero_on_axis(derived_model):
    z = np.linspace(-40, 40, 8).astype(np.float32)
    r = np.zeros_like(z)
    x = torch.tensor(
        np.column_stack([r, z, np.full(8, 1.0), np.full(8, N),
                         np.full(8, R_MEAN), np.full(8, L_COIL)]).astype(np.float32),
        device=DEVICE,
    )
    out = derived_model(x).detach().cpu().numpy()
    assert np.max(np.abs(out[:, 0])) == 0.0  # A_phi column


def test_derived_curl_consistency(derived_model):
    """B_r == -dA/dz and B_z == (1/r) d(rA)/dr via finite differences of A."""
    r, z = _random_points(32, seed=2)
    eps = 1e-2

    def a_phi(rv, zv):
        x = torch.tensor(
            np.column_stack([rv, zv, np.full(len(rv), 1.0), np.full(len(rv), N),
                             np.full(len(rv), R_MEAN), np.full(len(rv), L_COIL)]
                            ).astype(np.float32),
            device=DEVICE,
        )
        return derived_model(x).detach().cpu().numpy()[:, 0]

    Br, Bz = predict_field(derived_model, r, z, 1.0, N, R_MEAN, L_COIL, False, DEVICE)

    dA_dz = (a_phi(r, z + eps) - a_phi(r, z - eps)) / (2 * eps)
    d_rA_dr = ((r + eps) * a_phi(r + eps, z) - (r - eps) * a_phi(r - eps, z)) / (2 * eps)

    # Tolerance is limited by central-difference truncation (O(eps^2) * f''')
    # on a random-init network in float32, not by the model identity itself
    np.testing.assert_allclose(Br, -dA_dz, rtol=5e-2, atol=1e-3)
    np.testing.assert_allclose(Bz, d_rA_dr / r, rtol=5e-2, atol=1e-3)


def test_derived_second_derivative_fd_crosscheck(derived_model):
    """Autograd dBz/dz (a second derivative of f) matches finite differences of Bz."""
    r, z = _random_points(32, seed=3)
    eps = 1e-2

    _, _, _, _, _, dBz_dz = predict_field_with_grad(
        derived_model, r, z, 1.0, N, R_MEAN, L_COIL, False, DEVICE,
    )
    _, Bz_p = predict_field(derived_model, r, z + eps, 1.0, N, R_MEAN, L_COIL, False, DEVICE)
    _, Bz_m = predict_field(derived_model, r, z - eps, 1.0, N, R_MEAN, L_COIL, False, DEVICE)
    fd = (Bz_p - Bz_m) / (2 * eps)

    np.testing.assert_allclose(dBz_dz, fd, rtol=2e-2, atol=1e-4)


def test_derived_state_dict_roundtrip(tmp_path, derived_model):
    """A saved derived checkpoint is detected and reloaded as derived."""
    ckpt = {
        "step": 300000,
        "model_state_dict": derived_model.state_dict(),
        "loss": 1e-8,
        "backend": "physicsnemo",
        "model_version": 8,
        "derived_b": True,
    }
    path = tmp_path / "pinn_v8_test.pt"
    torch.save(ckpt, path)

    model2, cn, meta = load_model_from_checkpoint(path, DEVICE)
    assert meta["derived_b"] is True
    assert is_derived_b(model2)

    r, z = _random_points(8, seed=4)
    _, Bz_a = predict_field(derived_model, r, z, 1.0, N, R_MEAN, L_COIL, False, DEVICE)
    _, Bz_b = predict_field(model2, r, z, 1.0, N, R_MEAN, L_COIL, False, DEVICE)
    np.testing.assert_allclose(Bz_a, Bz_b, rtol=1e-6)


@needs_checkpoint
def test_production_div_b_sanity():
    """The production checkpoint's div(B) residual matches its architecture:
    ~exact for derived-B (v8+), a finite training residual for legacy."""
    model, cn, meta = load_model_from_checkpoint(CHECKPOINT, DEVICE)
    r, z = _random_points(128, seed=1)

    Br, Bz, dBr_dr, _, _, dBz_dz = predict_field_with_grad(
        model, r, z, 100.0, N, R_MEAN, L_COIL, cn, DEVICE,
    )
    div = dBr_dr + Br / r + dBz_dz
    scale = np.abs(dBr_dr) + np.abs(Br / r) + np.abs(dBz_dz) + 1e-12
    norm_div = np.mean(np.abs(div) / scale)

    if meta["derived_b"]:
        assert norm_div < 1e-4, "derived-B model must satisfy div(B)=0 structurally"
    else:
        assert norm_div > 1e-4, "legacy model has a finite div residual by nature"
