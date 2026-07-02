#!/usr/bin/env bash
# Remote PINN training — syncs files to a remote Windows GPU machine, runs
# data gen + training, and pulls the checkpoints back.
#
# Usage:
#   bash scripts/remote_train.sh
#
# Prerequisites:
#   - SSH key auth to REMOTE_HOST (note: ICMP is firewalled on the remote —
#     preflight with `ssh <host> "echo ok"`, not ping)
#   - Remote machine has Python + uv installed
#   - Remote shell is cmd.exe (Windows OpenSSH default)
#
# The trained checkpoint is pulled back as pinn_v8_candidate.pt and the
# periodic step checkpoints land in models/pinn_checkpoint/candidates/v8/.
# The local production pinn_best.pt is NOT overwritten — promotion happens
# only after the adoption gate (evaluate_pinn + evaluate_candidates +
# v7_optimization_diagnostic) passes.

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────────
REMOTE_HOST="10.10.1.187"
REMOTE_USER="ARL-S"
REMOTE_DIR="omnimarble_train"  # relative to remote user's home directory
# ────────────────────────────────────────────────────────────────────────────

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

echo "=== Remote PINN Training (v8 derived-B) ==="
echo "  Local:  ${LOCAL_ROOT}"
echo "  Remote: ${REMOTE}:~/${REMOTE_DIR}"
echo ""

# ── Step 0: Preflight ───────────────────────────────────────────────────────
ssh -o ConnectTimeout=10 "${REMOTE}" "echo remote-ok" || {
    echo "ERROR: remote ${REMOTE} unreachable via ssh"; exit 1;
}

# ── Step 1: Create remote directory structure ───────────────────────────────
echo "[1/5] Creating remote directories..."
ssh "${REMOTE}" "powershell -Command \"New-Item -ItemType Directory -Force -Path ${REMOTE_DIR}/scripts, ${REMOTE_DIR}/config, ${REMOTE_DIR}/data, ${REMOTE_DIR}/models/pinn_checkpoint, ${REMOTE_DIR}/results/plots/pinn_validation | Out-Null; echo 'dirs ready'\""

# ── Step 2: Sync source files ───────────────────────────────────────────────
echo "[2/5] Syncing source files..."

scp "${LOCAL_ROOT}/scripts/analytical_bfield.py" \
    "${LOCAL_ROOT}/scripts/generate_training_data.py" \
    "${LOCAL_ROOT}/scripts/pinn_loader.py" \
    "${LOCAL_ROOT}/scripts/train_pinn.py" \
    "${LOCAL_ROOT}/scripts/evaluate_pinn.py" \
    "${REMOTE}:${REMOTE_DIR}/scripts/"

scp "${LOCAL_ROOT}/config/coil_params.json" \
    "${REMOTE}:${REMOTE_DIR}/config/"

scp "${LOCAL_ROOT}/pyproject.toml" \
    "${REMOTE}:${REMOTE_DIR}/"

if [[ -f "${LOCAL_ROOT}/uv.lock" ]]; then
    scp "${LOCAL_ROOT}/uv.lock" "${REMOTE}:${REMOTE_DIR}/"
fi

echo "  Files synced."

# ── Step 3: Generate training data on remote ────────────────────────────────
echo "[3/5] Generating training data on remote (this may take a while)..."
ssh "${REMOTE}" "cd ${REMOTE_DIR} && uv run python scripts/generate_training_data.py"

# ── Step 4: Train ───────────────────────────────────────────────────────────
echo "[4/5] Training v8 PINN on remote GPU (300k steps)..."
echo "  (This will take a while — watch progress below)"
echo "  (If the derivative graph OOMs at batch 131072, rerun with --batch-size 65536)"
echo ""
ssh "${REMOTE}" "cd ${REMOTE_DIR} && uv run python scripts/train_pinn.py"

# ── Step 5: Pull checkpoints back ───────────────────────────────────────────
echo ""
echo "[5/5] Pulling checkpoints back to local machine..."
mkdir -p "${LOCAL_ROOT}/models/pinn_checkpoint/candidates/v8"

scp "${REMOTE}:${REMOTE_DIR}/models/pinn_checkpoint/pinn_best.pt" \
    "${LOCAL_ROOT}/models/pinn_checkpoint/pinn_v8_candidate.pt"

scp "${REMOTE}:${REMOTE_DIR}/models/pinn_checkpoint/pinn_best_step*.pt" \
    "${LOCAL_ROOT}/models/pinn_checkpoint/candidates/v8/" 2>/dev/null || true

echo ""
echo "=== Done ==="
echo "Candidate saved to: models/pinn_checkpoint/pinn_v8_candidate.pt"
echo "Step checkpoints:   models/pinn_checkpoint/candidates/v8/"
echo ""
echo "Next: run the adoption-gate evaluation locally:"
echo "  uv run python scripts/evaluate_pinn.py --ckpt models/pinn_checkpoint/pinn_v8_candidate.pt"
echo "  uv run python scripts/evaluate_candidates.py --candidates-dir models/pinn_checkpoint/candidates/v8 --final models/pinn_checkpoint/pinn_v8_candidate.pt"
echo "  uv run python scripts/v7_optimization_diagnostic.py --ckpt models/pinn_checkpoint/pinn_v8_candidate.pt"
