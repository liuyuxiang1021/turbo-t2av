#!/usr/bin/env bash
# Default TurboT2AV recipe: DCM 500 warmup -> SCM to 1000 -> full rCM.

set -euo pipefail
export PYTHONUNBUFFERED=1

usage() {
    cat <<'EOF'
Usage:
  ./scripts/train_default_distillation.sh [--no_save] [--no_visualize]

Default recipe:
  1. DCM warmup for 500 steps.
  2. SCM-only training from the DCM checkpoint to step 1000.
  3. Full rCM training from the SCM checkpoint.

Useful environment variables:
  DCM_STEPS=500
  SCM_STEPS=1000
  RCM_STEPS=<unset by default>
  DCM_CONFIG=configs/bidirectional_dcm.yaml
  SCM_CONFIG=configs/bidirectional_scm.yaml
  RCM_CONFIG=configs/bidirectional_rcm.yaml
  OUTPUT_PATH=/path/to/outputs
  PIPELINE_RUN_PREFIX=<timestamp by default>
  DCM_CHECKPOINT=/path/to/checkpoint_000500/model.pth  # skip DCM warmup
  SCM_CHECKPOINT=/path/to/checkpoint_001000/model.pth  # skip DCM warmup and SCM
  DRY_RUN=1

Distributed variables are forwarded to train_bidirectional.sh:
  NPROC_PER_NODE or NUM_GPUS, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT.
EOF
}

if [[ $# -gt 0 && "$1" =~ ^(-h|--help)$ ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTILLATION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LTX2_ROOT="$(cd "${DISTILLATION_ROOT}/../.." && pwd)"
cd "${DISTILLATION_ROOT}"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "${LTX2_ROOT}/.pixi/envs/default/bin/python" ]; then
        PYTHON_BIN="${LTX2_ROOT}/.pixi/envs/default/bin/python"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "[error] neither python nor python3 was found" >&2
        exit 2
    fi
fi

DCM_STEPS="${DCM_STEPS:-500}"
SCM_STEPS="${SCM_STEPS:-500}"
DCM_CONFIG="${DCM_CONFIG:-configs/bidirectional_dcm.yaml}"
SCM_CONFIG="${SCM_CONFIG:-configs/bidirectional_scm.yaml}"
RCM_CONFIG="${RCM_CONFIG:-configs/bidirectional_rcm.yaml}"
PIPELINE_RUN_PREFIX="${PIPELINE_RUN_PREFIX:-$(date +%m%d_%H%M%S)_TurboT2AV_default}"

read_output_path() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
print(str(cfg.get("output_path", "")))
PY
}

OUTPUT_ROOT="${OUTPUT_PATH:-${TURBO_OUTPUT_PATH:-$(read_output_path "${DCM_CONFIG}")}}"
if [ -z "${OUTPUT_ROOT}" ]; then
    echo "[error] output_path is empty; set OUTPUT_PATH or configure ${DCM_CONFIG}" >&2
    exit 2
fi

checkpoint_path() {
    local run_dir="$1"
    local step="$2"
    printf "%s/%s/checkpoints/checkpoint_%06d/model.pth" "${OUTPUT_ROOT}" "${run_dir}" "${step}"
}

require_checkpoint() {
    local label="$1"
    local path="$2"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        return 0
    fi
    if [ ! -f "${path}" ]; then
        echo "[error] ${label} checkpoint was not found: ${path}" >&2
        exit 1
    fi
}

# Override config paths via env vars (set once, applies to all stages)
patch_config() {
    local src="$1"
    local dst="$2"
    "${PYTHON_BIN}" - "${src}" "${dst}" <<'PY'
import sys
from omegaconf import OmegaConf
import os
cfg = OmegaConf.load(sys.argv[1])
for key, env in [
    ("checkpoint_path", "TURBO_CHECKPOINT_PATH"),
    ("gemma_path", "TURBO_GEMMA_PATH"),
    ("data_path", "TURBO_DATA_PATH"),
    ("scm_data_path", "TURBO_SCM_DATA_PATH"),
    ("output_path", "TURBO_OUTPUT_PATH"),
]:
    if os.environ.get(env):
        cfg[key] = os.environ[env]
OmegaConf.save(cfg, sys.argv[2])
PY
}

# Patch configs with env var paths
for cfg_var in DCM_CONFIG SCM_CONFIG RCM_CONFIG; do
    src="${!cfg_var}"
    patched="/tmp/turbot2av_${cfg_var}_$(date +%s).yaml"
    patch_config "${src}" "${patched}"
    eval "${cfg_var}=${patched}"
done

run_stage() {
    local label="$1"
    shift
    echo "========================================================"
    echo "${label}"
    echo "========================================================"
    "$@"
}

DCM_RUN_DIR="${DCM_RUN_DIR:-${PIPELINE_RUN_PREFIX}_dcm500_warmup}"
SCM_RUN_DIR="${SCM_RUN_DIR:-${PIPELINE_RUN_PREFIX}_scm1000_from_dcm500}"
RCM_RUN_DIR="${RCM_RUN_DIR:-${PIPELINE_RUN_PREFIX}_rcm_from_scm1000}"

echo "Default TurboT2AV recipe"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run prefix:  ${PIPELINE_RUN_PREFIX}"
echo "Recipe:      DCM ${DCM_STEPS} -> SCM ${SCM_STEPS} -> rCM"

if [ -n "${SCM_CHECKPOINT:-}" ]; then
    echo "Using SCM_CHECKPOINT and skipping DCM/SCM: ${SCM_CHECKPOINT}"
    SCM_CKPT="${SCM_CHECKPOINT}"
else
    if [ -n "${DCM_CHECKPOINT:-}" ]; then
        echo "Using DCM_CHECKPOINT and skipping DCM: ${DCM_CHECKPOINT}"
        DCM_CKPT="${DCM_CHECKPOINT}"
    else
        dcm_env=(
            "PYTHONUNBUFFERED=1"
            "RUN_DIR_NAME=${DCM_RUN_DIR}"
            "MAX_STEPS=${DCM_STEPS}"
            "CHECKPOINT_ITERS=${DCM_CHECKPOINT_ITERS:-${DCM_STEPS}}"
            "WANDB_NAME=${DCM_WANDB_NAME:-dcm500_warmup}"
        )
        if [ -n "${OUTPUT_PATH:-}" ]; then
            dcm_env+=("OUTPUT_PATH=${OUTPUT_PATH}")
        fi
        run_stage "Phase 1/3: DCM ${DCM_STEPS} warmup" \
            env "${dcm_env[@]}" "${SCRIPT_DIR}/train_dcm.sh" "${DCM_CONFIG}" "$@"
        DCM_CKPT="$(checkpoint_path "${DCM_RUN_DIR}" "${DCM_STEPS}")"
    fi
    require_checkpoint "DCM" "${DCM_CKPT}"

    scm_env=(
        "RUN_DIR_NAME=${SCM_RUN_DIR}"
        "WARMUP_CHECKPOINT=${DCM_CKPT}"
        "MAX_STEPS=${SCM_STEPS}"
        "CHECKPOINT_ITERS=${SCM_CHECKPOINT_ITERS:-500}"
        "WANDB_NAME=${SCM_WANDB_NAME:-scm1000_from_dcm500}"
    )
    if [ -n "${OUTPUT_PATH:-}" ]; then
        scm_env+=("OUTPUT_PATH=${OUTPUT_PATH}")
    fi
    run_stage "Phase 2/3: SCM to step ${SCM_STEPS} from DCM warmup" \
        env "${scm_env[@]}" "${SCRIPT_DIR}/train_scm.sh" "${SCM_CONFIG}" "$@"
    SCM_CKPT="$(checkpoint_path "${SCM_RUN_DIR}" "${SCM_STEPS}")"
fi
require_checkpoint "SCM" "${SCM_CKPT}"

rcm_env=(
    "RUN_DIR_NAME=${RCM_RUN_DIR}"
    "WARMUP_CHECKPOINT=${SCM_CKPT}"
    "WANDB_NAME=${RCM_WANDB_NAME:-rcm_from_scm1000}"
)
if [ -n "${RCM_STEPS:-}" ]; then
    rcm_env+=("MAX_STEPS=${RCM_STEPS}")
fi
if [ -n "${RCM_CHECKPOINT_ITERS:-}" ]; then
    rcm_env+=("CHECKPOINT_ITERS=${RCM_CHECKPOINT_ITERS}")
fi
if [ -n "${OUTPUT_PATH:-}" ]; then
    rcm_env+=("OUTPUT_PATH=${OUTPUT_PATH}")
fi

run_stage "Phase 3/3: full rCM from SCM checkpoint" \
    env "${rcm_env[@]}" "${SCRIPT_DIR}/train_bidirectional.sh" rcm "${RCM_CONFIG}" "$@"

echo "Default TurboT2AV recipe finished."
