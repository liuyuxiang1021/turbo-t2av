#!/usr/bin/env bash
# Unified launcher for bidirectional turbo-t2av distillation modes.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./scripts/train_bidirectional.sh <mode> [config_path] [--no_save] [--no_visualize]

Modes:
  dcm       DCM warmup only
  scm       SCM only
  dmd       DMD only
  rcm       rCM-style joint SCM + DMD
  scm_dmd   alias for rcm

Warmup:
  Set DCM_CHECKPOINT=/path/to/dcm/checkpoint/model.pth to initialize scm, dmd,
  or rcm from a DCM checkpoint without loading optimizer/scheduler state.

Distributed environment:
  NPROC_PER_NODE or NUM_GPUS, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTILLATION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LTX2_ROOT="$(cd "${DISTILLATION_ROOT}/../.." && pwd)"

if [[ $# -gt 0 && "$1" =~ ^(-h|--help)$ ]]; then
    usage
    exit 0
fi

MODE="${1:-${MODE:-dmd}}"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
    shift
fi

case "${MODE}" in
    dcm)
        DEFAULT_CONFIG="configs/bidirectional_dcm.yaml"
        TITLE="Bidirectional DCM Warmup"
        ;;
    scm)
        DEFAULT_CONFIG="configs/bidirectional_scm.yaml"
        TITLE="Bidirectional SCM"
        ;;
    dmd)
        DEFAULT_CONFIG="configs/bidirectional_dmd.yaml"
        TITLE="Bidirectional DMD"
        ;;
    rcm|scm_dmd|scm+dmd)
        MODE="rcm"
        DEFAULT_CONFIG="configs/bidirectional_rcm.yaml"
        TITLE="Bidirectional rCM (SCM + DMD)"
        ;;
    *)
        echo "[error] unsupported mode: ${MODE}" >&2
        usage >&2
        exit 2
        ;;
esac

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
    CONFIG_PATH="$1"
    shift
else
    CONFIG_PATH="${DEFAULT_CONFIG}"
fi

cd "${DISTILLATION_ROOT}"
echo "Working dir: $(pwd)"

if [ -n "${VENV_PATH:-}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Activated venv: ${VENV_PATH}"
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "[error] neither python nor python3 was found" >&2
        exit 2
    fi
fi

export PYTHONPATH="${DISTILLATION_ROOT}/src:${LTX2_ROOT}/packages/ltx-causal/src:${LTX2_ROOT}/packages/ltx-core/src:${LTX2_ROOT}/packages/ltx-pipelines/src${PYTHONPATH:+:${PYTHONPATH}}"

WARMUP_CHECKPOINT="${WARMUP_CHECKPOINT:-${DCM_CHECKPOINT:-}}"
GENERATED_CONFIG=""
cleanup_generated_config() {
    if [ -n "${GENERATED_CONFIG}" ]; then
        rm -f "${GENERATED_CONFIG}"
    fi
}
trap cleanup_generated_config EXIT

if [ -n "${WARMUP_CHECKPOINT}" ]; then
    if [ "${MODE}" = "dcm" ]; then
        echo "[warn] DCM_CHECKPOINT is ignored for mode=dcm" >&2
    else
        GENERATED_CONFIG="$(mktemp "${TMPDIR:-/tmp}/turbo-t2av-${MODE//[^a-zA-Z0-9]/_}.XXXXXX.yaml")"
        "${PYTHON_BIN}" - "$CONFIG_PATH" "$GENERATED_CONFIG" "$WARMUP_CHECKPOINT" "${CHECKPOINT_LOAD_MODE:-parallel}" <<'PY'
import sys
from omegaconf import OmegaConf

src, dst, warmup, load_mode = sys.argv[1:5]
cfg = OmegaConf.load(src)
cfg.resume_checkpoint = warmup
cfg.checkpoint_load_mode = load_mode
cfg.resume_training_state = False
cfg.skip_initial_checkpoint = True
if "wandb_name" in cfg and not str(cfg.wandb_name).endswith("_from_dcm_warmup"):
    cfg.wandb_name = f"{cfg.wandb_name}_from_dcm_warmup"
OmegaConf.save(cfg, dst)
PY
        CONFIG_PATH="${GENERATED_CONFIG}"
    fi
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS:-${LOCAL_WORLD_SIZE:-8}}}"
if [ -z "${NNODES:-}" ]; then
    if [ -n "${SLURM_NNODES:-}" ]; then
        NNODES="${SLURM_NNODES}"
    elif [ -n "${GROUP_WORLD_SIZE:-}" ]; then
        NNODES="${GROUP_WORLD_SIZE}"
    elif [ -n "${WORLD_SIZE:-}" ] && [ -n "${LOCAL_WORLD_SIZE:-}" ] && [ "${LOCAL_WORLD_SIZE}" -gt 0 ] && [ $((WORLD_SIZE % LOCAL_WORLD_SIZE)) -eq 0 ]; then
        NNODES="$((WORLD_SIZE / LOCAL_WORLD_SIZE))"
    else
        NNODES=1
    fi
fi

if [ -z "${NODE_RANK:-}" ]; then
    if [ -n "${SLURM_NODEID:-}" ]; then
        NODE_RANK="${SLURM_NODEID}"
    elif [ -n "${GROUP_RANK:-}" ]; then
        NODE_RANK="${GROUP_RANK}"
    elif [ -n "${RANK:-}" ] && [ -n "${LOCAL_WORLD_SIZE:-}" ] && [ "${LOCAL_WORLD_SIZE}" -gt 0 ]; then
        NODE_RANK="$((RANK / LOCAL_WORLD_SIZE))"
    else
        NODE_RANK=0
    fi
fi

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

NUM_CPUS="${NUM_CPUS:-128}"
export OMP_NUM_THREADS=$((NUM_CPUS / NPROC_PER_NODE))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TOTAL_GPUS=$((NPROC_PER_NODE * NNODES))

echo "========================================================"
echo "${TITLE}"
echo "========================================================"
echo "Mode:          ${MODE}"
echo "Config:        ${CONFIG_PATH}"
if [ -n "${WARMUP_CHECKPOINT}" ] && [ "${MODE}" != "dcm" ]; then
    echo "DCM warmup:    ${WARMUP_CHECKPOINT}"
fi
echo "Nodes:         ${NNODES}  |  GPUs/Node: ${NPROC_PER_NODE}  |  Total: ${TOTAL_GPUS}"
echo "Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "========================================================"

readarray -t CONFIG_LOG_VALUES < <(
"${PYTHON_BIN}" - <<'PY' "$CONFIG_PATH"
import sys
cfg_path = sys.argv[1]
output_path = ""
wandb_name = ""
with open(cfg_path, "r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if s.startswith("output_path:") and not output_path:
            output_path = s.split(":", 1)[1].strip().strip('"').strip("'")
        elif s.startswith("wandb_name:") and not wandb_name:
            wandb_name = s.split(":", 1)[1].strip().strip('"').strip("'")
print(output_path)
print(wandb_name)
PY
)
CONFIG_OUTPUT_PATH="${CONFIG_LOG_VALUES[0]:-}"
CONFIG_WANDB_NAME="${CONFIG_LOG_VALUES[1]:-${MODE}}"
if [ -n "$CONFIG_OUTPUT_PATH" ]; then
    RUN_DIR_NAME="${RUN_DIR_NAME:-$(date +%m%d_%H%M%S)_${CONFIG_WANDB_NAME}}"
    export LTX_RUN_DIR_NAME="$RUN_DIR_NAME"
    RUN_OUTPUT_PATH="$CONFIG_OUTPUT_PATH/$RUN_DIR_NAME"
    LOG_FILE_DEFAULT="$RUN_OUTPUT_PATH/train.log"
else
    LOG_FILE_DEFAULT="train.log"
fi
LOG_FILE="${LOG_FILE:-$LOG_FILE_DEFAULT}"
echo "Logging:       ${LOG_FILE}"
echo "========================================================"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1, torchrun command was not executed."
    exit 0
fi

if [ -n "${RUN_OUTPUT_PATH:-}" ]; then
    mkdir -p "$RUN_OUTPUT_PATH"
fi

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m ltx_distillation.train_distillation \
    --config_path "${CONFIG_PATH}" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"
