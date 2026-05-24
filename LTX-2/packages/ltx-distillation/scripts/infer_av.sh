#!/usr/bin/env bash
# Standalone audio-video inference launcher for teacher or student checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTILLATION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LTX2_ROOT="$(cd "${DISTILLATION_ROOT}/../.." && pwd)"

MODEL_KIND="${MODEL_KIND:-${1:-student}}"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
    shift
fi

CONFIG_PATH="${CONFIG_PATH:-configs/stage1_bidirectional_dmd.yaml}"
PROMPTS_FILE="${PROMPTS_FILE:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/release_prompts.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/inference}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
SEED="${SEED:-12345}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"

cd "${DISTILLATION_ROOT}"

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

CMD=(
    "${PYTHON_BIN}" -m ltx_distillation.tools.run_av_inference_eval
    --config_path "${CONFIG_PATH}"
    --prompts_file "${PROMPTS_FILE}"
    --output_dir "${OUTPUT_DIR}"
    --model_kind "${MODEL_KIND}"
    --num_prompts "${NUM_PROMPTS}"
    --seed "${SEED}"
    --num_shards "${NUM_SHARDS}"
    --shard_id "${SHARD_ID}"
)

if [ "${MODEL_KIND}" = "student" ]; then
    if [ -z "${STUDENT_CHECKPOINT:-}" ]; then
        echo "[error] STUDENT_CHECKPOINT is required when MODEL_KIND=student" >&2
        exit 2
    fi
    CMD+=(--student_checkpoint "${STUDENT_CHECKPOINT}")
    CMD+=(--student_param "${STUDENT_PARAM:-auto}")
    if [ "${STUDENT_STRICT:-0}" = "1" ]; then
        CMD+=(--student_strict)
    fi
elif [ "${MODEL_KIND}" = "teacher" ]; then
    CMD+=(--teacher_mode "${TEACHER_MODE:-native_rf}")
    CMD+=(--teacher_steps "${TEACHER_STEPS:-40}")
else
    echo "[error] MODEL_KIND must be student or teacher, got: ${MODEL_KIND}" >&2
    exit 2
fi

if [ "${OVERWRITE:-0}" = "1" ]; then
    CMD+=(--overwrite)
fi

"${CMD[@]}" "$@"
