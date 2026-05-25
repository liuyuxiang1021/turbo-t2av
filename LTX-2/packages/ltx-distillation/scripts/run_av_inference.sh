#!/usr/bin/env bash
# Launch single-GPU AV inference for one checkpoint.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./scripts/run_av_inference.sh \
    --config PATH \
    --checkpoint PATH \
    --output-dir DIR \
    [--prompts-file PATH] [--num-prompts 200] [--gpu 0]

Useful environment variables:
  PYTHON_BIN=/path/to/python
  EXTRA_ARGS="--overwrite"
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTILLATION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DISTILLATION_ROOT}/../../.." && pwd)"

CONFIG_PATH=""
CHECKPOINT_PATH=""
OUTPUT_DIR=""
PROMPTS_FILE="/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/release_prompts.txt"
NUM_PROMPTS="200"
GPU="0"
STUDENT_PARAM="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|--config-path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --checkpoint|--student-checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --prompts-file)
            PROMPTS_FILE="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --student-param)
            STUDENT_PARAM="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[error] unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${CONFIG_PATH}" || -z "${CHECKPOINT_PATH}" || -z "${OUTPUT_DIR}" ]]; then
    echo "[error] --config, --checkpoint, and --output-dir are required" >&2
    usage >&2
    exit 2
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${REPO_ROOT}/LTX-2/.pixi/envs/default/bin/python" ]]; then
        PYTHON_BIN="${REPO_ROOT}/LTX-2/.pixi/envs/default/bin/python"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "[error] no python interpreter found; set PYTHON_BIN" >&2
        exit 2
    fi
fi

mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${DISTILLATION_ROOT}/src:${REPO_ROOT}/LTX-2/packages/ltx-causal/src:${REPO_ROOT}/LTX-2/packages/ltx-core/src:${REPO_ROOT}/LTX-2/packages/ltx-pipelines/src${PYTHONPATH:+:${PYTHONPATH}}"

extra_args=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${EXTRA_ARGS})
fi

exec "${PYTHON_BIN}" -u -m ltx_distillation.tools.run_av_inference_eval \
  --config_path "${CONFIG_PATH}" \
  --prompts_file "${PROMPTS_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_kind student \
  --student_checkpoint "${CHECKPOINT_PATH}" \
  --student_param "${STUDENT_PARAM}" \
  --num_prompts "${NUM_PROMPTS}" \
  --num_shards 1 \
  --shard_id 0 \
  "${extra_args[@]}"
