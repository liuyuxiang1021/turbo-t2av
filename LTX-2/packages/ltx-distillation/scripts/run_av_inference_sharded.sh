#!/usr/bin/env bash
# Launch sharded AV inference while serializing checkpoint/model initialization.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./scripts/run_av_inference_sharded.sh \
    --config PATH \
    --checkpoint PATH \
    --output-dir DIR \
    [--prompts-file PATH] [--num-prompts 200] [--num-shards 8] [--gpus 0,1,2,3,4,5,6,7]

The script launches one process per shard. Each shard runs inference on its own
GPU, but all shards share the same --init_lock_path so checkpoint/model
initialization happens one process at a time. A shard starts inference
immediately after its initialization completes.

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
NUM_SHARDS="8"
GPUS="0,1,2,3,4,5,6,7"
MODEL_KIND="student"
STUDENT_PARAM="auto"
INIT_LOCK_PATH=""

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
        --num-shards)
            NUM_SHARDS="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --student-param)
            STUDENT_PARAM="$2"
            shift 2
            ;;
        --init-lock-path)
            INIT_LOCK_PATH="$2"
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

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -ne "${NUM_SHARDS}" ]]; then
    echo "[error] --gpus count (${#GPU_LIST[@]}) must match --num-shards (${NUM_SHARDS})" >&2
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

if [[ -z "${INIT_LOCK_PATH}" ]]; then
    lock_key="$(printf '%s\0%s' "$(readlink -f "${CHECKPOINT_PATH}")" "$(readlink -f "${CONFIG_PATH}")" | sha256sum | awk '{print $1}' | cut -c1-16)"
    INIT_LOCK_PATH="/tmp/turbot2av_av_eval_init_${lock_key}.lock"
fi

export PYTHONPATH="${DISTILLATION_ROOT}/src:${REPO_ROOT}/LTX-2/packages/ltx-causal/src:${REPO_ROOT}/LTX-2/packages/ltx-core/src:${REPO_ROOT}/LTX-2/packages/ltx-pipelines/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[AVEvalLaunch] output=${OUTPUT_DIR}"
echo "[AVEvalLaunch] python=${PYTHON_BIN}"
echo "[AVEvalLaunch] shards=${NUM_SHARDS} gpus=${GPUS}"
echo "[AVEvalLaunch] init_lock=${INIT_LOCK_PATH}"

for shard_id in $(seq 0 "$((NUM_SHARDS - 1))"); do
    gpu="${GPU_LIST[$shard_id]}"
    shard_name="$(printf 'shard_%02d' "${shard_id}")"
    log_path="${OUTPUT_DIR}/${shard_name}.log"
    command_path="${OUTPUT_DIR}/${shard_name}.command.sh"
    pid_path="${OUTPUT_DIR}/${shard_name}.pid"

cat > "${command_path}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${PYTHONPATH}"
extra_args=()
if [[ -n "\${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(\${EXTRA_ARGS})
fi
exec "${PYTHON_BIN}" -u -m ltx_distillation.tools.run_av_inference_eval \\
  --config_path "${CONFIG_PATH}" \\
  --prompts_file "${PROMPTS_FILE}" \\
  --output_dir "${OUTPUT_DIR}" \\
  --model_kind "${MODEL_KIND}" \\
  --student_checkpoint "${CHECKPOINT_PATH}" \\
  --student_param "${STUDENT_PARAM}" \\
  --num_prompts "${NUM_PROMPTS}" \\
  --num_shards "${NUM_SHARDS}" \\
  --shard_id "${shard_id}" \\
  --init_lock_path "${INIT_LOCK_PATH}" \\
  "\${extra_args[@]}"
EOF
    chmod +x "${command_path}"
    : > "${log_path}"
    setsid "${command_path}" >> "${log_path}" 2>&1 < /dev/null &
    echo "$!" > "${pid_path}"
    echo "[AVEvalLaunch] ${shard_name} gpu=${gpu} pid=$(cat "${pid_path}") log=${log_path}"
done

echo "[AVEvalLaunch] launched. Monitor with:"
echo "  tail -f ${OUTPUT_DIR}/shard_*.log"
