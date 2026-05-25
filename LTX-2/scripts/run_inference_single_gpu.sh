#!/usr/bin/env bash
# Single-GPU sequential inference for the default TurboT2AV comparison set.
#
# Usage:
#   bash LTX-2/scripts/run_inference_single_gpu.sh [GPU_ID] [NUM_PROMPTS]
#
# Useful environment variables:
#   PYTHON_BIN=/path/to/python
#   DATA_ROOT=/data/datasets/turbodiff_datasets_and_ckpt
#   RUN_ROOT=/path/to/training-runs
#   OUTPUT_ROOT=/path/to/inference-output
#   PROMPTS_FILE=/path/to/prompts.txt

set -euo pipefail

GPU="${1:-0}"
NUM_PROMPTS="${2:-200}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LTX2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DISTILLATION_ROOT="${LTX2_ROOT}/packages/ltx-distillation"
REPO_ROOT="$(cd "${LTX2_ROOT}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/data/datasets/turbodiff_datasets_and_ckpt}"
RUN_ROOT="${RUN_ROOT:-${DATA_ROOT}/my_omniforcing}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/my_TurboT2AV/inference_single_seed}"
PROMPTS_FILE="${PROMPTS_FILE:-${DATA_ROOT}/tavgbench/release_prompts.txt}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${LTX2_ROOT}/.pixi/envs/default/bin/python" ]]; then
        PYTHON_BIN="${LTX2_ROOT}/.pixi/envs/default/bin/python"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "[error] no python interpreter found; set PYTHON_BIN" >&2
        exit 2
    fi
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${DISTILLATION_ROOT}/src:${LTX2_ROOT}/packages/ltx-causal/src:${LTX2_ROOT}/packages/ltx-core/src:${LTX2_ROOT}/packages/ltx-pipelines/src${PYTHONPATH:+:${PYTHONPATH}}"

run_one() {
    local idx="$1"
    local name="$2"
    local config="$3"
    local checkpoint="$4"
    local out="${OUTPUT_ROOT}/${idx}_${name}"

    if [[ ! -f "${config}" ]]; then
        echo "[error] config not found: ${config}" >&2
        exit 1
    fi
    if [[ ! -f "${checkpoint}" ]]; then
        echo "[error] checkpoint not found: ${checkpoint}" >&2
        exit 1
    fi
    if [[ ! -f "${PROMPTS_FILE}" ]]; then
        echo "[error] prompts file not found: ${PROMPTS_FILE}" >&2
        exit 1
    fi

    rm -rf "${out}"
    mkdir -p "${out}"

    echo "=== [$(date +%H:%M:%S)] ${idx}_${name} ==="
    "${PYTHON_BIN}" -u -m ltx_distillation.tools.run_av_inference_eval \
      --config_path "${config}" \
      --prompts_file "${PROMPTS_FILE}" \
      --output_dir "${out}" \
      --model_kind student \
      --student_checkpoint "${checkpoint}" \
      --student_param auto \
      --num_prompts "${NUM_PROMPTS}" \
      --num_shards 1 \
      --shard_id 0 \
      --no_init_lock \
      > "${out}/run.log" 2>&1

    find "${out}" -maxdepth 1 -name 'sample_*.wav' -delete 2>/dev/null
    echo "sample,prompt" > "${out}/samples.csv"
    awk -v limit="${NUM_PROMPTS}" 'NR <= limit {printf "sample_%04d.mp4,\"%s\"\n", NR-1, $0}' "${PROMPTS_FILE}" >> "${out}/samples.csv"
    echo "  Done: $(find "${out}" -maxdepth 1 -name 'sample_*.mp4' | wc -l) videos"
}

run_one "00" "dcm500" \
    "${RUN_ROOT}/0519_200804_dcm_test_48steps/config.yaml" \
    "${RUN_ROOT}/0519_200804_dcm_test_48steps/checkpoints/checkpoint_000500/model.pth"

run_one "01" "scm1000" \
    "${RUN_ROOT}/0520_011955_scm_from_dcm_warmup/config.yaml" \
    "${RUN_ROOT}/0520_011955_scm_from_dcm_warmup/checkpoints/checkpoint_001000/model.pth"

run_one "02" "dmd_only" \
    "${RUN_ROOT}/0522_201755_dmd_only_from_scm1000_0412aligned_test/config.yaml" \
    "${RUN_ROOT}/0522_201755_dmd_only_from_scm1000_0412aligned_test/checkpoints/checkpoint_001000/model.pth"

run_one "03" "scm_dmd_scm1" \
    "${RUN_ROOT}/0523_170704_scm_dmd_sum_from_1000_reverse_scm_sharedbatch/config.yaml" \
    "${RUN_ROOT}/0523_170704_scm_dmd_sum_from_1000_reverse_scm_sharedbatch/checkpoints/checkpoint_001000/model.pth"

run_one "04" "scm_dmd_scm1000" \
    "${RUN_ROOT}/0523_201907_scm_dmd_sum_scm1000_dmd1_from_1000_reverse_scm_sharedbatch/config.yaml" \
    "${RUN_ROOT}/0523_201907_scm_dmd_sum_scm1000_dmd1_from_1000_reverse_scm_sharedbatch/checkpoints/checkpoint_001000/model.pth"

run_one "05" "scm_dmd_scm100" \
    "${RUN_ROOT}/0523_235247_scm_dmd_sum_scm100_dmd1_to2051_from_1000_reverse_scm_sharedbatch/config.yaml" \
    "${RUN_ROOT}/0523_235247_scm_dmd_sum_scm100_dmd1_to2051_from_1000_reverse_scm_sharedbatch/checkpoints/checkpoint_002000/model.pth"

run_one "06" "scm_dmd_scm10" \
    "${RUN_ROOT}/0524_090536_scm_dmd_sum_scm10_dmd1_after2051_from_1000_reverse_scm_sharedbatch/config.yaml" \
    "${RUN_ROOT}/0524_090536_scm_dmd_sum_scm10_dmd1_after2051_from_1000_reverse_scm_sharedbatch/checkpoints/checkpoint_002000/model.pth"

echo "=== [$(date +%H:%M:%S)] All done. ==="
