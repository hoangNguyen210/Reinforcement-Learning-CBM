#!/usr/bin/env bash
# run_schain_filter_parallel.sh
#
# Runs the difficulty filter on S-Chain 10% and 40% datasets using all 8 A100 GPUs.
# One model instance per GPU (each handles 1/8 of the samples in parallel).
# Then merges shard results and writes drop-in JSON replacements.
#
# Usage:
#   bash run_schain_filter_parallel.sh
#   # Logs go to $LOG_DIR/{10pct,40pct}_shard{0..7}.log

set -euo pipefail

# ── Config ─────────────────────────────────────────────────────────────────────
SCHAIN_ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SChain"
DATA_CBM="${SCHAIN_ROOT}/data/experiments/cbm"
IMAGE_DIR="${SCHAIN_ROOT}/data/main/data/images"
MODEL="${MODEL:-${SCHAIN_ROOT}/model_weights/Qwen3-VL-8B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILTER_PY="${SCRIPT_DIR}/schain_filter_difficulty.py"

VISUAL_RFT="${SCHAIN_ROOT}/Hoang-Development/Reinforcement-Learning-with-CBM-for-Verified-Visual-COT/Visual-RFT"
BUILD_PY="${VISUAL_RFT}/dataset/build_schain_sft_dataset.py"

NUM_GPUS=8
N_ROLLOUTS="${N_ROLLOUTS:-8}"
MIN_PASS="${MIN_PASS:-1}"
MAX_PASS="${MAX_PASS:-6}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

LOG_DIR="${SCHAIN_ROOT}/runs/schain_filter"
TMP_DIR="${LOG_DIR}/tmp_shards"
mkdir -p "${LOG_DIR}" "${TMP_DIR}"

# ── Conda ──────────────────────────────────────────────────────────────────────
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA_Quantization/chi/miniforge3/etc/profile.d/conda.sh
conda activate visual-rft
export LD_LIBRARY_PATH=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.conda/envs/visual-rft/lib:${LD_LIBRARY_PATH:-}

echo "================================================================"
echo " S-Chain Difficulty Filter  —  8 × A100 parallel"
echo "  Model      : ${MODEL}"
echo "  N rollouts : ${N_ROLLOUTS}   keep pass=[${MIN_PASS}, ${MAX_PASS}]"
echo "  Logs       : ${LOG_DIR}"
echo "================================================================"

# ── Helper: run one dataset in parallel across all GPUs ───────────────────────
filter_dataset() {
    local label="$1"       # e.g. "40pct"
    local input_json="$2"
    local output_json="$3"

    echo ""
    echo "── ${label}: launching ${NUM_GPUS} shards ──────────────────────────"

    local pids=()
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        local shard_out="${TMP_DIR}/${label}_shard${gpu}.jsonl"
        local log="${LOG_DIR}/${label}_shard${gpu}.log"

        CUDA_VISIBLE_DEVICES="${gpu}" python "${FILTER_PY}" \
            --input           "${input_json}" \
            --images          "${IMAGE_DIR}" \
            --model           "${MODEL}" \
            --shard_idx       "${gpu}" \
            --num_shards      "${NUM_GPUS}" \
            --save_pass_counts "${shard_out}" \
            --n_rollouts      "${N_ROLLOUTS}" \
            --temperature     "${TEMPERATURE}" \
            --max_new_tokens  "${MAX_NEW_TOKENS}" \
            > "${log}" 2>&1 &
        pids+=($!)
        echo "  GPU ${gpu}  PID ${pids[-1]}  log: ${log}"
    done

    echo "  Waiting for all ${NUM_GPUS} shards to finish ..."
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            echo "  ERROR: shard PID ${pid} failed — check logs in ${LOG_DIR}"
            failed=$((failed + 1))
        fi
    done
    if [ "${failed}" -gt 0 ]; then
        echo "ABORT: ${failed} shard(s) failed for ${label}." >&2
        exit 1
    fi
    echo "  All shards done."

    # Merge + filter
    echo "  Merging shards and filtering → ${output_json}"
    python "${FILTER_PY}" \
        --merge_shards \
        --input          "${input_json}" \
        --images         "${IMAGE_DIR}" \
        --output         "${output_json}" \
        --shard_pattern  "${TMP_DIR}/${label}_shard*.jsonl" \
        --n_rollouts     "${N_ROLLOUTS}" \
        --min_pass       "${MIN_PASS}" \
        --max_pass       "${MAX_PASS}"

    echo "  Done: ${output_json}"
}

# ── Run 40% then 10% ──────────────────────────────────────────────────────────
filter_dataset "40pct" \
    "${DATA_CBM}/ablation1_40percent_all_questions.json" \
    "${DATA_CBM}/filtered_40pct.json"

filter_dataset "10pct" \
    "${DATA_CBM}/ablation1_10percent_all_questions.json" \
    "${DATA_CBM}/filtered_10pct.json"

# ── Build HF datasets for Hoang's Visual-RFT pipeline ─────────────────────────
if [ -f "${BUILD_PY}" ]; then
    echo ""
    echo "── Building HF datasets for Visual-RFT ────────────────────────────"
    for pct in 10 40; do
        HF_OUT="${VISUAL_RFT}/share_data/schain_sft_filtered_${pct}pct"
        echo "  ${pct}% → ${HF_OUT}"
        python "${BUILD_PY}" \
            --json   "${DATA_CBM}/filtered_${pct}pct.json" \
            --images "${IMAGE_DIR}" \
            --out    "${HF_OUT}"
    done
fi

echo ""
echo "================================================================"
echo " ALL DONE"
echo ""
echo "  Dang pipeline  (--data_path):"
echo "    10%: ${DATA_CBM}/filtered_10pct.json"
echo "    40%: ${DATA_CBM}/filtered_40pct.json"
echo ""
echo "  Hoang pipeline  (DATA_PATH=...):"
echo "    10%: ${VISUAL_RFT}/share_data/schain_sft_filtered_10pct"
echo "    40%: ${VISUAL_RFT}/share_data/schain_sft_filtered_40pct"
echo "================================================================"
