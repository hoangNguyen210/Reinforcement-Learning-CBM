#!/usr/bin/env bash
# run_schain_filter.sh — Difficulty-filter S-Chain training data (10% and 40%).
#
# Uses Qwen3-VL-8B-Instruct (already downloaded) on 2 GPUs.
# To adapt to a different model, just set MODEL= from the environment:
#   MODEL=/path/to/Qwen2.5-VL-7B-Instruct bash run_schain_filter.sh
#
# Outputs (drop-in replacements for the originals):
#   $DATA_CBM/filtered_10pct.json   ← swap into Dang's --data_path
#   $DATA_CBM/filtered_40pct.json   ← swap into Dang's --data_path
#   $VISUAL_RFT/share_data/schain_sft_filtered_10pct  ← DATA_PATH for Hoang's script
#   $VISUAL_RFT/share_data/schain_sft_filtered_40pct  ← DATA_PATH for Hoang's script

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
SCHAIN_ROOT="${SCHAIN_ROOT:-/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SChain}"
DATA_CBM="${DATA_CBM:-${SCHAIN_ROOT}/data/experiments/cbm}"
IMAGE_DIR="${IMAGE_DIR:-${SCHAIN_ROOT}/data/main/data/images}"
MEDVLTHINKER_DIR="${MEDVLTHINKER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VISUAL_RFT="${VISUAL_RFT:-${SCHAIN_ROOT}/Hoang-Development/Reinforcement-Learning-with-CBM-for-Verified-Visual-COT/Visual-RFT}"

# ── Model — swap MODEL= to use a different VLM ────────────────────────────────
MODEL="${MODEL:-${SCHAIN_ROOT}/model_weights/Qwen3-VL-8B-Instruct}"

# ── Filter settings ────────────────────────────────────────────────────────────
GPUS="${GPUS:-0,1}"                 # 2 GPUs; device_map=auto spreads the 8B model
N_ROLLOUTS="${N_ROLLOUTS:-8}"       # rollouts per sample
MIN_PASS="${MIN_PASS:-1}"           # keep pass_count >= 1 (not always wrong)
MAX_PASS="${MAX_PASS:-6}"           # keep pass_count <= 6 (not trivially easy)
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

# ── Conda ──────────────────────────────────────────────────────────────────────
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA_Quantization/chi/miniforge3/etc/profile.d/conda.sh
conda activate visual-rft
export LD_LIBRARY_PATH=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.conda/envs/visual-rft/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES="${GPUS}"

echo "============================================================"
echo " S-Chain Difficulty Filter"
echo "  Model      : ${MODEL}"
echo "  GPUs       : ${GPUS}"
echo "  N rollouts : ${N_ROLLOUTS}  keep=[${MIN_PASS}, ${MAX_PASS}]"
echo "============================================================"

filter_one() {
    local label="$1"
    local input_json="$2"
    local output_json="$3"
    local pass_jsonl="${output_json%.json}_pass_counts.jsonl"

    echo ""
    echo "── ${label} ──────────────────────────────────────────────"
    echo "  in : ${input_json}"
    echo "  out: ${output_json}"

    python "${MEDVLTHINKER_DIR}/schain_filter_difficulty.py" \
        --input           "${input_json}" \
        --images          "${IMAGE_DIR}" \
        --output          "${output_json}" \
        --model           "${MODEL}" \
        --n_rollouts      "${N_ROLLOUTS}" \
        --min_pass        "${MIN_PASS}" \
        --max_pass        "${MAX_PASS}" \
        --temperature     "${TEMPERATURE}" \
        --max_new_tokens  "${MAX_NEW_TOKENS}" \
        --save_pass_counts "${pass_jsonl}"

    echo "  pass counts saved: ${pass_jsonl}"
}

# ── Filter 10% ─────────────────────────────────────────────────────────────────
filter_one "10% dataset  (1077 samples)" \
    "${DATA_CBM}/ablation1_10percent_all_questions.json" \
    "${DATA_CBM}/filtered_10pct.json"

# ── Filter 40% ─────────────────────────────────────────────────────────────────
filter_one "40% dataset  (4274 samples)" \
    "${DATA_CBM}/ablation1_40percent_all_questions.json" \
    "${DATA_CBM}/filtered_40pct.json"

# ── Build HF datasets for Hoang's Visual-RFT pipeline ─────────────────────────
BUILD="${VISUAL_RFT}/dataset/build_schain_sft_dataset.py"
if [ -f "${BUILD}" ]; then
    echo ""
    echo "── Building HF datasets for Visual-RFT pipeline ──────────"
    for pct in 10 40; do
        HF_OUT="${VISUAL_RFT}/share_data/schain_sft_filtered_${pct}pct"
        echo "  ${pct}% → ${HF_OUT}"
        python "${BUILD}" \
            --json   "${DATA_CBM}/filtered_${pct}pct.json" \
            --images "${IMAGE_DIR}" \
            --out    "${HF_OUT}"
    done
else
    echo "WARNING: build_schain_sft_dataset.py not found — skipping HF build."
fi

echo ""
echo "============================================================"
echo " DONE"
echo ""
echo "  Dang pipeline  (--data_path):"
echo "    10%: ${DATA_CBM}/filtered_10pct.json"
echo "    40%: ${DATA_CBM}/filtered_40pct.json"
echo ""
echo "  Hoang pipeline  (DATA_PATH=...):"
echo "    10%: ${VISUAL_RFT}/share_data/schain_sft_filtered_10pct"
echo "    40%: ${VISUAL_RFT}/share_data/schain_sft_filtered_40pct"
echo ""
echo "  To use a different model next time:"
echo "    MODEL=/path/to/other-model bash run_schain_filter.sh"
echo "============================================================"
