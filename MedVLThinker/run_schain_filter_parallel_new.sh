#!/usr/bin/env bash
# run_schain_filter_parallel_new.sh
#
# Applies difficulty filtering to the FULL 100% training dataset, then
# randomly samples down to 10% and 40% of the original full-dataset size.
#
# This is the "fair" version: the old approach filtered from already-subsampled
# 10%/40% slices (biased pool). This script filters the entire dataset first
# and then draws equally-sized samples — making the comparison fair.
#
# Output:
#   data/experiments/cbm/filtered_10pct_new.json  (1078 samples, ~10% of full)
#   data/experiments/cbm/filtered_40pct_new.json  (4313 samples, ~40% of full)
#
# Run from anywhere:
#   bash run_schain_filter_parallel_new.sh
#   # Logs: $SCHAIN_ROOT/runs/schain_filter_new/

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
SCHAIN_ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SChain"
DATA_MAIN="${SCHAIN_ROOT}/data/main/data"
DATA_CBM="${SCHAIN_ROOT}/data/experiments/cbm"
IMAGE_DIR="${DATA_MAIN}/images"
FULL_TRAIN="${DATA_MAIN}/llava_med_mri_bbox_train_CoT_new.json"   # 10 783 samples
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
SEED=42

# 10% and 40% of the full dataset (10 783 samples)
TARGET_10PCT=1078
TARGET_40PCT=4313

LOG_DIR="${SCHAIN_ROOT}/runs/schain_filter_new"
TMP_DIR="${LOG_DIR}/tmp_shards"
mkdir -p "${LOG_DIR}" "${TMP_DIR}"

# ── Conda ───────────────────────────────────────────────────────────────────
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA_Quantization/chi/miniforge3/etc/profile.d/conda.sh
conda activate visual-rft
export LD_LIBRARY_PATH=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.conda/envs/visual-rft/lib:${LD_LIBRARY_PATH:-}

echo "================================================================"
echo " S-Chain Difficulty Filter (NEW — full-dataset base)"
echo "  Input      : ${FULL_TRAIN}"
echo "  Model      : ${MODEL}"
echo "  N rollouts : ${N_ROLLOUTS}   keep pass=[${MIN_PASS}, ${MAX_PASS}]"
echo "  Targets    : 10%=${TARGET_10PCT}  40%=${TARGET_40PCT} samples"
echo "  Logs       : ${LOG_DIR}"
echo "================================================================"

# ── Step 1: run difficulty filter on full 100% training data ───────────────
FULL_FILTERED="${DATA_CBM}/filtered_100pct_all_new.json"

if [ -f "${FULL_FILTERED}" ]; then
    echo "Reusing existing full-filtered file: ${FULL_FILTERED}"
else
    echo ""
    echo "── Filtering full 100% dataset: launching ${NUM_GPUS} shards ─────────"
    pids=()
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        shard_out="${TMP_DIR}/100pct_shard${gpu}.jsonl"
        log="${LOG_DIR}/100pct_shard${gpu}.log"

        # Stagger: avoid 8 procs simultaneously mmap-loading the same model
        # files from /pfss/ — concurrent loads were crashing all shards at
        # ~20% weight load. 20s offset is enough to space out the load phase.
        if [ "${gpu}" -gt 0 ]; then sleep 20; fi
        # nohup + setsid: detach each shard into its own session so an SSH
        # disconnect / terminal close does not SIGHUP the whole group.
        CUDA_VISIBLE_DEVICES="${gpu}" nohup setsid python "${FILTER_PY}" \
            --input            "${FULL_TRAIN}" \
            --images           "${IMAGE_DIR}" \
            --model            "${MODEL}" \
            --shard_idx        "${gpu}" \
            --num_shards       "${NUM_GPUS}" \
            --save_pass_counts "${shard_out}" \
            --n_rollouts       "${N_ROLLOUTS}" \
            --temperature      "${TEMPERATURE}" \
            --max_new_tokens   "${MAX_NEW_TOKENS}" \
            < /dev/null > "${log}" 2>&1 &
        pids+=($!)
        echo "  GPU ${gpu}  PID ${pids[-1]}  log: ${log}"
    done

    echo "  Waiting for all ${NUM_GPUS} shards (this will take several hours)..."
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            echo "  ERROR: shard PID ${pid} failed — check logs in ${LOG_DIR}"
            failed=$((failed + 1))
        fi
    done
    [ "${failed}" -gt 0 ] && { echo "ABORT: ${failed} shard(s) failed." >&2; exit 1; }
    echo "  All shards done. Merging and filtering..."

    python "${FILTER_PY}" \
        --merge_shards \
        --input         "${FULL_TRAIN}" \
        --images        "${IMAGE_DIR}" \
        --output        "${FULL_FILTERED}" \
        --shard_pattern "${TMP_DIR}/100pct_shard*.jsonl" \
        --n_rollouts    "${N_ROLLOUTS}" \
        --min_pass      "${MIN_PASS}" \
        --max_pass      "${MAX_PASS}"

    echo "  Full filtered set: ${FULL_FILTERED}"
fi

# ── Step 2: random-sample to 10% and 40% of original full-dataset size ─────
echo ""
echo "── Sampling ${TARGET_10PCT} and ${TARGET_40PCT} from filtered pool ──────"
python - <<PYEOF
import json, random, sys

random.seed(${SEED})

with open("${FULL_FILTERED}") as f:
    pool = json.load(f)

n_pool = len(pool)
print(f"Filtered pool: {n_pool} samples  (from ${FULL_TRAIN})")

if n_pool < ${TARGET_40PCT}:
    print(f"WARNING: pool ({n_pool}) < target 40% (${TARGET_40PCT}); taking all for 40%.", file=sys.stderr)

# Sample 40% (larger), then take 10% as independent sample for fairness
sample_40 = random.sample(pool, min(n_pool, ${TARGET_40PCT}))
sample_10 = random.sample(pool, min(n_pool, ${TARGET_10PCT}))

out_40 = "${DATA_CBM}/filtered_40pct_new.json"
out_10 = "${DATA_CBM}/filtered_10pct_new.json"

with open(out_10, "w") as f:
    json.dump(sample_10, f, indent=2)
with open(out_40, "w") as f:
    json.dump(sample_40, f, indent=2)

print(f"filtered_10pct_new.json : {len(sample_10)} samples  -> {out_10}")
print(f"filtered_40pct_new.json : {len(sample_40)} samples  -> {out_40}")

# Class distribution
for label, ds in [("10pct_new", sample_10), ("40pct_new", sample_40)]:
    from collections import Counter
    import re
    counts = Counter()
    for item in ds:
        ans = item["conversations"][-1]["value"] if item.get("conversations") else ""
        m = re.search(r"Final Answer:\s*(Non-Dementia|Mild-Dementia|Moderate-Dementia)", ans)
        counts[m.group(1) if m else "unknown"] += 1
    total = len(ds)
    print(f"\n{label} class distribution:")
    for cls, n in sorted(counts.items()):
        print(f"  {cls:<22}: {n:>5} ({n/total*100:.1f}%)")
PYEOF

# ── Step 3: build HF datasets for Hoang's Visual-RFT pipeline ──────────────
if [ -f "${BUILD_PY}" ]; then
    echo ""
    echo "── Building HF datasets for Visual-RFT ────────────────────────────"
    for pct in 10 40; do
        HF_OUT="${VISUAL_RFT}/share_data/schain_sft_filtered_${pct}pct_new"
        echo "  ${pct}%_new → ${HF_OUT}"
        python "${BUILD_PY}" \
            --json   "${DATA_CBM}/filtered_${pct}pct_new.json" \
            --images "${IMAGE_DIR}" \
            --out    "${HF_OUT}"
    done
fi

# cleanup temp shards
rm -rf "${TMP_DIR}"

echo ""
echo "================================================================"
echo " ALL DONE"
echo ""
echo "  Dang pipeline  (--data_path):"
echo "    10%_new: ${DATA_CBM}/filtered_10pct_new.json"
echo "    40%_new: ${DATA_CBM}/filtered_40pct_new.json"
echo ""
echo "  Hoang pipeline  (DATA_PATH=...):"
echo "    10%_new: ${VISUAL_RFT}/share_data/schain_sft_filtered_10pct_new"
echo "    40%_new: ${VISUAL_RFT}/share_data/schain_sft_filtered_40pct_new"
echo ""
echo "  Next: run the _new training scripts after dataset is ready."
echo "================================================================"
