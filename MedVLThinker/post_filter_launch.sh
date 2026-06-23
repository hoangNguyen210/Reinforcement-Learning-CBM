#!/usr/bin/env bash
# post_filter_launch.sh
#
# Waits for the difficulty-filter job to finish, validates its outputs,
# then runs a 10-step smoke test on both training pipelines to catch bugs
# before committing to full training runs.
#
# Usage (already launched in background by the main session):
#   nohup bash post_filter_launch.sh > /path/to/runs/schain_filter/post_filter.log 2>&1 &

set -euo pipefail

SCHAIN_ROOT=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SChain
DATA_CBM=${SCHAIN_ROOT}/data/experiments/cbm
LOG_DIR=${SCHAIN_ROOT}/runs/schain_filter
FILTER_MAIN_LOG=${LOG_DIR}/main.log

EXGRA_MED_DIR=${SCHAIN_ROOT}/Dang-Development/Reinforcement-Learning-with-CBM-for-Verified-Visual-COT/architectures/Exgra-Med
VISUAL_RFT_SCRIPTS=${SCHAIN_ROOT}/Hoang-Development/Reinforcement-Learning-with-CBM-for-Verified-Visual-COT/Visual-RFT/src/scripts

# ── Step 1: wait for filter job ───────────────────────────────────────────────
echo "[$(date)] Waiting for filter job to complete (watching ${FILTER_MAIN_LOG}) ..."
until grep -q "ALL DONE" "${FILTER_MAIN_LOG}" 2>/dev/null; do
    sleep 30
done
echo "[$(date)] Filter job finished."

# ── Step 2: validate filtered JSONs ───────────────────────────────────────────
echo ""
echo "── Validating filtered datasets ────────────────────────────────"
python3 - <<'PYEOF'
import json, sys

checks = [
    ("10pct", "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SChain/data/experiments/cbm/filtered_10pct.json", 1077),
    ("40pct", "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SChain/data/experiments/cbm/filtered_40pct.json", 4274),
]
ok = True
for label, path, orig_n in checks:
    with open(path) as f:
        data = json.load(f)
    n = len(data)
    pct = n / orig_n * 100
    status = "OK" if n > 0 else "EMPTY — FAIL"
    print(f"  {label}: {n} / {orig_n} kept ({pct:.1f}%)  [{status}]")
    if n == 0:
        ok = False
if not ok:
    print("VALIDATION FAILED — check filter logs"); sys.exit(1)
print("Validation passed.")
PYEOF

# ── Step 3: smoke test — Hoang's pipeline (filtered 10%) ─────────────────────
echo ""
echo "── Smoke test 1: Hoang / Qwen2-VL / filtered 10% (max_steps=10) ──"
SMOKE_LOG=${LOG_DIR}/smoke_hoang_10pct.log
SAVE_SMOKE_HOANG=${SCHAIN_ROOT}/Hoang-Development/Reinforcement-Learning-with-CBM-for-Verified-Visual-COT/Visual-RFT/share_models/smoke_filtered10pct

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA_Quantization/chi/miniforge3/etc/profile.d/conda.sh
conda activate visual-rft
export LD_LIBRARY_PATH=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.conda/envs/visual-rft/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3

VIRFT_DIR=${SCHAIN_ROOT}/Hoang-Development/Reinforcement-Learning-with-CBM-for-Verified-Visual-COT/Visual-RFT/src/virft
REPO_ROOT=${SCHAIN_ROOT}/Hoang-Development/Reinforcement-Learning-with-CBM-for-Verified-Visual-COT/Visual-RFT

pip install -e "${VIRFT_DIR}" --no-deps -q
cd "${VIRFT_DIR}"

set +e
torchrun --nproc_per_node=4 \
    --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=12350 \
    src/open_r1/sft_schain.py \
    --output_dir         "${SAVE_SMOKE_HOANG}" \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name       "${REPO_ROOT}/share_data/schain_sft_filtered_10pct" \
    --deepspeed          ./local_scripts/zero2.json \
    --max_length 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_steps 10 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --logging_steps 1 \
    --save_steps 999999 \
    --report_to none \
    --seed 42 \
    --run_name smoke_filtered10pct \
    --dataloader_num_workers 2 \
    --remove_unused_columns false \
    > "${SMOKE_LOG}" 2>&1
HOANG_EXIT=$?
set -e

if [ ${HOANG_EXIT} -eq 0 ]; then
    echo "  [PASS] Hoang pipeline smoke test — no errors in 10 steps."
else
    echo "  [FAIL] Hoang pipeline crashed — see ${SMOKE_LOG}"
    echo "  Last 30 lines:"
    tail -30 "${SMOKE_LOG}"
fi
rm -rf "${SAVE_SMOKE_HOANG}"   # clean up smoke test checkpoint

# ── Step 4: smoke test — Dang's pipeline (filtered 10%) ──────────────────────
echo ""
echo "── Smoke test 2: Dang / Exgra-Med / filtered 10% (max_steps=10) ──"
SMOKE_LOG_DANG=${LOG_DIR}/smoke_dang_10pct.log
SAVE_SMOKE_DANG=${SCHAIN_ROOT}/runs/smoke_dang_filtered10pct/model

conda deactivate
conda activate schain
export LD_LIBRARY_PATH=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA_Quantization/chi/miniforge3/envs/schain/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p "$(dirname ${SAVE_SMOKE_DANG})"
cd "${EXGRA_MED_DIR}"

set +e
torchrun --nnodes=1 --nproc_per_node=4 --master_port=25060 \
    llava/train/train_mem_CoT.py \
    --model_name_or_path  ${SCHAIN_ROOT}/model_weights/exgra-med/ \
    --data_path           ${DATA_CBM}/filtered_10pct.json \
    --image_folder        ${SCHAIN_ROOT}/data/main/data/images \
    --vision_tower        ${SCHAIN_ROOT}/model_weights/clip-vit-large-patch14 \
    --mm_projector_type   mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_dense_connector_type none \
    --num_l 6 \
    --prompt_mode cot \
    --use_rag false \
    --bf16 True \
    --output_dir          ${SAVE_SMOKE_DANG} \
    --max_steps 10 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none \
    --seed 42 \
    > "${SMOKE_LOG_DANG}" 2>&1
DANG_EXIT=$?
set -e

if [ ${DANG_EXIT} -eq 0 ]; then
    echo "  [PASS] Dang pipeline smoke test — no errors in 10 steps."
else
    echo "  [FAIL] Dang pipeline crashed — see ${SMOKE_LOG_DANG}"
    echo "  Last 30 lines:"
    tail -30 "${SMOKE_LOG_DANG}"
fi
rm -rf "$(dirname ${SAVE_SMOKE_DANG})"   # clean up

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " POST-FILTER SUMMARY  [$(date)]"
echo ""
if [ ${HOANG_EXIT} -eq 0 ] && [ ${DANG_EXIT} -eq 0 ]; then
    echo "  Both pipelines PASSED smoke test."
    echo "  Safe to launch full training:"
    echo ""
    echo "  Hoang 10%:  bash ${VISUAL_RFT_SCRIPTS}/schain_qwen2vl_sft-cot_filtered10pct.sh"
    echo "  Hoang 40%:  bash ${VISUAL_RFT_SCRIPTS}/schain_qwen2vl_sft-cot_filtered40pct.sh"
    echo ""
    echo "  Dang 10%:   (from ${EXGRA_MED_DIR})"
    echo "              bash bashscript/hoang_training_script_filtered10pct_sft.sh"
    echo "  Dang 40%:   bash bashscript/hoang_training_script_filtered40pct_sft.sh"
else
    echo "  One or more smoke tests FAILED — fix before running full training."
    echo "  Hoang exit: ${HOANG_EXIT}   log: ${SMOKE_LOG}"
    echo "  Dang  exit: ${DANG_EXIT}    log: ${SMOKE_LOG_DANG}"
fi
echo "================================================================"
