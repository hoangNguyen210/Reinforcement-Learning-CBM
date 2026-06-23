# eval


## Convert checkpoints

```bash
python third_party/verl/scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir checkpoints/verl_grpo_pmc_vqa/qwen2_5_vl_7b_function_rm-bs_64/global_step_562/actor \
    --target_dir outputs/converted/qwen2_5_vl_7b_function_rm-bs_64/global_step_562
```

```bash
# xk-jump-aws-a100-1
input_dir=/opt/dlami/nvme/xhuan192/codes/med-vlrm/checkpoints/med-vlrm/
output_dir=/opt/dlami/nvme/xhuan192/codes/med-vlrm/outputs/converted/

args=(
    "train-qwen2_5_vl_3b-pmc_vqa-m23k_sft_epoch_3 1150"
    "train-qwen2_5_vl_7b-pmc_vqa-m23k_rl 900"
)

for item in "${args[@]}"; do
    read -r ckpt_name step <<< "$item"
    echo "Processing checkpoint: $ckpt_name at step: $step"

    ls $input_dir/$ckpt_name/global_step_$step/
    if [[ ! -d $input_dir/$ckpt_name/global_step_$step/actor ]]; then
        echo "Directory $input_dir/$ckpt_name/global_step_$step/actor does not exist. Skipping."
        continue
    fi

    python third_party/verl/scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir $input_dir/$ckpt_name/global_step_$step/actor \
    --target_dir $output_dir/$ckpt_name-step_$step
done
```


## Upload

```bash
REPO_TYPE=model # model, dataset
NUM_WORKERS=8

PATH_TO_DATA_DIR=checkpoints/converted/verl_grpo_pmc_vqa/qwen2_5_vl_7b_function_rm-bs_64/global_step_562
REPO_NAME=med-vlrm/grpo-pmc_vqa-qwen2_5_vl_7b-bs_64-step_562

huggingface-cli upload-large-folder "$REPO_NAME" --repo-type="${REPO_TYPE}" "$PATH_TO_DATA_DIR" --num-workers="${NUM_WORKERS}"
```

## Download

```bash
REPO_TYPE=model # model, dataset
LOCAL_DIR=checkpoints/download/grpo-pmc_vqa-qwen2_5_vl_7b-bs_64-step_562

REPO_URL=med-vlrm/grpo-pmc_vqa-qwen2_5_vl_7b-bs_64-step_562

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
```


## Eval

```bash
python eval/run_offline_inference.py --model Qwen/Qwen2.5-VL-7B-Instruct
```
## (Duplicated) Eval

`eval/run_inference_parallel.py` uses `ThreadPoolExecutor`, if you use data-parallel, it causes `vllm` engine crushed.
However, even with 1 GPUs, this script is way faster (~2min).

While using 8 data parallel size with `eval/run_inference_by_batch.py`, it costs about 10 minutes, and only the first GPUs are running.

```bash
# on 24GB
python -m vllm.entrypoints.openai.api_server \
    --model checkpoints/download/grpo-pmc_vqa-qwen2_5_vl_7b-bs_64-step_562 \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --port 8000 \
    --max_model_len=100000 \
    --force-eager


python eval/run_inference_by_batch.py \
    --dataset-name AdaptLLM/biomed-VQA-benchmark \
    --subset PMC-VQA \
    --split test \
    --base-url http://localhost:8000/v1 \
    --model checkpoints/download/grpo-pmc_vqa-qwen2_5_vl_7b-bs_64-step_562 \
    --out-file outputs/preds_pmc_test.json \
    --batch-size 64


python eval/run_inference_parallel.py \
    --dataset-name AdaptLLM/biomed-VQA-benchmark \
    --subset PMC-VQA \
    --split test \
    --base-url http://localhost:8000/v1 \
    --model checkpoints/download/grpo-pmc_vqa-qwen2_5_vl_7b-bs_64-step_562 \
    --out-file outputs/preds_pmc_test.json \
    --num-workers 64
```


Qwen
```bash
set -a && source .env && set +a

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --port 8000 \
    --enforce-eager \
    --max_model_len=100000

python eval/run_inference_by_batch.py \
    --dataset-name AdaptLLM/biomed-VQA-benchmark \
    --subset PMC-VQA \
    --split test \
    --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --out-file outputs/preds_pmc_test-qwen2_5_vl_7b.json \
    --batch-size 64


python eval/run_inference_parallel.py \
    --dataset-name AdaptLLM/biomed-VQA-benchmark \
    --subset PMC-VQA \
    --split test \
    --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --out-file outputs/preds_pmc_test.json \
    --num-workers 64
```