# Data

## Data Format

Example: https://huggingface.co/datasets/med-vlrm/med-vlm-eval

The fields:
```python
from dataclasses import dataclass
from datasets import Image, Sequence
from typing import Optional, Dict

@dataclass
class DataFormat:
    images: Optional[Sequence[Image]]
    question: str
    options: Dict[str, str]
    answer_label: str
    answer: str
    reasoning: Optional[str]
    dataset_name: str
    dataset_index: int
    # unused
    hash: str
    misc: Optional[Dict[str, object]]
```

To save and load the `options` as a `dict`:
```python
"""

"""

# save
dict(
    "options": json.dumps(letter_options, ensure_ascii=False)
)

# load
options = json.loads(options)
```


The `dataset_index` is used to locate and differentiate each sample.


To compute `hash`:
```python
from hashlib import sha256

def get_str_hash(input_string: str) -> str:
    return sha256(input_string.encode()).hexdigest()[:16]

get_str_hash(f"{question}{answer}")
```

## Prepare Eval datasets

The dataset has been uploaded to `med-vlrm/med-vlm-eval`.

```bash
python data_process/prepare_vlm_eval_data.py
```

## (old) Prepare PMC-VQA

```bash
python scripts/resize_filter_pmc_vqa_easyr1.py \
    --local_dir data/processed/pmc_vqa_limit_tokens_2048 \
    --max_image_size 1024 \
    --max_token_length 2048 \
    --num_proc 32

python scripts/convert_pmc_vqa_easyr1_to_verl.py \
    --data_source data/processed/pmc_vqa_limit_tokens_2048 \
    --is_local \
    --local_dir data/verl/pmc_vqa_limit_tokens_2048 \
    --num_proc 32 \
    --num_samples 20000
```


### (old) To build data from scratch

```bash
python scripts/convert_pmc_vqa_to_easyr1.py
```


## Prepare pass rate, filter by pass rate and order to easy-to-hard

Estimate pass rate by rollout 16 times. termperature = 1.0

```bash
eval/estimate_pass_rate.sh
```

Then merge the results and get stats

```bash
python eval/merge_results.py -d [dir]
python eval/gather_stats.py -d [dir]
python eval/analyze_pass_rate.py -d [dir]
```

upload the pass rate results to hf:

```bash
# upload
REPO_TYPE=dataset # model, dataset
NUM_WORKERS=8

LOCAL_DIR=outputs/estimate_pass_rate
REPO_URL=med-vlrm/estimate_pass_rate

huggingface-cli upload-large-folder "$REPO_URL" --repo-type="${REPO_TYPE}" "$LOCAL_DIR" --num-workers="${NUM_WORKERS}"

# download
REPO_TYPE=dataset # model, dataset
LOCAL_DIR=outputs/estimate_pass_rate

REPO_URL=med-vlrm/estimate_pass_rate

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
```

Build the dataset.

We filter those samples pass rate <1 and >7

```bash
python data_process/train_dataset/order_easy_to_hard.py \
    --dataset_name med-vlrm/med-vlm-m23k \
    --split train \
    --results_jsonl_path outputs/estimate_pass_rate/qwen2_5-vl-3b-instruct-med-vlm-m23k/eval_results.jsonl \
    --save_to_disk_path data/local/med-vlm-m23k-qwen2_5_vl_3b-easy_to_hard \
    --num_proc 16


python data_process/train_dataset/order_easy_to_hard.py \
    --dataset_name med-vlrm/med-vlm-pmc_vqa \
    --split train \
    --results_jsonl_path outputs/estimate_pass_rate/qwen2_5-vl-3b-instruct-med-vlm-pmc_vqa/eval_results.jsonl \
    --save_to_disk_path data/local/med-vlm-pmc_vqa-qwen2_5_vl_3b-easy_to_hard \
    --num_proc 16 \
    --keep_in_memory False


python data_process/train_dataset/order_easy_to_hard.py \
    --dataset_name med-vlrm/med-vlm-pmc_vqa-gpt_4o_reasoning \
    --split train \
    --results_jsonl_path outputs/estimate_pass_rate/qwen2_5-vl-3b-instruct-med-vlm-pmc_vqa/eval_results.jsonl \
    --save_to_disk_path data/local/med-vlm-pmc_vqa-gpt_4o_reasoning-qwen2_5_vl_3b-easy_to_hard \
    --num_proc 16 \
    --keep_in_memory False
```

Check pass rate and token length distribution

```bash
python data_process/train_dataset/check_pass_rate_dataset.py \
    -d data/local/med-vlm-m23k-qwen2_5_vl_3b-easy_to_hard \
    -n 64

python data_process/train_dataset/check_pass_rate_dataset.py \
    -d data/local/med-vlm-pmc_vqa-qwen2_5_vl_3b-easy_to_hard \
    -n 64
```

## convert dataset to target format

```
data_process/train_dataset/convert_trl_format.py
data_process/train_dataset/convert_verl_format.py
```

## Prepare cot for pmc-vqa

Generate cot for pmc-vqa with gpt 4o
```bash
bash data_process/train_dataset/pmc_vqa-gpt_4o_cot.sh
```

Upload and download

```bash
REPO_TYPE=dataset # model, dataset
NUM_WORKERS=8

PATH_TO_DATA_DIR=outputs/med-vlm-pmc_vqa-gpt-4o-cot
REPO_NAME=med-vlrm/med-vlm-pmc_vqa-gpt-4o-cot

huggingface-cli upload-large-folder "$REPO_NAME" --repo-type="${REPO_TYPE}" "$PATH_TO_DATA_DIR" --num-workers="${NUM_WORKERS}"


huggingface-cli download --repo-type $REPO_TYPE --local-dir $PATH_TO_DATA_DIR ${REPO_NAME}
```

Merge cot to the dataset

```bash
```


