# 7b
model=Qwen/Qwen2.5-VL-7B-Instruct
dataset_name=UCSC-VLAA/MedVLThinker-m23k-tokenized
output_dir=outputs/estimate_pass_rate/qwen2_5-vl-7b-instruct-med-vlm-m23k
python eval/run_offline_inference.py \
    --model "${model}" \
    --output_dir "${output_dir}" \
    --dataset_name "${dataset_name}" \
    --split train \
    --batch_size 512 \
    --max_tokens 4096 \
    --n 16 \
    --temperature 1.0 \
    --dp_size 32 \
    --tp_size 1 \
    --node_size 4 \
    --node_rank 0
    # --overwrite \

model=Qwen/Qwen2.5-VL-7B-Instruct
dataset_name=UCSC-VLAA/MedVLThinker-pmc_vqa
output_dir=outputs/estimate_pass_rate/qwen2_5-vl-7b-instruct-med-vlm-pmc_vqa
python eval/run_offline_inference.py \
    --model "${model}" \
    --output_dir "${output_dir}" \
    --dataset_name "${dataset_name}" \
    --split train \
    --batch_size 512 \
    --max_tokens 4096 \
    --n 16 \
    --temperature 1.0 \
    --dp_size 32 \
    --tp_size 1 \
    --node_size 4 \
    --node_rank 0
    # --overwrite \


# 3b
model=Qwen/Qwen2.5-VL-3B-Instruct
dataset_name=UCSC-VLAA/MedVLThinker-m23k-tokenized
output_dir=outputs/estimate_pass_rate/qwen2_5-vl-3b-instruct-med-vlm-m23k
python eval/run_offline_inference.py \
    --model "${model}" \
    --output_dir "${output_dir}" \
    --dataset_name "${dataset_name}" \
    --split train \
    --batch_size 512 \
    --max_tokens 4096 \
    --n 16 \
    --temperature 1.0 \
    --dp_size 32 \
    --tp_size 1 \
    --node_size 4 \
    --node_rank 0
    # --overwrite \


model=Qwen/Qwen2.5-VL-3B-Instruct
dataset_name=UCSC-VLAA/MedVLThinker-pmc_vqa
output_dir=outputs/estimate_pass_rate/qwen2_5-vl-3b-instruct-med-vlm-pmc_vqa
python eval/run_offline_inference.py \
    --model "${model}" \
    --output_dir "${output_dir}" \
    --dataset_name "${dataset_name}" \
    --split train \
    --batch_size 512 \
    --max_tokens 4096 \
    --n 16 \
    --temperature 1.0 \
    --dp_size 32 \
    --tp_size 1 \
    --node_size 4 \
    --node_rank 0
    # --overwrite \



# 32b
model=Qwen/Qwen2.5-VL-32B-Instruct
dataset_name=UCSC-VLAA/MedVLThinker-m23k-tokenized
output_dir=outputs/estimate_pass_rate/qwen2_5-vl-32b-instruct-med-vlm-m23k
python eval/run_offline_inference.py \
    --model "${model}" \
    --output_dir "${output_dir}" \
    --dataset_name "${dataset_name}" \
    --split train \
    --batch_size 512 \
    --max_tokens 4096 \
    --n 16 \
    --temperature 1.0 \
    --dp_size 16 \
    --tp_size 2 \
    --node_size 4 \
    --node_rank 0
    # --overwrite \


model=Qwen/Qwen2.5-VL-32B-Instruct
dataset_name=UCSC-VLAA/MedVLThinker-pmc_vqa
output_dir=outputs/estimate_pass_rate/qwen2_5-vl-32b-instruct-med-vlm-pmc_vqa
python eval/run_offline_inference.py \
    --model "${model}" \
    --output_dir "${output_dir}" \
    --dataset_name "${dataset_name}" \
    --split train \
    --batch_size 512 \
    --max_tokens 4096 \
    --n 16 \
    --temperature 1.0 \
    --dp_size 16 \
    --tp_size 2 \
    --node_size 4 \
    --node_rank 0
    # --overwrite \
