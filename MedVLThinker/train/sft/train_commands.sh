# qwen2.5-vl-3b-instruct
bash train/sft/sft_local.sh \
    --train_dataset_name UCSC-VLAA/MedVLThinker-m23k-tokenized \
    --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
    --epochs 5 \
    --gpu_count 8 \
    --output_dir outputs/sft-m23k \
    --exp_name 3b \
    --gradient_checkpointing False \
    --use_flash_attention_2 False


# qwen2.5-vl-7b-instruct
bash train/sft/sft_local.sh \
    --train_dataset_name UCSC-VLAA/MedVLThinker-m23k-tokenized \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --epochs 5 \
    --gpu_count 8 \
    --output_dir outputs/sft-m23k \
    --exp_name 7b \
    --gradient_checkpointing False \
    --use_flash_attention_2 False


# qwen2.5-vl-32b-instruct
bash train/sft/sft_local.sh \
    --train_dataset_name UCSC-VLAA/MedVLThinker-m23k-tokenized \
    --model_name "Qwen/Qwen2.5-VL-32B-Instruct" \
    --epochs 5 \
    --gpu_count 8 \
    --output_dir outputs/sft-m23k \
    --exp_name 32b \
    --gradient_checkpointing True \
    --use_flash_attention_2 False \
    --nnodes 2 \
    --head_node_ip ???
