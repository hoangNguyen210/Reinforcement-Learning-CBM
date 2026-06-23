input_dir=checkpoints/med-vlrm/
output_dir=outputs/converted/

args=(
    # ckpt dir name, step
    "train-qwen2_5_vl_3b-pmc_vqa-m23k_sft_epoch_3 1150"
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
