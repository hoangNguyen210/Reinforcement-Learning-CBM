model_list=(
    Qwen/Qwen2.5-VL-3B-Instruct
    Qwen/Qwen2.5-VL-7B-Instruct
    # SFT
    MedVLThinker-3B-SFT_m23k
    MedVLThinker-7B-SFT_m23k
    MedVLThinker-3B-SFT_PMC
    MedVLThinker-7B-SFT_PMC
    MedVLThinker-3B-SFT_m23k-RL_PMC
    # RL
    MedVLThinker-3B-RL_m23k
    MedVLThinker-3B-RL_PMC
    MedVLThinker-3B-RL_m23k-RL_PMC
    MedVLThinker-7B-SFT_m23k-RL_PMC
    MedVLThinker-7B-RL_m23k
    MedVLThinker-7B-RL_PMC
    MedVLThinker-7B-RL_m23k-RL_PMC
)


# greedy inference
exp_name="greedy"
versions=(0 1 2)
temperature=0.0
n=1

for version in ${versions[@]}; do
    for model in "${model_list[@]}"; do
        echo "Running inference for model: $model"

        model_name=$(basename $model)
        python eval/run_offline_inference.py \
            --model $model \
            --dp_size 8 \
            --tp_size 1 \
            --seed 42 \
            --temperature ${temperature} \
            --max_tokens 4096 \
            --n ${n} \
            --output_dir outputs/${exp_name}/v${version}/$model_name
    done
done


model_list=(
    Qwen/Qwen2.5-VL-32B-Instruct
    MedVLThinker-32B-RL_m23k
)
# greedy inference
exp_name="greedy"
versions=(0 1 2)
temperature=0.0
n=1

for version in ${versions[@]}; do
    for model in "${model_list[@]}"; do
        echo "Running inference for model: $model"

        model_name=$(basename $model)
        python eval/run_offline_inference.py \
            --model $model \
            --dp_size 2 \
            --tp_size 4 \
            --seed 42 \
            --temperature ${temperature} \
            --max_tokens 4096 \
            --n ${n} \
            --output_dir outputs/${exp_name}/v${version}/$model_name
    done
done