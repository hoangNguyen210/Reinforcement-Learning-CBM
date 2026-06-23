
model_list=(
    # others
    Eren-Senoglu/llava-med-v1.5-mistral-7b-hf
)
# greedy inference
exp_name="greedy_llava_med"
versions=(0 1 2)
temperature=0.0
n=1

for version in ${versions[@]}; do
    for model in "${model_list[@]}"; do
        echo "Running inference for model: $model"

        model_name=$(basename $model)
        python eval/run_offline_inference_v2.py \
            --model $model \
            --dp_size 8 \
            --tp_size 1 \
            --seed 42 \
            --temperature ${temperature} \
            --max_tokens 4096 \
            --n ${n} \
            --batch_size 128 \
            --output_dir outputs/${exp_name}/v${version}/$model_name \
            --instruction_prompt 'You will solve a problem/request. First think step by step, and then answer with the option letter from the given choices directly after "Answer: "' \
            --enforce_eager --overwrite
    done
done


# Huatuo-Vision
# Check here: https://github.com/xk-huang/HuatuoGPT-Vision


model_list=(
    # others
    FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL
)
# greedy inference
exp_name="greedy_llava_med"
versions=(0 1 2)
temperature=0.0
n=1

for version in ${versions[@]}; do
    for model in "${model_list[@]}"; do
        echo "Running inference for model: $model"

        model_name=$(basename $model)
        python eval/run_offline_inference_v2.py \
            --model $model \
            --dp_size 8 \
            --tp_size 1 \
            --seed 42 \
            --temperature ${temperature} \
            --max_tokens 4096 \
            --n ${n} \
            --batch_size 128 \
            --output_dir outputs/${exp_name}/v${version}/$model_name \
            --instruction_prompt 'You will solve a problem/request. First think step by step, and then answer with the option letter from the given choices directly after "Answer: "' \
            --enforce_eager --overwrite
    done
done


# close models
export AZURE_API_KEY=???
export AZURE_API_BASE="https://openai-vlaa-westus3.openai.azure.com"
export AZURE_API_VERSION="2024-12-01-preview"

model_list=(
    azure/gpt-4o-1120-nofilter-global
    azure/gpt-4o-mini-20240718-nofilter-global
)

exp_name="greedy-closed_models"
versions=(0)
temperature=0.0
n=1

for version in ${versions[@]}; do
    for model in "${model_list[@]}"; do
        echo "Running inference for model: $model"

        model_name=$(basename $model)

        AZURE_OPENAI_MODEL=$model \
        python eval/run_api_inference.py \
        --dp_size 4 \
        --batch_size 64 \
        --seed 42 \
        --output_dir outputs/${exp_name}/v${version}/$model_name \
        --n 1 \
        2>&1 | tee misc/api_log.log
        # --dataset_size 4 \

    done
done


model_list=(
    # others
    FreedomIntelligence/HuatuoGPT-Vision-34B-hf
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
        python eval/run_offline_inference_v2.py \
            --model $model \
            --dp_size 2 \
            --tp_size 4 \
            --seed 42 \
            --temperature ${temperature} \
            --max_tokens 4096 \
            --n ${n} \
            --batch_size 128 \
            --output_dir outputs/${exp_name}/v${version}/$model_name
    done
done


model_list=(
    # others
    google/medgemma-27b-it  # add model_prompt_type="gemma3" to the command line args
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
        python eval/run_offline_inference_v2.py \
            --model $model \
            --dp_size 2 \
            --tp_size 4 \
            --seed 42 \
            --temperature ${temperature} \
            --max_tokens 4096 \
            --n ${n} \
            --batch_size 128 \
            --output_dir outputs/${exp_name}/v${version}/$model_name \
            --model_prompt_type "gemma3"
    done
done


model_list=(
    # others
    google/gemma-3-4b-it  # add model_prompt_type="gemma3" to the command line args
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
        python eval/run_offline_inference_v2.py \
            --model $model \
            --dp_size 8 \
            --tp_size 1 \
            --seed 42 \
            --temperature ${temperature} \
            --max_tokens 4096 \
            --n ${n} \
            --batch_size 128 \
            --output_dir outputs/${exp_name}/v${version}/$model_name \
            --model_prompt_type "gemma3"
    done
done


model_list=(
    # others
    google/gemma-3-27b-it  # add model_prompt_type="gemma3" to the command line args
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
        python eval/run_offline_inference_v2.py \
            --model $model \
            --dp_size 2 \
            --tp_size 4 \
            --seed 42 \
            --temperature ${temperature} \
            --max_tokens 4096 \
            --n ${n} \
            --batch_size 128 \
            --output_dir outputs/${exp_name}/v${version}/$model_name \
            --model_prompt_type "gemma3"
    done
done



