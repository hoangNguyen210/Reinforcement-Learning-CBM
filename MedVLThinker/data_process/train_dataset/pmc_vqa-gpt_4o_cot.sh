# 7b
model=gpt-4o
dataset_name=med-vlrm/med-vlm-pmc_vqa
output_dir=outputs/med-vlm-pmc_vqa-gpt-4o-cot
python eval/run_api_inference.py \
    --model "${model}" \
    --output_dir "${output_dir}" \
    --dataset_name "${dataset_name}" \
    --split train \
    --batch_size 32 \
    --max_tokens 4096 \
    --n 3 \
    --temperature 1.0 \
    --dp_size 4 \
    --tp_size 1 \
    --shuffle

python eval/run_api_inference_single.py \
    --config_path outputs/med-vlm-pmc_vqa-gpt-4o-cot/args.json \
    --local_dp_rank 0 \
    --global_dp_rank 0


python eval/run_api_inference_single.py \
    --config_path outputs/med-vlm-pmc_vqa-gpt-4o-cot/args.json \
    --local_dp_rank 1 \
    --global_dp_rank 1 


python eval/run_api_inference_single.py \
    --config_path outputs/med-vlm-pmc_vqa-gpt-4o-cot/args.json \
    --local_dp_rank 3 \
    --global_dp_rank 3