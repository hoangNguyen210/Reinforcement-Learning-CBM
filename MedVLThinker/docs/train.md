# Train

Download models

```bash
set -a && source .env && set +a

python3 -c "import transformers; transformers.pipeline(model='Qwen/Qwen2.5-VL-3B-Instruct')"
python3 -c "import transformers; transformers.pipeline(model='Qwen/Qwen2.5-VL-7B-Instruct')"
```

## pmc vqa

```bash
set -a && source .env && set +a


bash train/run_qwen2_5_lv-3b.sh \
    data.train_files=data/verl/pmc_vqa_limit_tokens_2048/train.parquet \
    data.val_files=data/verl/pmc_vqa_limit_tokens_2048/test.parquet \
    data.max_prompt_length=2048 \
    data.filter_overlong_prompts=False \
    data.train_batch_size=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    trainer.save_freq=100 \
    trainer.test_freq=100000 \
    trainer.total_epochs=1

bash train/run_qwen2_5_lv-3b.sh \
    data.train_files=data/verl/pmc_vqa_limit_tokens_2048/train.parquet \
    data.val_files=data/verl/pmc_vqa_limit_tokens_2048/test.parquet \
    data.max_prompt_length=2048 \
    data.filter_overlong_prompts=False \
    data.train_batch_size=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    trainer.save_freq=200 \
    trainer.test_freq=400 \
    trainer.total_epochs=2 \
    trainer.project_name='verl_grpo_pmc_vqa' \
    trainer.experiment_name='qwen2_5_vl_3b_function_rm-bs_64'

# even long epochs
bash train/run_qwen2_5_lv-3b.sh \
    data.train_files=data/verl/pmc_vqa_limit_tokens_2048/train.parquet \
    data.val_files=data/verl/pmc_vqa_limit_tokens_2048/test.parquet \
    data.max_prompt_length=2048 \
    data.filter_overlong_prompts=False \
    data.train_batch_size=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    trainer.save_freq=200 \
    trainer.test_freq=400 \
    trainer.total_epochs=12 \
    trainer.project_name='verl_grpo_pmc_vqa' \
    trainer.experiment_name='qwen2_5_vl_3b_function_rm-bs_64-epoch_12' \
    trainer.resume_mode='resume_path' \
    trainer.resume_from_path='checkpoints/verl_grpo_pmc_vqa/qwen2_5_vl_3b_function_rm-bs_64/global_step_400'


bash train/run_qwen2_5_lv-7b.sh \
    data.train_files=data/verl/pmc_vqa_limit_tokens_2048/train.parquet \
    data.val_files=data/verl/pmc_vqa_limit_tokens_2048/test.parquet \
    data.max_prompt_length=2048 \
    data.filter_overlong_prompts=False \
    data.train_batch_size=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    trainer.save_freq=100 \
    trainer.test_freq=100000 \
    trainer.total_epochs=1


# rollout 50-60 gb, update up to 73 gb
bash train/run_qwen2_5_lv-7b.sh \
    data.train_files=data/verl/pmc_vqa_limit_tokens_2048/train.parquet \
    data.val_files=data/verl/pmc_vqa_limit_tokens_2048/test.parquet \
    data.max_prompt_length=2048 \
    data.filter_overlong_prompts=False \
    data.train_batch_size=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    trainer.save_freq=200 \
    trainer.test_freq=400 \
    trainer.total_epochs=2 \
    trainer.project_name='verl_grpo_pmc_vqa' \
    trainer.experiment_name='qwen2_5_vl_7b_function_rm-bs_64'

# Continue training for 2 more epochs
# NOTE: This is unusual approach.
# usually we need to first merge the ckpt, and load it via `actor_rollout_ref.model.path`
bash train/run_qwen2_5_lv-7b.sh \
    data.train_files=data/verl/pmc_vqa_limit_tokens_2048/train.parquet \
    data.val_files=data/verl/pmc_vqa_limit_tokens_2048/test.parquet \
    data.max_prompt_length=2048 \
    data.filter_overlong_prompts=False \
    data.train_batch_size=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    trainer.save_freq=200 \
    trainer.test_freq=400 \
    trainer.total_epochs=4 \
    trainer.project_name='verl_grpo_pmc_vqa' \
    trainer.experiment_name='qwen2_5_vl_7b_function_rm-bs_64' \
    trainer.resume_mode='resume_path' \
    trainer.resume_from_path='checkpoints/verl_grpo_pmc_vqa/qwen2_5_vl_7b_function_rm-bs_64/global_step_562'

# even long epochs
bash train/run_qwen2_5_lv-7b.sh \
    data.train_files=data/verl/pmc_vqa_limit_tokens_2048/train.parquet \
    data.val_files=data/verl/pmc_vqa_limit_tokens_2048/test.parquet \
    data.max_prompt_length=2048 \
    data.filter_overlong_prompts=False \
    data.train_batch_size=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    trainer.save_freq=200 \
    trainer.test_freq=400 \
    trainer.total_epochs=12 \
    trainer.project_name='verl_grpo_pmc_vqa' \
    trainer.experiment_name='qwen2_5_vl_7b_function_rm-bs_64-epoch_12' \
    trainer.resume_mode='resume_path' \
    trainer.resume_from_path='checkpoints/verl_grpo_pmc_vqa/qwen2_5_vl_7b_function_rm-bs_64/global_step_1124'
```


## SFT

Use `train/sft/convert_weights` to convert trl saved llm ckpt to vl model.