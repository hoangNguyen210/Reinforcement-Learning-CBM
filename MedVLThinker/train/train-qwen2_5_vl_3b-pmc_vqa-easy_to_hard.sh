# qwen2.5 VL repo: https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5
# third_party/verl/examples/grpo_trainer/run_qwen2_5_vl-7b.sh
set -x

MODEL=Qwen/Qwen2.5-VL-3B-Instruct
ENGINE=vllm

set -a && source .env && set +a


# data/verl/med-vlm-eval-qwen2_5_vl_size/test.parquet 6,050
# data/verl/med-vlm-m23k-qwen2_5_vl_3b-easy_to_hard/train.parquet  23,456
# data/verl/med-vlm-pmc_vqa-qwen2_5_vl_3b-easy_to_hard/train.parquet  176,896

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/verl/med-vlm-pmc_vqa-qwen2_5_vl_3b-easy_to_hard/train.parquet \
    data.val_files=data/verl/med-vlm-eval-qwen2_5_vl_size/test.parquet \
    data.shuffle=False \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.filter_overlong_prompts=False \
    data.max_response_length=4096 \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='med-vlrm' \
    trainer.experiment_name='train-qwen2_5_vl_3b-pmc_vqa-easy_to_hard' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=2 \
    custom_reward_function.path=./train/my_reward.py $@