#!/usr/bin/env bash
# Code-agent run for 4x5090 with vLLM rollout backend.

set -euo pipefail
set -x

export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

python -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name='gsm8k_multiturn_grpo' \
  algorithm.adv_estimator=grpo \
  data.return_raw_chat=True \
  data.train_files="$HOME/data/gsm8k_codeagent/train.parquet" \
  data.val_files="$HOME/data/gsm8k_codeagent/test.parquet" \
  data.train_batch_size=128 \
  data.max_prompt_length=1024 \
  data.max_response_length=768 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.rollout.agent.default_agent_loop=code_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/codeagent_agent_loop.yaml" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/codeagent_tool_config.yaml" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.test_freq=20 \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name='codeagent_rl' \
  trainer.experiment_name='codeagent_4x5090_vllm' \
  "$@"
