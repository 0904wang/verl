# 4x5090 CodeAgent Minimal Runbook

## Phase 0: Environment and resource precheck

```bash
bash examples/sglang_multiturn/run_env_precheck_4x5090.sh
```

Recommended:

- Python `3.12`
- CUDA `>=12.8`
- FSDP-first setup:

```bash
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

## Phase 1: Baseline (`tool_agent`)

```bash
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_4x5090.sh
```

## Phase 2: Main project (`code_agent`)

1. Prepare dataset with `agent_name=code_agent`:

```bash
python examples/data_preprocess/gsm8k_codeagent_agent_loop.py --local_save_dir ~/data/gsm8k_codeagent
```

2. Fill your sandbox endpoint in:

- `examples/sglang_multiturn/config/tool_config/codeagent_tool_config.yaml`
  - `sandbox_fusion_url`

3. Start training:

```bash
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_codeagent_4x5090.sh
```

## OOM / instability fallback order

1. `actor_rollout_ref.rollout.n: 4 -> 2 -> 1`
2. `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: 8 -> 4 -> 2`
3. `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu: 8 -> 4 -> 2`
4. `data.max_response_length: 768 -> 512`
5. `actor_rollout_ref.rollout.gpu_memory_utilization: 0.35 -> 0.30`

## First-round acceptance metrics

- `tool_call_success_rate/mean`
- task `reward` aggregate (`val/test_score/*`)
- `timing_s/agent_loop/tool_calls/mean`
- `timing_s/agent_loop/generate_sequences/mean`
- `response/aborted_ratio`
