# verl 框架学习顺序指南

## 框架概述

**verl** (Volcano Engine Reinforcement Learning for LLMs) 是字节跳动 Seed 团队开发的大语言模型强化学习训练框架，实现了 **HybridFlow** 架构，支持 PPO、GRPO 等多种 RL 算法。

### 核心特性

- **训练后端**: FSDP、FSDP2、Megatron-LM
- **推理引擎**: vLLM、SGLang、HF Transformers
- **支持算法**: PPO、GRPO、RLOO、ReMax、REINFORCE++、DAPO 等
- **扩展性**: 支持 671B 模型，数百 GPU 集群

---

## 推荐学习顺序

### 第一阶段：概念理解（1-2天）

| 顺序 | 文件 | 说明 |
|------|------|------|
| 1 | `README.md` | 整体了解项目目标、特性、支持的算法 |
| 2 | `docs/start/quickstart.rst` | 快速理解 PPO 训练的完整流程 |
| 3 | 论文 [HybridFlow](https://arxiv.org/abs/2409.19256v2) | 理解核心架构设计理念 |

---

### 第二阶段：核心数据结构（2-3天）

| 顺序 | 文件 | 说明 |
|------|------|------|
| 4 | `verl/__init__.py` | 入口文件，了解主要导出 (`DataProto`) |
| 5 | **`verl/protocol.py`** ⭐ | **最重要的基础文件**，定义 `DataProto` - 所有组件间数据传输的标准协议 |
| 6 | `verl/base_config.py` | 配置基类 |

**重点理解**:
- `DataProto` 基于 `TensorDict`，是组件间传递数据的"粘合剂"
- 包含 tensor 数据和非 tensor 元数据
- 支持分布式 gather/scatter 操作

---

### 第三阶段：分布式控制器（3-5天）

这是理解 verl 如何管理分布式 Worker 的关键。

| 顺序 | 文件 | 说明 |
|------|------|------|
| 7 | `verl/single_controller/__init__.py` | 控制器模块入口 |
| 8 | `verl/single_controller/base/worker.py` | Worker 基类定义 |
| 9 | `verl/single_controller/base/worker_group.py` | WorkerGroup 抽象 - 管理一组分布式 Worker |
| 10 | `verl/single_controller/base/decorator.py` | `@register` 装饰器 - 暴露远程调用方法 |
| 11 | **`verl/single_controller/ray/base.py`** ⭐ | Ray 实现 - `RayWorkerGroup`, `RayResourcePool`, `RayClassWithInitArgs` |

**重点理解**:
- Single Controller 模式：Driver 节点调度所有 Worker
- `@register` 装饰器让 Worker 方法可被远程调用
- `RayResourcePool` 管理 GPU 资源分配

---

### 第四阶段：Worker 实现（5-7天）

这是实际计算逻辑所在。

| 顺序 | 文件 | 说明 |
|------|------|------|
| 12 | `verl/workers/__init__.py` | Workers 模块入口 |
| 13 | `verl/workers/actor/` 目录 | Actor Worker 实现 |
| 14 | `verl/workers/critic/` 目录 | Critic Worker 实现 |
| 15 | `verl/workers/rollout/` 目录 | Rollout 生成（vLLM/SGLang 集成） |
| 16 | `verl/workers/reward_model/` | 奖励模型 Worker |
| 17 | `verl/workers/reward_manager/` | 奖励计算管理 |
| 18 | **`verl/workers/fsdp_workers.py`** ⭐ | FSDP 后端的完整 Worker 实现（核心文件） |
| 19 | `verl/workers/megatron_workers.py` | Megatron 后端 Worker 实现 |
| 20 | `verl/workers/sharding_manager/` | 模型分片管理（3D-HybridEngine） |

**重点理解**:
- 每个 Role（Actor、Critic、Ref、Reward）都是独立 Worker
- Worker 可以 colocate（共享 GPU）或 split（分离部署）
- `sharding_manager` 实现训练/推理模式切换时的权重重分片

---

### 第五阶段：Trainer 训练逻辑（3-5天）

| 顺序 | 文件 | 说明 |
|------|------|------|
| 21 | `verl/trainer/__init__.py` | Trainer 模块入口 |
| 22 | `verl/trainer/main_ppo.py` | PPO 训练入口脚本 |
| 23 | **`verl/trainer/ppo/ray_trainer.py`** ⭐ | **核心文件** - `RayPPOTrainer` 训练编排器 |
| 24 | **`verl/trainer/ppo/core_algos.py`** ⭐ | PPO/GRPO 等 RL 算法核心实现 |
| 25 | `verl/trainer/ppo/reward.py` | 奖励计算逻辑 |
| 26 | `verl/trainer/ppo/metric_utils.py` | 指标计算 |
| 27 | `verl/trainer/config/` 目录 | 配置文件定义 |

**重点理解**:
- `RayPPOTrainer` 是整个训练流程的编排中心
- 训练循环：Generate → Compute Reward → Compute Advantage → Update Actor/Critic
- `core_algos.py` 包含 GAE、PPO loss、GRPO loss 等核心算法

---

### 第六阶段：模型与工具（2-3天）

| 顺序 | 文件 | 说明 |
|------|------|------|
| 28 | `verl/models/` 目录 | 模型定义（Llama, Qwen2 等） |
| 29 | `verl/utils/dataset/` | 数据集处理 |
| 30 | `verl/utils/reward_score/` | 奖励函数实现（gsm8k, math 等） |
| 31 | `verl/utils/checkpoint/` | 检查点保存/加载 |
| 32 | `verl/utils/fsdp_utils.py` | FSDP 工具函数 |
| 33 | `verl/utils/megatron_utils.py` | Megatron 工具函数 |

---

### 第七阶段：高级特性（可选）

| 顺序 | 文件 | 说明 |
|------|------|------|
| 34 | `verl/experimental/agent_loop/` | 多轮对话 Agent 训练 |
| 35 | `verl/experimental/fully_async_policy/` | 异步策略训练 |
| 36 | `verl/interactions/` | 交互式环境 |
| 37 | `verl/trainer/fsdp_sft_trainer.py` | SFT 训练器 |

---

## 核心文件优先级

如果时间有限，优先深入理解这 5 个文件：

| 优先级 | 文件 | 作用 |
|--------|------|------|
| ⭐⭐⭐ | `verl/protocol.py` | 数据协议，所有组件的粘合剂 |
| ⭐⭐⭐ | `verl/trainer/ppo/ray_trainer.py` | 训练流程编排 |
| ⭐⭐⭐ | `verl/trainer/ppo/core_algos.py` | RL 算法实现 |
| ⭐⭐ | `verl/single_controller/ray/base.py` | Ray 分布式管理 |
| ⭐⭐ | `verl/workers/fsdp_workers.py` | Worker 完整实现 |

---

## 架构图示

```
┌─────────────────────────────────────────────────────────────┐
│                     RayPPOTrainer                           │
│               (verl/trainer/ppo/ray_trainer.py)             │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Training Loop                       │   │
│  │  1. Generate rollouts                                │   │
│  │  2. Compute rewards                                  │   │
│  │  3. Compute advantages (GAE)                         │   │
│  │  4. Update Actor & Critic                            │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │ 编排调度
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  ActorRollout │  │    Critic     │  │ RewardModel   │
│    Worker     │  │    Worker     │  │   Worker      │
│               │  │               │  │               │
│ - Generation  │  │ - Value est.  │  │ - Scoring     │
│ - Training    │  │ - Training    │  │               │
│ - Ref logprob │  │               │  │               │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │    verl/protocol.py     │
              │       DataProto         │
              │                         │
              │  - TensorDict wrapper   │
              │  - Batch metadata       │
              │  - Distributed ops      │
              └─────────────────────────┘
```

---

## 目录结构速查

```
verl/
├── __init__.py              # 入口，导出 DataProto
├── protocol.py              # ⭐ 核心数据协议
├── base_config.py           # 配置基类
│
├── single_controller/       # 分布式控制器
│   ├── base/
│   │   ├── worker.py        # Worker 基类
│   │   ├── worker_group.py  # WorkerGroup 抽象
│   │   └── decorator.py     # @register 装饰器
│   └── ray/
│       └── base.py          # ⭐ Ray 实现
│
├── workers/                 # Worker 实现
│   ├── actor/               # Actor Worker
│   ├── critic/              # Critic Worker
│   ├── rollout/             # Rollout 生成
│   ├── reward_model/        # 奖励模型
│   ├── reward_manager/      # 奖励管理
│   ├── sharding_manager/    # 分片管理
│   ├── fsdp_workers.py      # ⭐ FSDP 后端
│   └── megatron_workers.py  # Megatron 后端
│
├── trainer/                 # 训练器
│   ├── main_ppo.py          # PPO 入口
│   ├── ppo/
│   │   ├── ray_trainer.py   # ⭐ RayPPOTrainer
│   │   ├── core_algos.py    # ⭐ RL 算法实现
│   │   ├── reward.py        # 奖励计算
│   │   └── metric_utils.py  # 指标工具
│   ├── fsdp_sft_trainer.py  # SFT 训练器
│   └── config/              # 配置定义
│
├── models/                  # 模型定义
│   ├── llama/
│   ├── qwen2/
│   ├── mcore/               # Megatron-Core
│   └── transformers/        # HF Transformers
│
├── utils/                   # 工具函数
│   ├── dataset/             # 数据集处理
│   ├── reward_score/        # 奖励函数（gsm8k, math）
│   ├── checkpoint/          # 检查点管理
│   ├── fsdp_utils.py
│   ├── megatron_utils.py
│   └── ...
│
└── experimental/            # 实验性功能
    ├── agent_loop/          # Agent 多轮训练
    ├── fully_async_policy/  # 异步策略
    └── ...
```

---

## 学习建议

### 1. 先跑通示例

```bash
# 准备数据
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

# 下载模型
python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"

# 运行 PPO 训练
python3 -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.n_gpus_per_node=1
```

### 2. 打断点调试

在 `ray_trainer.py` 的训练循环中打断点，跟踪数据流：

```python
# verl/trainer/ppo/ray_trainer.py 中的关键位置
def fit(self):
    for epoch in range(self.config.trainer.total_epochs):
        # 断点 1: 生成 rollout
        batch = self.actor_rollout.generate_sequences(...)
        
        # 断点 2: 计算奖励
        batch = self.reward_model.compute_reward(batch)
        
        # 断点 3: 计算优势
        batch = compute_advantage(batch, ...)
        
        # 断点 4: 更新模型
        self.actor.update(batch)
        self.critic.update(batch)
```

### 3. 画调用图

记录 `DataProto` 在各组件间如何传递：

```
Prompt DataProto
    ↓ (actor_rollout.generate)
Response DataProto (+ tokens, log_probs)
    ↓ (reward_model.compute)
Reward DataProto (+ scores)
    ↓ (compute_advantage)
Advantage DataProto (+ advantages, returns)
    ↓ (actor/critic.update)
Updated Models
```

### 4. 对比算法

比较 PPO 和 GRPO 在 `core_algos.py` 中的实现差异：

- PPO: Actor-Critic 架构，需要 Value function
- GRPO: 无 Critic，使用 group-relative baseline

---

## 推荐阅读资料

1. **官方文档**: https://verl.readthedocs.io/
2. **HybridFlow 论文**: https://arxiv.org/abs/2409.19256v2
3. **社区博客**:
   - [HybridFlow verl 原文浅析](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/readme.md)
   - [verl x SGLang Multi-turn Code Walkthrough](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme_EN.md)

---

## 常见问题

### Q: verl 和 OpenRLHF、DeepSpeed-Chat 有什么区别？

verl 的核心优势是 **HybridFlow** 架构：
- 解耦计算和数据依赖
- 灵活的设备映射（colocate/split placement）
- 3D-HybridEngine 减少模型重分片开销

### Q: 如何添加新的 RL 算法？

1. 在 `verl/trainer/ppo/core_algos.py` 中实现算法核心逻辑
2. 在 `verl/trainer/ppo/ray_trainer.py` 中添加训练流程
3. 参考 `examples/grpo_trainer/` 的实现

### Q: 如何添加新的模型？

参考文档：
- FSDP 后端: `docs/advance/fsdp_extension.rst`
- Megatron 后端: `docs/advance/megatron_extension.rst`

---

*文档生成时间: 2026-01-29*
