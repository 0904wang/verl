# verl 学习路线图

`verl` (Volcano Engine Reinforcement Learning) 是一个专为大语言模型 (LLM) 设计的高性能强化学习 (RL) 训练库。它基于 Ray 和 PyTorch 构建，支持 HybridFlow 编程模型，能够灵活高效地进行 PPO、GRPO 等算法的训练。

以下是为您定制的逐文件学习路线，分为 **入门准备**、**核心流程**、**关键组件**、**进阶架构** 四个阶段。

---

## 📅 第一阶段：入门与快速上手 (Quick Start)
**目标**：跑通一个 Demo，建立感性认识。

1.  **全局概览**
    *   **阅读文件**: `README.md`
    *   **重点**: 了解 Project 是做什么的 (RL specifically for LLMs)，核心特性 (Modular APIs, HybridFlow, SOTA throughout)，以及支持的算法 (PPO, GRPO)。

2.  **运行示例 (Hello World)**
    *   **阅读文件**: `examples/ppo_trainer/README.md`
    *   **查看脚本**: `examples/ppo_trainer/run_deepseek7b_llm.sh` (或其他类似脚本)
    *   **行动**: 按照 `docs` 中的安装说明配置环境，并尝试运行这个脚本。
    *   **思考**: 脚本中传入了哪些参数？(比如 `actor_rollout_ref.model.path`, `data.train_files`)，这一步让你知道如何配置训练任务。

---

## ⚙️ 第二阶段：核心训练流程 (Core Flow)
**目标**：理解代码是如何运行起来的，数据如何在系统中流转。

3.  **入口分析**
    *   **阅读文件**: `verl/trainer/main_ppo.py`
    *   **重点**:
        *   `main` 函数：使用 `hydra` 管理配置。
        *   `run_ppo` 函数：初始化 Ray 集群。
        *   `TaskRunner` 类：这是核心指挥官，负责组装 Actor, Critic, RewardModel 等 Worker。
    *   **收获**: 理解系统是如何启动的，以及各个角色的创建过程。

4.  **训练主循环**
    *   **阅读文件**: `verl/trainer/ppo/ray_trainer.py`
    *   **重点关注类**: `RayPPOTrainer`
    *   **重点方法**:
        *   `fit()`: 训练的主循环 (Loop)。
        *   `_create_dataloader`: 数据加载。
        *   `run_generation`: 生成 Rollout 阶段。
        *   `update_policy`: PPO 更新阶段。
    *   **收获**: 理解 RLHF 的训练心跳：**Generate (Rollout) -> Compute Reward -> Compute Advantage (GAE/GRPO) -> Update Actor/Critic**。

---

## 🛠️ 第三阶段：关键定制组件 (Customization)
**目标**：学会如何替换数据、修改 Reward 函数，这是实际使用中最常修改的部分。

5.  **数据处理**
    *   **阅读文件**: `verl/utils/dataset/rl_dataset.py`
    *   **重点**: `RLDataset` 类，了解数据是如何被 Tokenizer 处理并转换成 Tensor 的。
    *   **思考**: 如果我有自己的数据格式，应该如何修改这里？

6.  **奖励函数 (Reward Function)**
    *   **阅读文件**: `verl/utils/reward_score/gsm8k.py` (以数学任务为例)
    *   **重点**: `compute_score` 函数。
    *   **目录浏览**: `verl/utils/reward_score/` 下的其他文件，看看如何编写自定义规则的 Reward。

7.  **算法配置**
    *   **阅读文件**: `verl/trainer/ppo/core_algos.py`
    *   **重点**: PPO 的核心数学实现，如 `compute_gae_advantage_return`, `clip_loss` 等。如果你想修改算法细节（如引入新的 Loss），这里是必看之地。

---

## 🏗️ 第四阶段：进阶架构与底层 (Advanced Architecture)
**目标**：理解 HybridFlow 的精髓，以及分布式是如何实现的。

8.  **Worker 实现**
    *   **阅读目录**: `verl/workers/`
    *   **重点文件**:
        *   `verl/workers/rollout/vllm_rollout.py`: 结合 vLLM 进行高效推理生成。
        *   `verl/workers/fsdp_workers.py` 或 `megatron_workers.py`: 如何用 FSDP/Megatron 进行模型训练。
    *   **收获**: 理解 `verl` 如何将 Inference (vLLM) 和 Training (PyTorch FSDP) 结合在一起。

9.  **Controller 与 Ray**
    *   **阅读目录**: `verl/single_controller/`
    *   **重点**: 这里封装了对 Ray 的调用，实现了所谓的 "Single Controller" 模式，即一个主控节点调度多个 Ray Worker。

---

## 🗺️ 总结建议

建议您的阅读顺序：
1.  **Usage Level**: `examples/ppo_trainer/run_*.sh` -> `examples/ppo_trainer/config/*.yaml`
2.  **Logic Level**: `verl/trainer/main_ppo.py` -> `verl/trainer/ppo/ray_trainer.py`
3.  **Component Level**: `verl/utils/reward_score/*.py` -> `verl/utils/dataset/*.py`
4.  **System Level**: `verl/workers/*.py`
