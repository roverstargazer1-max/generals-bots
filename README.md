# Generals Bots

`generals-bots` 是一个基于 JAX 的 Generals.io 双人对战模拟器与 bot 实验框架。项目目标是提供可复现、可批量并行、适合强化学习研究的游戏环境。

更完整的中文手册见 [docs/zh-manual.md](docs/zh-manual.md)。

## 项目概览

- `generals/core/`：核心游戏逻辑、环境包装、动作、观测、地图生成和奖励函数。
- `generals/agents/`：内置 agent，包括 `RandomAgent` 和 `ExpanderAgent`。
- `generals/gui/`：pygame 可视化和 replay GUI。
- `generals/remote/`：连接 generals.io 远程服务的客户端代码。
- `examples/`：单局、向量化、可视化示例。
- `examples/_experimental/ppo/`：实验性 PPO、行为克隆、策略评估和策略可视化工具。
- `tests/`：pytest 测试。
- `docs/`：中文手册和开发记录。

## 快速搭建

推荐使用 `uv` 按锁文件安装依赖。项目要求 Python 3.11 或更高版本。

```bash
git clone https://github.com/CodeBoy2006/generals-bots.git
cd generals-bots
uv sync --extra dev
```

确认包和 JAX 后端可用：

```bash
uv run python -c "import generals; print(generals.GeneralsEnv)"
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
```

CUDA 13 环境可安装 GPU extra：

```bash
uv sync --extra dev --extra cuda13
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
```

## 核心接口

`GeneralsEnv` 是主要环境入口。当前接口中，`reset` 会返回预生成 reset pool 和初始状态；`step` 需要显式传入该 pool。

```python
import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.agents import ExpanderAgent, RandomAgent

env = GeneralsEnv(grid_dims=(10, 10), truncation=500)
agent_0 = RandomAgent()
agent_1 = ExpanderAgent()

key = jrandom.PRNGKey(42)
pool, state = env.reset(key)

while True:
    obs_0 = get_observation(state, 0)
    obs_1 = get_observation(state, 1)

    key, k1, k2 = jrandom.split(key, 3)
    actions = jnp.stack([
        agent_0.act(obs_0, k1),
        agent_1.act(obs_1, k2),
    ])

    timestep, state = env.step(state, actions, pool)
    if bool(timestep.terminated) or bool(timestep.truncated):
        break

print(f"Winner: {int(timestep.info.winner)}")
```

动作格式为长度 5 的整数数组：

```text
[pass, row, col, direction, split]
```

- `pass`：`1` 表示跳过，`0` 表示移动。
- `row`、`col`：源格子坐标。
- `direction`：`0=上`，`1=下`，`2=左`，`3=右`。
- `split`：`1` 表示移动一半军队，`0` 表示移动除 1 个驻军外的全部军队。

## 常用实验

单局对战：

```bash
uv run python examples/simple_example.py
```

并行环境示例：

```bash
uv run python examples/vectorized_example.py
```

pygame 可视化：

```bash
uv run python examples/visualization_example.py
```

玩家对战训练好的 PPO checkpoint：

```bash
uv run python examples/play_against_model.py /tmp/generals-ppo-8x8-generated.eqx \
  --grid-size 8 \
  --map-generator generated \
  --policy-mode greedy \
  --human-player 0 \
  --fps 30
```

控制方式：

- 左键点击自己的可移动格子，再点击相邻目标格移动。
- `S` 切换下一步是否移动一半军队，`P` 跳过本回合。
- 右键或 `Esc` 取消选中，终局后按 `R` 重开，`Q` 退出。
- 选中的源格会显示黄色边框，可移动目标格会显示绿色边框。
- 右侧面板会显示当前选择、split 状态和最近一次点击结果。

加载 checkpoint 时，`--grid-size` 必须与保存该 `.eqx` 模型时使用的网络尺寸一致。`.eqx` 属于实验产物，建议放在 `/tmp` 或其他实验目录，不要提交进 Git。

4x4 PPO smoke test：

```bash
uv run python examples/_experimental/ppo/train.py 64 \
  --num-steps 64 \
  --num-iterations 10 \
  --model-path /tmp/generals-ppo-4x4.eqx
```

8x8 generated 地图 PPO：

```bash
uv run python examples/_experimental/ppo/train.py 64 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --num-steps 64 \
  --num-iterations 10 \
  --pool-size 512 \
  --opponent-pool random,expander,city-rush,balanced \
  --model-path /tmp/generals-ppo-8x8-generated.eqx
```

行为克隆 warm start：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/behavior_clone.py 128 \
  --grid-size 8 \
  --pool-size 4096 \
  --num-steps 32 \
  --num-iterations 2000 \
  --lr 0.0007 \
  --teacher-pool expander-soft,expander,city-rush,balanced \
  --opponent-pool random,expander,balanced \
  --model-path /tmp/generals-bc-8x8-soft.eqx
```

BC 日志会输出 `Top1`、`Top3`、`KL` 和本批次 per-teacher 样本数。BC 主要用于 warm start，不应把 imitation accuracy 当成最终强度指标。

从已有 checkpoint 继续 finetune：

```bash
uv run python examples/_experimental/ppo/train.py 128 \
  --grid-size 8 \
  --init-model-path /tmp/generals-bc-8x8-soft.eqx \
  --model-path /tmp/generals-ppo-8x8-finetuned.eqx
```

`--init-model-path` 只加载网络权重，不恢复 optimizer state、PRNG key 或 iteration，因此这是 finetune，不是完整 resume。输入和输出 checkpoint 的 `--grid-size` 必须一致。

推荐迭代路线：

1. 用 `behavior_clone.py` 从多个 heuristic teacher 做 warm start。
2. 用 `train.py --init-model-path ... --opponent-pool ...` 对 heuristic opponent pool 做 PPO finetune。
3. 用 `evaluate_policy.py --opponent-pool ...` 在独立地图 seed 上评估，再决定是否继续训练。

批量评估 checkpoint：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_policy.py /tmp/generals-bc-8x8-soft.eqx \
  --num-games 2048 \
  --grid-size 8 \
  --map-generator generated \
  --max-steps 500 \
  --opponent-pool random,expander,city-rush,balanced \
  --policy-mode sample \
  --csv-path /tmp/generals-eval.csv
```

评估会按 opponent 输出 win rate、decisive win rate、draw rate 和 mean final time。后续比较 checkpoint 时，应优先看多 opponent suite 的评估结果，而不是单一 teacher 的 BC top-1。

## 验证

运行完整测试：

```bash
uv run pytest
```

编译检查：

```bash
uv run python -m compileall generals examples tests
```

提交前建议至少运行：

```bash
uv run pytest
git diff --check
git status
```

## 注意事项

- 4x4 训练命令主要用于 smoke test，不适合作为策略质量结论。
- 更有意义的实验建议使用 8x8 或更大 generated 地图，并在独立 seed 上批量评估。
- `.eqx` checkpoint 属于实验产物，建议放在 `/tmp` 或其他实验目录，不要提交进 Git。
- `bench.py` 和 `examples/_experimental/benchmark_performance.py` 仍含旧版环境接口痕迹，使用前应先按当前 `reset(key) -> (pool, state)` 和 `step(state, actions, pool)` 接口修复。
