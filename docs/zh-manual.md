# Generals Bots 中文手册

本文面向希望快速运行、理解和扩展本项目的开发者与强化学习实验人员。项目提供了一个基于 JAX 的 Generals.io 双人对战模拟器，重点是高吞吐、可向量化、可复现实验，以及用于训练和评估 bot 的实验脚本。

## 1. 项目定位

`generals-bots` 是一个面向 Generals.io bot 研究的 Python 包。它把游戏核心逻辑写成纯函数式、JAX 友好的形式，使同一套环境既可以单局调试，也可以用 `jax.vmap` 同时推进大量对局。

项目适合做三类工作：

- 编写规则型 agent，例如随机移动或扩张型策略。
- 批量运行环境，用于性能测试、数据采样和策略评估。
- 训练实验性强化学习策略，目前仓库内提供 PPO、行为克隆、策略评估和可视化脚本。

当前仓库的主要特点：

- 核心模拟器位于 `generals/core/`，使用 JAX array 和不可变 `NamedTuple` 状态。
- 环境包装类 `GeneralsEnv` 支持固定地图和变尺寸/填充地图，并使用预生成 state pool 做快速 auto-reset。
- 内置 `RandomAgent` 和 `ExpanderAgent` 两个基线 agent。
- `examples/` 提供单局、向量化、GUI 可视化示例。
- `examples/_experimental/ppo/` 提供实验性训练、行为克隆和批量评估工具。

## 2. 目录结构

```text
.
├── generals/                    # Python 包主体
│   ├── core/                    # JAX 游戏逻辑、环境、动作、观测、地图生成、奖励
│   ├── agents/                  # Agent 抽象类和内置策略
│   ├── gui/                     # pygame GUI 和 replay 渲染
│   ├── remote/                  # generals.io 远程客户端相关代码
│   └── assets/                  # GUI 图片和字体资源
├── examples/                    # 用户示例
│   ├── simple_example.py        # 单局对战示例
│   ├── vectorized_example.py    # vmap 并行环境示例
│   ├── visualization_example.py # pygame 可视化示例
│   └── _experimental/           # 实验性 benchmark、PPO、策略可视化
├── tests/                       # pytest 测试
├── docs/                        # 文档和开发记录
├── pyproject.toml               # 包元数据、依赖、可选 extra
├── uv.lock                      # uv 锁文件
├── requirements.txt             # 传统 pip 依赖列表
└── README.md                    # 英文快速介绍
```

## 3. 核心概念

### 3.1 地图与格子编码

游戏地图是二维整数数组。核心生成器在 `generals/core/grid.py`：

- `-2`：mountain，不可通行。
- `0`：空地。
- `1`：玩家 0 的 general。
- `2`：玩家 1 的 general。
- `40-50` 附近的正整数：city，数值表示初始守军数量，具体范围可配置。

`generate_grid(...)` 会随机放置双方 general、mountain、city，并尽量保证地图连通和双方距离约束。训练脚本也支持 `simple` 地图，即只包含两个随机 general 的空地图，适合作为极快的 smoke test。

### 3.2 GameState

`GameState` 定义在 `generals/core/game.py`，包含完整隐藏状态：

- `armies`：每个格子的军队数量。
- `ownership`：形状为 `(2, H, W)` 的玩家占领掩码。
- `ownership_neutral`：中立格子掩码。
- `generals`、`cities`、`mountains`、`passable`：地图结构掩码。
- `general_positions`：双方 general 坐标。
- `time`：当前步数。
- `winner`：未结束为 `-1`，否则为获胜玩家编号。
- `pool_idx`：auto-reset 时从 state pool 取下一局的索引。

状态是不可变 `NamedTuple`，更新时通过 `_replace` 返回新状态，适合 JAX JIT 编译和批处理。

### 3.3 Observation

`Observation` 定义在 `generals/core/observation.py`。每个玩家只能看到战争迷雾下的局部信息，包括：

- `armies`
- `generals`
- `cities`
- `mountains`
- `neutral_cells`
- `owned_cells`
- `opponent_cells`
- `fog_cells`
- `structures_in_fog`
- `owned_land_count`
- `owned_army_count`
- `opponent_land_count`
- `opponent_army_count`
- `timestep`

`Observation.as_tensor()` 会把观测转换成神经网络可用的张量。单个观测输出 `(14, H, W)`；批量观测会保留批处理和玩家维度。

### 3.4 Action

动作是长度为 5 的整数数组：

```text
[pass, row, col, direction, split]
```

字段含义：

- `pass`：`1` 表示跳过本回合，`0` 表示移动。
- `row`、`col`：源格子坐标。
- `direction`：`0=上`，`1=下`，`2=左`，`3=右`。
- `split`：`1` 表示移动一半军队，`0` 表示移动除 1 个驻军外的全部军队。

常用工具函数：

```python
from generals import create_action, compute_valid_move_mask

action = create_action(to_pass=False, row=3, col=4, direction=1, to_split=False)
mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
```

`compute_valid_move_mask` 返回形状为 `(H, W, 4)` 的布尔数组，表示从每个格子朝四个方向移动是否合法。

### 3.5 GeneralsEnv

`GeneralsEnv` 是主要环境入口，定义在 `generals/core/env.py`。当前接口要点：

```python
import jax.random as jrandom
from generals import GeneralsEnv

env = GeneralsEnv(grid_dims=(10, 10), truncation=500)
key = jrandom.PRNGKey(42)
pool, state = env.reset(key)
timestep, state = env.step(state, actions, pool)
```

需要注意：

- `reset(key)` 返回 `(pool, init_state)`，其中 `pool` 是预生成的批量初始状态池。
- `step(state, actions, pool)` 推进一步游戏；终局或达到 `truncation` 后，会从 `pool` 自动重置。
- `actions` 的形状是 `(2, 5)`，分别对应双方玩家。
- `TimeStep` 包含 `observation`、`reward`、`terminated`、`truncated`、`info` 和 `last_state`。

`GeneralsEnv` 支持两种地图模式：

```python
# 固定尺寸
env = GeneralsEnv(grid_dims=(10, 10), truncation=500)

# 变尺寸地图，pad 到统一大小以便批处理
env = GeneralsEnv(min_grid_size=8, max_grid_size=24, pad_to=24)
```

## 4. 快速搭建环境

### 4.1 前置条件

建议环境：

- Ubuntu 24.04 x86-64。
- Python 3.11 或更高版本。
- `uv`，用于创建可复现环境并按 `uv.lock` 安装依赖。
- 可选：NVIDIA GPU 和 CUDA 13 运行环境，用于大规模训练。

如果还没有安装 `uv`，可按 uv 官方安装方式安装。安装完成后确认：

```bash
uv --version
```

### 4.2 获取代码

```bash
git clone https://github.com/CodeBoy2006/generals-bots.git
cd generals-bots
```

如果是在当前开发机的已有仓库中工作，直接进入仓库目录即可：

```bash
cd /home/codeboy/research/generals-bots
```

### 4.3 CPU 开发环境

安装运行和开发依赖：

```bash
uv sync --extra dev
```

确认包能导入：

```bash
uv run python -c "import generals; print(generals.GeneralsEnv)"
```

确认 JAX 后端：

```bash
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
```

在 CPU 环境中，输出通常会显示 `cpu`。

### 4.4 CUDA 13 GPU 环境

如果机器有可用 NVIDIA GPU，并希望使用 CUDA 13 版本的 JAX 插件：

```bash
uv sync --extra dev --extra cuda13
```

验证 GPU：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
```

期望看到 `gpu` 和 `CudaDevice(...)`。`XLA_PYTHON_CLIENT_PREALLOCATE=false` 可以避免 JAX 一启动就预占大量显存，适合与其他任务共用 GPU。

### 4.5 传统 pip 安装方式

如果不使用 `uv`，也可以用可编辑安装：

```bash
python3 -m pip install -e .
```

若要运行测试，还需要安装开发依赖：

```bash
python3 -m pip install -e ".[dev]"
```

项目内的推荐命令仍以 `uv run ...` 为准，因为它更贴近当前锁文件和实验脚本。

## 5. 基础实验

### 5.1 单局对战

运行随机策略对扩张策略的单局对战：

```bash
uv run python examples/simple_example.py
```

该脚本会：

1. 创建 `GeneralsEnv(grid_dims=(10, 10), truncation=500)`。
2. 调用 `env.reset(key)` 得到 `pool` 和初始 `state`。
3. 每步分别取两个玩家的 `Observation`。
4. 由 `RandomAgent` 和 `ExpanderAgent` 产生动作。
5. 调用 `env.step(state, actions, pool)` 直到终局或截断。

适合用于确认环境安装和核心 API 是否可用。

### 5.2 并行环境

运行向量化示例：

```bash
uv run python examples/vectorized_example.py
```

该脚本使用 `jax.vmap` 批量推进多个环境。核心思想是：

- 用一次 `env.reset(pool_key)` 生成共享 reset pool。
- 用 `jax.vmap(env.init_state)` 生成多个初始状态。
- 对 `get_observation`、agent `act` 和 `env.step` 分别做 `vmap`。

并行环境是本项目做 RL rollout 和大规模评估的基础。

### 5.3 GUI 可视化

运行 pygame 可视化：

```bash
uv run python examples/visualization_example.py
```

该示例会显示一局 `RandomAgent` 对 `ExpanderAgent` 的游戏过程。若在无显示服务器的远程机器上运行，pygame 窗口可能无法打开；此时优先运行非 GUI 示例或使用本地桌面/远程显示转发。

## 6. 性能实验

### 6.1 吞吐实验入口

当前最稳妥的吞吐验证入口是向量化示例：

```bash
uv run python examples/vectorized_example.py
```

它会在同一进程内并行运行 256 个 10x10 环境，并定期输出双方平均占地。若需要做正式吞吐 benchmark，可以基于这个脚本扩展计时代码：把多步 rollout 包进 `jax.jit`/`jax.lax.scan`，并在计时结束前同步设备结果。

性能测试应注意：

- JAX 第一次运行会触发 JIT 编译，不能把编译时间直接当成稳态吞吐。
- 对 GPU/TPU 计时要在结果上调用 `block_until_ready()`。
- 比较不同设置时，应固定地图尺寸、并行环境数量、步数和硬件后端。

### 6.2 旧 benchmark 脚本

仓库中还有 `bench.py` 和 `examples/_experimental/benchmark_performance.py`，目标是测量环境吞吐。不过它们包含旧版 `GeneralsEnv` 接口痕迹，使用前应先按当前 `reset(key) -> (pool, state)` 和 `step(state, actions, pool)` 接口修复。新实验建议先从 `examples/vectorized_example.py` 派生。

## 7. PPO 与策略实验

实验性训练代码位于：

```text
examples/_experimental/ppo/
```

主要脚本：

- `train.py`：基于 raw game API 的 PPO 训练路径，当前推荐作为快速实验入口。
- `train2.py`：基于 `GeneralsEnv` 包装的 PPO 训练路径。
- `behavior_clone.py`：从 Expander teacher 做行为克隆。
- `evaluate_policy.py`：批量评估保存的 `.eqx` 策略。
- `network.py`：Equinox 策略价值网络。
- `common.py`：地图生成、动作编码、策略动作选择等共享工具。

### 7.1 4x4 smoke 训练

先跑一个很小的训练，验证依赖、JAX 后端和模型保存流程：

```bash
uv run python examples/_experimental/ppo/train.py 64 \
  --num-steps 64 \
  --num-iterations 10 \
  --model-path /tmp/generals-ppo-4x4.eqx
```

4x4 只适合作为 smoke test，不适合做策略质量结论。

### 7.2 8x8 简单地图 PPO

```bash
uv run python examples/_experimental/ppo/train.py 64 \
  --grid-size 8 \
  --num-steps 64 \
  --num-iterations 10 \
  --model-path /tmp/generals-ppo-8x8-simple.eqx
```

`simple` 是默认地图生成器，只放置两个 general，训练速度快，但缺少 mountain 和 city。

### 7.3 8x8 generated 地图 PPO

更接近实际环境的实验应使用 generated 地图：

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
  --model-path /tmp/generals-ppo-8x8-generated.eqx
```

常用参数说明：

- `--grid-size`：方形地图尺寸。
- `--map-generator simple|generated`：空地图或带 terrain 的地图。
- `--pool-size`：预生成 reset state 数量，必须至少等于并行环境数量。
- `--truncation`：单局最大步数。
- `--opponent`：单一训练对手，默认 `random`。
- `--opponent-pool`：逗号分隔的训练对手池；设置后优先于 `--opponent`，每个环境会随机抽取对手。
- `--mountain-density-min/max`：mountain 密度范围。
- `--num-cities-min/max`：city 数量范围。
- `--min-generals-distance`：双方 general 最小距离；未设置时训练脚本会取 `max(3, grid_size // 2)`。
- `--init-model-path`：可选输入 checkpoint，用于从已有 `.eqx` 网络权重开始 finetune。
- `--model-path`：Equinox `.eqx` checkpoint 保存路径。

`--init-model-path` 只恢复网络权重，不恢复 optimizer state、PRNG key 或 iteration，因此它是 finetune，不是完整 resume。输入 checkpoint 和本次训练的 `--grid-size` 必须一致。

### 7.4 GPU 训练命令模板

在 CUDA 机器上，建议显式指定后端：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train.py 128 \
  --grid-size 8 \
  --opponent-pool random,expander,city-rush,balanced \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --num-steps 128 \
  --num-iterations 100 \
  --pool-size 2048 \
  --model-path /tmp/generals-ppo-8x8-gpu.eqx
```

如果显存不足，优先降低：

- 并行环境数量，也就是位置参数 `128`。
- `--num-steps`。
- `--pool-size`。
- `--grid-size`。

### 7.5 行为克隆 warm start

仓库已有从 randomized Expander teacher 学习的行为克隆脚本，适合作为 PPO 或 self-play 的 warm start：

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

`--teacher` 保留单 teacher 兼容路径；`--teacher-pool` 设置后会覆盖它，并在每个环境中随机抽 teacher。`--opponent-pool` 用于增加采样时对手的多样性，目前支持 `random` 和内置 heuristic。BC 日志中的 `Top1`、`Top3` 和 `KL` 只表示模仿 teacher 的程度，不等价于真实对战强度。

输出模型默认建议放在 `/tmp` 或其他实验目录，不要直接提交 `.eqx` checkpoint。

BC warm start 后继续 PPO finetune 的典型命令：

```bash
uv run python examples/_experimental/ppo/train.py 128 \
  --grid-size 8 \
  --opponent-pool random,expander,city-rush,balanced \
  --init-model-path /tmp/generals-bc-8x8-soft.eqx \
  --model-path /tmp/generals-ppo-8x8-finetuned.eqx
```

推荐的迭代路线是：先用多 teacher BC 做 warm start，再用 PPO 对 heuristic opponent pool finetune，随后用多 opponent suite 在独立地图 seed 上评估。BC 是让策略快速脱离随机初始行为的预训练阶段，后续提升应主要依赖 PPO/self-play 和独立评估结果。

### 7.6 批量评估 checkpoint

评估行为克隆或 PPO checkpoint：

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

关键输出：

- 每个 opponent 的 `Wins/Losses/Draws`
- `WinRate`
- `Decisive`
- `DrawRate`
- `MeanTime`
- `Seconds`

如果设置 `--csv-path`，脚本会输出稳定列名的 CSV，便于后续报告画柱状图或折线图。实验结论应优先基于多 opponent、多 seed、多批次评估，而不是单次训练日志或 BC top-1 accuracy。

### 7.7 可视化训练好的策略

可视化 `.eqx` 模型：

```bash
uv run python examples/_experimental/visualize_policy.py /tmp/generals-ppo-8x8-generated.eqx 10 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5
```

可视化时应保持 `--grid-size` 和地图生成参数与训练 checkpoint 兼容，否则网络尺寸或输入分布可能不匹配。

### 7.8 玩家对战训练好的策略

可以用本地 pygame 窗口和 `.eqx` PPO checkpoint 对战：

```bash
uv run python examples/play_against_model.py /tmp/generals-ppo-8x8-generated.eqx \
  --grid-size 8 \
  --map-generator generated \
  --policy-mode greedy \
  --human-player 0 \
  --fps 30
```

控制方式：

- 左键点击自己的可移动格子作为源格，再点击相邻目标格提交移动。
- `S` 切换下一步是否 split/半兵移动。
- `P` 跳过本回合。
- 右键或 `Esc` 取消当前选中。
- 终局或达到 `--max-steps` 后按 `R` 重开，`Q` 或关闭窗口退出。
- 选中的源格会显示黄色边框，可移动目标格会显示绿色边框。
- 右侧面板会显示当前选择、split 状态和最近一次点击结果。

该入口只支持当前 PPO `PolicyValueNetwork` 保存出的 Equinox `.eqx` checkpoint。`--grid-size` 必须和训练/保存模型时的网络尺寸一致，否则会加载失败或在推理时因输入尺寸不匹配报错。checkpoint 通常较大且属于实验产物，建议放在 `/tmp` 或专门的实验目录，不要提交进 Git。

## 8. 编写自己的 Agent

自定义 agent 需要继承 `generals.agents.agent.Agent` 并实现 `act(observation, key)`：

```python
import jax.numpy as jnp
from generals.agents.agent import Agent
from generals.core.action import compute_valid_move_mask


class FirstMoveAgent(Agent):
    def act(self, observation, key):
        mask = compute_valid_move_mask(
            observation.armies,
            observation.owned_cells,
            observation.mountains,
        )
        moves = jnp.argwhere(mask, size=mask.size, fill_value=-1)
        num_valid = jnp.sum(jnp.all(moves >= 0, axis=-1))
        move = moves[0]
        pass_action = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
        move_action = jnp.array([0, move[0], move[1], move[2], 0], dtype=jnp.int32)
        return jnp.where(num_valid > 0, move_action, pass_action)
```

为了保持 JAX 兼容性，agent 的 `act` 最好：

- 使用 `jax.numpy` 而不是普通 Python list 运算。
- 避免依赖可变全局状态。
- 对随机性使用传入的 `key`，不要复用同一个 PRNG key。
- 返回固定形状的 `jnp.ndarray`。

## 9. 测试与验证

运行完整测试：

```bash
uv run pytest
```

只运行地图生成相关测试：

```bash
uv run pytest tests/test_grid_generation_performance.py
```

编译检查常用命令：

```bash
uv run python -m compileall generals examples tests
```

提交前建议至少运行：

```bash
uv run pytest
git diff --check
git status
```

若改动涉及训练脚本，还应额外跑一个很小的 smoke train，例如：

```bash
uv run python examples/_experimental/ppo/train.py 2 \
  --num-steps 2 \
  --num-iterations 1 \
  --pool-size 8 \
  --model-path /tmp/generals-ppo-smoke.eqx
```

## 10. 实验建议

推荐从小到大推进：

1. 先运行 `examples/simple_example.py` 确认环境可用。
2. 再运行 `examples/vectorized_example.py` 确认 JAX 批处理正常。
3. 用 4x4 PPO smoke test 检查训练脚本。
4. 切到 8x8 generated 地图，固定 terrain 参数做短训练。
5. 用 `evaluate_policy.py` 在独立 seed 上批量评估。
6. 用 `visualize_policy.py` 观察策略是否出现明显无效行为。
7. 增加并行环境数、rollout 步数、迭代次数和地图尺寸。

做严肃对比时应记录：

- Git commit。
- JAX 后端和设备。
- 地图尺寸与生成参数。
- 训练命令完整参数。
- checkpoint 路径。
- 评估命令、seed、局数和最大步数。
- 胜/负/平、总胜率、decisive win rate、draw rate。

`docs/devlogs/` 中已有若干英文实验记录，可以作为记录格式参考。

## 11. 常见问题

### 11.1 README 里的部分代码和当前接口不一致怎么办？

以仓库源码和 `examples/` 下可运行脚本为准。当前 `GeneralsEnv.reset(key)` 返回 `(pool, state)`，`GeneralsEnv.step(...)` 需要传入 `pool`。

### 11.2 为什么要有 state pool？

JAX JIT 适合静态形状和函数式数据流。预生成 state pool 后，终局 auto-reset 可以通过数组索引完成，避免在每个 step 内重新生成复杂地图，也减少 JIT 重编译风险。

### 11.3 为什么 benchmark 第一次运行慢？

第一次运行会触发 JAX/XLA 编译。性能比较应忽略 warmup，并在计时结束前同步设备结果。

### 11.4 训练模型保存在哪里？

示例命令使用 `/tmp/*.eqx`。这类 checkpoint 通常较大且属于实验产物，不建议提交进 Git。

### 11.5 4x4 结果能说明策略强吗？

不能。4x4 主要用于 smoke test。更有意义的实验至少应使用 8x8 generated 地图，并在独立地图上批量评估。

### 11.6 CPU 可以训练吗？

可以，但大规模训练会慢很多。CPU 更适合安装验证、小规模 smoke test、单局调试和文档示例。
