# AutoSimSOTA

面向具身仿真 benchmark 的可审计自动研究系统。

一条指令接收 benchmark 仓库路径，系统自己完成：识别任务与能力 → 构建证据 → 由 LLM 研究控制器提出实验 →
定向采集数据 → 质检准入 → 训练 policy → 原生评测 → 候选选择 → 冻结确认 → 导出可运行的优化后仓库。
DeepSeek 负责研究提案，不负责执行 GPU 训练。

```bash
.venv/bin/autosim research /path/to/RoboSynChallenge \
  --task water_pouring --controller api --allow-api-egress --gpu 0 \
  --rounds 2 --attempts-per-round 100 --training-steps 20000 \
  --development-episodes 40 --selection-episodes 100 --final-episodes 200 \
  --train-seed 1000 --min-original-fraction 0.25 --hours 24
```

## 阶段性成果

### RoboSynChallenge / water_pouring：同种子配对确认

一次上述指令（`run_id=full_run_20260917`，2026-09-17，墙钟约 8.7 小时，2 轮全部完成），
在冻结的 200 局确认库上、三条臂使用**完全相同**的初始状态种子：

| 臂 | 成功 | 成功率 | vs 候选（配对 McNemar 精确检验） |
|---|---|---|---|
| 官方 ACT checkpoint | 94/200 | **47.0%** | 候选多赢 70 局、少赢 6 局，**+32.0 pp，p = 3.2e-15** |
| 等预算纯官方数据续训 | 103/200 | **51.5%** | 候选多赢 62 局、少赢 7 局，**+27.5 pp，p = 2.1e-12** |
| 本系统候选 | 158/200 | **79.0%** | — |

第二条对照是这个结果的关键：它与候选**同轮数、同 20000 步、同初始 checkpoint、同训练种子**，
只是数据全部来自官方数据集。所以提升不是"多训练了一会儿"造成的。

导出验证：`export_validation.status = passed`，优化后的仓库在真实仿真器中跑通原生 episode
（`execution_mode = real_simulation`）。这是导出可用性的冒烟检查，不是性能声明。

### 系统自己收敛到的方案

两轮的提案、假设、期望验证全部由 LLM 依据证据写出，没有任何按任务预置的规则：

- **第 1 轮** — 从 40 局开发证据里读出失败队列（bottle 净位移 0.092 m vs 成功队列 0.203 m、
  cup 缺少成功局的 z_range 抬升），判断为"抓取后未复位"的恢复类失败，选择 `targeted_recovery`
  定向采集 75 次、保留 25 次原始分布采样、0.5 采样质量、20000 步。
  开发库 15/40 → **31/40**。
- **第 2 轮** — 控制器自己指出第 1 轮没有分离出"定向数据有用"和"这个 profile 有用"，
  引用技能库中"profile 选择是本任务上最弱的杠杆"这条经验，设计了等质量等步数的对照：
  换成 `targeted_clutter`。开发库 31/40 → 29/40（配对零结果），选择阶段 77/100 vs 75/100，
  第 1 轮候选被选中。这正是它自己写下的零点条件——"换 profile 没有差别，说明 profile
  选择不是约束"——系统据此收敛，而没有继续在第三个 profile 上浪费预算。

每轮的提案、假设、理由、证据 id、期望验证都落在
`autoresearch_runs/<benchmark>/<run_id>/rounds/round_N/proposal.json`，可逐轮审计。

### 跨 benchmark 泛化

决策层（`autosim/research/decision.py`）里**没有任何 benchmark 名称**：提示词、schema 与校验
全部由适配器声明的优化空间生成。接入新 benchmark = 写一个适配器，不改决策层。

- **RoboSynChallenge** — 上表即为其结果。
- **RoboTwin 2.0 / beat_block_hammer** — 同一个决策层、同一条指令形态，`controller=api` 单轮闭环
  跑通（`robotwin_validate_20260917`）：clean 库 +30 pp（7:1，p=0.035）、randomized 库 +15 pp（3:0，p=0.125）。
  每库 20 局，**方向性结果，未达统计确证**。

### 陌生 benchmark 的自动接入

给一个**从未见过**的 benchmark 仓库路径，系统自己读它、写出适配声明、逐条核对：

```bash
.venv/bin/autosim scout /path/to/SomeBenchmark --run-id onboard
```

流程是：**确定性扫描**（文件树、清单、把每个源文件里出现的信号按文件列出来，不作判断）→
**LLM 决定读哪些文件** → **LLM 分两段写声明**（身份与任务合同 / 能力与可调空间）→
**系统逐条核对**：路径是否存在、模式匹配到几个文件、声明里写进 `path` 的是不是一句话、
声明的观测与动作维度是否与录制数据一致、仓库外的资产是否真有归属证据。
模型声称的每一条与系统核实的每一条分开记录在 `verification.json` 里。

在 **LIBERO** 上实测（`autoresearch_runs/scouting/libero_onboarded_v3`，约 40 秒）：

| 系统自动得到 | 结果 |
|---|---|
| 身份与任务 | 130 个任务，从 `bddl_files/*/*.bddl` 数出来 |
| 任务合同 | `state_dim=47, action_dim=7`，相机 `agentview_rgb`/`eye_in_hand_rgb` 128×128，`max_episode_steps=1000` —— **与录制的 HDF5 一致** |
| 可调空间 | 从 LIBERO 自己的 CLI/配置推出：`device`、`use-depth`、`policy.policy_type`、`lifelong.algo`、`train.loss_scale` … |
| 适配器协议 | `check_adapter` 报 0 个缺项 |
| `new_trajectory_generation` | **declared** —— 见下方「成功轨迹从哪来」 |
| `official_policy` | unsupported —— 它没有认领隔壁 RoboSyn 的 checkpoint，理由是这个路径不在本仓库任何脚本或配置里出现 |

**「成功轨迹从哪来」是方法库的一条 skill，不是一条检查。** 第一版把「仓库里有没有专家脚本」当成了「能不能产出成功轨迹」，
于是判定 LIBERO 不能产出数据 —— 这是错的。LIBERO 的环境每次 reset 都重采样物体位置，`step` 返回 `done = self._check_success()`，
所以**跑当前策略、用环境自己的判据把成功的留下**，就能产出训练数据，不需要专家、不需要人。
差别在于：专家从零就能产出，而过滤式采样的产出率≈当前策略成功率 —— 它是**自举**，不是供给。

这条推理写在 `autosim/autosim/skills/where-successes-come-from/SKILL.md` 里，作为**参考**注入 scout 与 planner 的提示词：
不参与校验、可以被推翻、加一条方法就是加一个文件。加上它之后，判定自动从 `unsupported` 变成 `declared`，
没有新增任何检查。

系统接着**自己提出研究方案**（`strategy.json`），并对着已声明的空间逐条校验。
它自己把 `targeted_collection` 判为暂不可用，理由比「有/没有专家」细得多：

> 第三个来源——**用 benchmark 自己的逐步成功判据过滤策略 rollout**——是能救回这一族的，
> 而 LIBERO 的环境确实在 reset 时重采样物体位置（除非 `deterministic_reset`），
> `step` 也返回 `done = self._check_success()`，所以这个来源**大概率存在**。
> 但方案不会把开头几轮花在它上面，因为这里真正的区别是**什么时候能用**，不是**能不能用**：
> 它的产出率取决于策略成功率，而本 benchmark 没有发布策略可供建立这个成功率
> （`official_policy` 为 unsupported，也没有任何 LIBERO 配置指向 checkpoint）。
> 因此把它记为「已声明但暂不可用」，并**记为第二轮产出高于下限后的条件回退项**。

第一轮干预（模型自己选的，并给出了可被推翻的判据）：

> 用 LIBERO 自己的 lifelong evaluator，把每轮评测局数提到足以分辨方法差异的量级；
> 如果放大后的区间仍然重叠，说明此前的方法差异本来就是噪声。
> **推翻它的条件**：同一策略跑两遍的区间已经盖住此前报告的组间差距；
> 另外「零产出或全零成功」也是真实结果，说明可用策略路径低于下限，第二轮就转向 `training_recipe`。

**「没有专家」不是「不能研究」，而是少了一族干预手段。** 闭环第一轮执行**方案里的第一个干预**，
而不是「必须采集」——采集只是其中一个族。`--strategy` 把方案按内容哈希冻进 `protocol.json`，
同 run-id 换方案续跑会被拒绝。

LIBERO 因此可以走进一条真实路径：官方 50 条演示训 ACT 基线 → 在「选哪些演示 / 怎么加权 / 怎么训」上干预
→ 用它自己的 50 个固定初始状态评测。全程不需要专家。

## 复现

上面的命令已用 `--dry-run` 逐字段比对过 `full_run_20260917` 记录的 `protocol.json`，
除 `dry_run` 标志本身外**零差异**——即这条命令就是当时跑出上表的命令。

前置条件：

1. benchmark 仓库 checkout（路径由你提供；也可用 `AUTOSIM_BENCHMARK_ROOT` 指定搜索目录）。
2. 仿真器与 ACT 训练所需的独立环境、数据与资产（本仓库不打包）。
3. 一块可用 GPU。
4. 项目根目录 `.env` 中的 `DEEPSEEK_API_KEY`（`chmod 600`；该文件已被 `.gitignore` 忽略，永不入库）。

产物落在 `autoresearch_runs/<benchmark>/<run_id>/`，包含逐轮提案、证据、评测、选择、
确认报告与导出验证。中途中断可用同一 `--run-id` 加 `--continue-run` 续跑。

想先确认环境而不动 API/GPU：

```bash
.venv/bin/autosim research /absolute/path/to/RoboSynChallenge --probe-only
```

## 诚实的边界

- 上表是**本机、单任务、200 局**的结果，不是官方隐藏赛道成绩，也不是 SOTA 认证；
  与公开榜单的数值不可直接相比（本地复评官方 checkpoint 为 47.0%，不等于榜单上该 checkpoint 的分数）。
- 候选臂相对等预算对照，同时改变了三件事：训练数据的来源、混合采样方式、损失与增广 profile。
  **本运行没有把增益归因到其中任何单独一项**——第 2 轮正是控制器自己去测其中一项的尝试。
- 同种子配对检验排除了初始状态方差，但**没有排除仿真器本身的确定性差异**；
  报告中的 `interpretation` 字段保留了这条限制原文。
- RoboTwin 一栏样本量小，仅作泛化性的方向性证据。
- 多卡与任意 benchmark 零样本适配仍未验证。

## 仓库边界

- `autosim/`：Python 包、CLI、适配器、研究执行器、技能库与测试。
- `patches/`：RoboSyn/RoboTwin 等外部仓库的本地兼容修改与基准版本记录，不含完整上游代码或资产。
- `docs/`：发布检查与可移植性限制。
- `.env.example`：无凭据配置模板。

第三方 RoboSynChallenge、RoboTwin、EmbodiChain 不是本项目核心代码。虚拟环境、模型、数据、缓存、
运行报告和本地历史 plan 不进入 Git；忽略规则不删除这些本地文件。实际维护的总计划位于本仓库外，
不是运行依赖。

## 安装与 CPU 检查

从本仓库根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ./autosim pytest
.venv/bin/autosim research --help
.venv/bin/python -m pytest
```

这只安装核心依赖，不安装仿真器、CUDA、ACT 训练依赖或资产。真实研究需要按对应 benchmark
配置独立环境及数据。项目根目录 `.env` 用于本地 API 配置；不要把密钥放进提交、报告或命令示例。

完整流程见[研究入口说明](autosim/autosim/research/README.md)；发布前请阅读[发布检查](docs/RELEASE_READINESS.md)。

## 许可与发布

尚未确定本仓库整体许可证，请维护者在公开发布前确认核心代码来源并选择许可证。
第三方组件各自遵循原许可证；兼容补丁不改变其权利归属。
不要因早期 README 的许可证徽章而推断本仓库整体已获 MIT 授权。
