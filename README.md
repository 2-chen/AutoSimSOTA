# AutoSimSOTA

面向具身仿真 benchmark 的可审计自动研究系统。

当前入口由通用仓库调查、声明、环境准备、命令推导和研究循环组成。并非任意仓库都已能完成
训练、确认和导出；运行结果按实际证据分层，缺失资产或不支持的工作流应明确记录边界。
目标与验收见 [总规划](plan/MASTER_PLAN.md)，当前证据、缺口和下一步见 [执行进度](plan/EXECUTION_STATUS.md)。

AutoSOTA 模型预算（2026-09-30，追加翻倍）：新运行默认初始回合额度 `$4`、整次费用上限 `$60`。
DeepSeek 按官方用量估算计费，局部额度可自动从剩余总预算扩展，扩展记录保存在费用账本。
可用 `--agent-turn-budget-usd` / `--agent-total-budget-usd` 配置；整次费用上限仍是硬边界，
未知用量保留预留，不将其当零费用。旧运行保留已冻结额度，默认值变化不会改写旧账本。
本地额度拒绝不再作为限流重试，局部失败交回恢复流程，不直接等同整次费用耗尽。
安装操作逐条封存完整证据与部分进度，RUN.md 展示实际执行记录；安装成功不等于仿真核验通过。

### 调度与并发（2026-09-30）

新 AutoSOTA run 自动生成 `scheduler_policy.json`：主 Agent 负责选择/采纳研究，最多两个
只读调查异步运行；Recorder 后台更新中文 RUN.md/RUN.html，模型费用共用总账本。
`wait_for_jobs` 用状态事件等待，不反复调用模型；纯等待不消耗 Scheduler 续段次数。

已核验的任意原生阶段均可提交后台作业；Agent 请求 CPU、内存、GPU、优先级与局部窗口。
同阶段输出独占，活跃作业阻止源码/环境变更。跨 run CPU/内存 admission、CPU affinity 与
物理 GPU 排他租约防止默认超卖；内存仍是合作式预留，不是 RSS 硬隔离。
依赖图显式声明独立性和资源后可并发，默认串行。单张 GPU 不默认并发多个重训练。

可信 baseline 后可用 `configure_screening / run_screening_trial / inspect_screening`
做 Agent 声明训练预算轴的多保真参数候选筛选；筛选结果单独存放，不能直接成为正式 best。
代码/算法候选保持原来的审核与源码事务。所有执行仍受同一 24 GPU 小时/墙钟/费用总边界约束。
`scheduling_metrics.json` 和 RUN.md 展示调度耗时、排队及筛选表；旧 run 不自动开启新调度。
详细接口和边界见 [最新规划](plan/MASTER_PLAN.md)。

### 环境复用（2026-09-30）

新 `autosota_sim_v1` 运行默认启用跨运行环境池，位置为
`autoresearch_cache/environments`。它复用依赖，不把数据、任务配置和训练权重当成环境已就绪的证明。

- 成功安装产生的 pip wheel（包括可识别的 HTTP 缓存对象）由控制器校验、按内容哈希发布；
  后续运行通过只读 `PIP_FIND_LINKS` / `UV_FIND_LINKS` 使用，HOME、安装 prefix 和下载缓存仍各自独立。
- 从本机 conda、已登记环境和近期运行环境读取静态包信息；Agent 只见短目录与不透明 ID，
  自主决定是否复用并记录理由。不会按 LIBERO/RoboTwin 等仓库名自动选环境。
- 安全 conda 基础环境用 `--offline --copy --clone` 复制到运行自己的 `env`，基础 prefix 与
  原包缓存只读。拒绝可编辑源码引用、外部 site-packages 和无法独立搬迁的 venv。
  缺少 conda 原包缓存时保留具体故障证据，交回 Fix；不声称所有现有环境都能离线复制。
- 环境 probes 通过后才尝试发布快照；后续复用先检查内容哈希，复制后仍须执行当前仓库的
  原生探针。包缓存/复制成功不能代替 GPU、reset/step、渲染、策略加载与 rollout 核验。
- `RUN.md` 展示可用 wheel 数、选择理由、复制与快照状态；可用数不是实际命中数。
  池默认容量上界 64 GiB，空间不足或缓存失败不推翻已经通过的环境；未完成副本保留供检查，不自动清理。

```bash
.venv/bin/autosim environments list
.venv/bin/autosim environments register --prefix /absolute/path/to/safe-conda-base
.venv/bin/autosim research /absolute/repo /absolute/new-run 2 '{}' \
  --framework autosota_sim_v1 --environment-store /absolute/shared-environments
# 完全禁用复用：为新运行添加 --no-environment-reuse
```

公共池不能与源仓库或输出目录互相包含；同一次续跑不能静默切换池或启停复用。
内置 Python/PyTorch/MuJoCo/SAPIEN 条目目前是能力说明，不是已经下载、通过 RTX 5090 核验的
预装镜像。真实小型 conda 离线复制、快照再复制与离线 wheel 安装已验收；具身仓库的加速收益仍须实测。

当前命令形式（仓库和输出目录均为必填）：

```bash
.venv/bin/autosim research /absolute/path/to/benchmark /absolute/path/to/run 2 '{"steps": 100, "episodes": 10}' --wall-seconds 3600
.venv/bin/autosim research /absolute/path/to/benchmark /absolute/path/to/run 2 '{}' --wall-seconds 3600 --isolated-copy
.venv/bin/autosim research /absolute/path/to/benchmark /absolute/path/to/run 0 '{}' --wall-seconds 600 --isolated-copy --tracked-copy
.venv/bin/autosim research /absolute/path/to/benchmark /absolute/path/to/run 2 '{}' --keep-only
.venv/bin/python tools/record.py /absolute/path/to/run
```

`--keep-only` 只使用输出目录已记录的声明、环境和命令，不重新调查或派生；
当所需记录齐全时会直接运行，不再为选步骤调用外部模型。新运行默认隔离（`--isolated-copy`），仅复制 Git
已跟踪的当前工作树文件（保留未提交的源码修改），`--tracked-copy` 是显式同义选项。
没有 Git 索引时拒绝自动猜测源码范围；确实需要完整副本时使用 `--full-copy`，仍受复制上限约束。
续跑默认沿用原先复制模式和资源连接，不迁移旧运行。
旧 legacy 原地运行必须显式传 `--in-place`（这会允许修改源仓库）；正式 `autosota_sim_v1` 禁止该选项。
输出目录必须与源仓库互不包含，检查在创建输出或环境之前执行。
新研究输出默认使用独立的 `task` 预算：每项研究最多 24 GPU 小时，同输出目录的续跑和长期作业共享账本。
`task_gpu_budget.json` 记录受控原生 GPU 阶段持有物理设备锁的时长，包含设备占用期间的准备/清理，
不是 GPU 内核利用率积分；LLM 等待和控制器 CPU 时间不计入。启动 GPU 阶段时预留额度，结束按占用时长结算，
异常崩溃遗留预留需核对进程后处理，不自动清零。GPU 总限会约束原生作业硬超时，局部延长不能越过它。
既有运行继续沿用旧的跨仓库历史账本，不将旧墙钟消耗伪装为实测 GPU 时间；
新任务可显式用 `--budget-scope repository` 选择旧的累计墙钟上界政策。运行中不能切换计量口径。
未指定 `--wall-seconds` 时，新 AutoSOTA 运行默认 48 小时墙钟（给安装/LLM/CPU 工作留空间），
legacy 运行仍为一小时；24 GPU 小时的独立硬上限不变，旧运行沿用记录的期限。
`--rederive` 会丢弃该输出目录中已保存的派生记录，应只在明确要重新推导时使用。
不传 `settings-json` 时沿用 benchmark 自身默认配置，系统不再暗中注入 epoch/episode 数量。
`--wall-seconds` 是可选的整次运行墙钟上限；恢复同一运行时不能静默改小或改大。
墙钟期限与 GPU/模型费用分别管理；当前暂停期间墙钟期限继续流逝，记录为 `deadline_continues`。
AutoSOTA 入口先检查本机代理监听、Claude Code 和进程隔离能力，记录 `runtime_preflight.json`；
失败时生成中文启动报告并退出，不调用 Agent 重试。控制器需要允许模型联网/本机监听和设备访问的运行环境；
仓库命令仍由系统自己的隔离执行器约束。启动后的代理权限失败也会封存脱敏堆栈、结束进程记录，
并以 `infrastructure_blocked` 返回，不空转 Scheduler/Monitor/Fix。
`--isolated-copy` 在输出目录创建有字节上限的独立源码副本，避免研究补丁直接改原 checkout；
默认上限 4 GiB，可用 `--copy-limit-bytes` 调整。它不是容器沙箱，绝对路径、外部服务和资产仍需审计。

代码副本与已有资源分开输入，例如（路径仅作示例）：

```bash
.venv/bin/autosim research /absolute/repo /absolute/new-run 1 '{}' \
  --framework autosota_sim_v1 --isolated-copy \
  --resource data=/absolute/datasets/task-data \
  --resource pretrained=/absolute/checkpoints/policy
```

`--resource` 可重复，格式是 `运行内相对路径=已有资源绝对路径`。系统不复制资源、不创建指向源仓库
的软链接或硬链接；在受控诊断/实验进程内只读挂载，宿主文件浏览器只会看到空挂载占位路径。
位于源仓库内、被显式声明为资源的文件/目录，即使被 Git 跟踪，也会从代码复制范围排除。
资源目标不得覆盖其余已复制的源码或互相重叠；系统不按仓库名或扩展名猜测资源范围。
训练输出、数据预处理结果、缓存和新权重必须写到其他
run-local 路径；需要原地修改数据的脚本应调整输出位置，不能把原始资源改成可写。
`workspace_snapshot.json` 记录连接、源身份与最多 128 个被省略路径候选；RUN.md 和主 Agent 显示摘要。
未连接的资源不自动授权访问，也不因为未复制就判定原仓库不存在资源。连接只证明访问路径存在，
原生 loader、任务匹配、权重实际加载、内容哈希和评测仍须验证。
资源是实时只读视图，不是冻结副本：其他宿主进程仍能修改资源内容；当前身份校验检测路径替换，
不宣称完整数据不可变。续跑拒绝资源消失、替换或连接变更。资源不随代码导出，搬迁需重新连接并核验。
资源目录应是自包含的数据/资产树，不应包含指向宿主其他位置的依赖链接。

对已经产生的测量，可另开进程运行 `PYTHONPATH=autosim .venv/bin/python -m autosim.research.receipt_verifier <run-dir> <measurement-label>`，从原始评测日志或冻结的 JSON/CSV 结果重读指标，并核对 attempt 回执与记载的数值。`consistent` 只表示这些记录内部一致；它不是原生策略加载、独立复评或统计显著性结论。
接入记录可显式声明 `execution_graph`：节点使用仓库自己的阶段名，`depends_on` 指定顺序，`bindings` 把上游产物传给下游输入，`score_target` 指定评测节点。图会在执行前校验并冻结；研究循环可从图的评测节点读主指标。图节点运行完成仍不等于独立确认或 SOTA。
运行时从一开始生成 `RUN.md` 与 `feasibility.json`；原生长阶段每 45 秒刷新 Markdown 状态。
显式主指标支持日志、JSON 和 CSV 读取，但这仍不是任务/种子/样本协议的完整确认。
下方结果来自较早的专用实验路径，是历史证据，**不是当前通用入口的验收结果**。

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

## 历史实验复现说明（旧专用管线）

本节描述 2026-09-17 的历史记录；其中旧旗标如 `--probe-only`、`--task`、`--gpu`
已不是当前 `autosim research` 的参数。请用文首当前命令形式启动新运行。

上面的命令已用 `--dry-run` 逐字段比对过 `full_run_20260917` 记录的 `protocol.json`，
除 `dry_run` 标志本身外**零差异**——即这条命令就是当时跑出上表的命令。

前置条件：

1. benchmark 仓库 checkout（路径由你提供；也可用 `AUTOSIM_BENCHMARK_ROOT` 指定搜索目录）。
2. 仿真器与 ACT 训练所需的独立环境、数据与资产（本仓库不打包）。
3. 一块可用 GPU。
4. 项目根目录 `.env` 中的 `DEEPSEEK_API_KEY`（`chmod 600`；该文件已被 `.gitignore` 忽略，永不入库）。

产物落在 `autoresearch_runs/<benchmark>/<run_id>/`，包含逐轮提案、证据、评测、选择、
确认报告与导出验证。中途中断可用同一 `--run-id` 加 `--continue-run` 续跑。

历史入口曾提供以下探针模式；当前通用入口尚未提供等价旗标，不能直接运行：

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
