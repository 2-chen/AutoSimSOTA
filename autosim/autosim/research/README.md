# AutoSimSOTA Repository AutoResearch

统一入口接收一个官方 benchmark 仓库，输出带 checkpoint、manifest 和完整研究账本的衍生仓库。RoboSynChallenge执行器已通过多次原生运行；RoboTwin ACT执行器现已接入同一入口，依次执行官方数据基线、clean/randomized失败证据、API/对照决策、有界专家采集、硬链接混合数据、候选训练、同seed原生评测和导出。RoboTwin已有 API 决策的单轮闭环运行：clean 库 +30 pp（7:1，p=0.035）、randomized 库 +15 pp（3:0，p=0.125），每库 20 局，属方向性证据。RoboSynChallenge/water_pouring 的 200 局同种子配对确认见仓库根 README。

```bash
/path/to/AutoSimSOTA/.venv/bin/autosim research /path/to/RoboSynChallenge \
  --task auto --controller api --run-id my_run --hours 24
```

默认输出固定在 `AutoSimSOTA/autoresearch_runs`，与启动命令的工作目录无关。凭据只读取 `AutoSimSOTA/.env` 或显式 `AUTOSIM_ENV_FILE`；不会向父目录搜索。`--task auto` 按 registry 稳定顺序选择首个同时具备官方 ACT checkpoint 和官方数据的任务。

## 常用模式

```bash
# 只做任务合同/资产/能力盘点；缺资产也形成合法报告，不调用 API/GPU
autosim research RoboSynChallenge --task water_pouring --probe-only --run-id water_probe

# 完整资产及协议验证，不调用 API/GPU
autosim research RoboSynChallenge --task click_bell --dry-run --run-id click_dry

# 缩小预算的真实管线验收（不是性能结论）
autosim research RoboSynChallenge --task click_bell --controller api \
  --rounds 2 --attempts-per-round 4 --training-steps 200 \
  --development-episodes 3 --selection-episodes 3 --final-episodes 3 \
  --run-id click_pipeline_probe --hours 3

# 同权限、同预算决策对照
autosim research RoboSynChallenge --task click_bell --controller fixed --run-id fixed_seed1
autosim research RoboSynChallenge --task click_bell --controller random --run-id random_seed1
autosim research RoboSynChallenge --task click_bell --controller heuristic --run-id heuristic_seed1

# 完整确认运行（根 README 中 47.0% / 51.5% / 79.0% 那一行的命令）
autosim research RoboSynChallenge --task water_pouring --controller api \
  --allow-api-egress --gpu 0 --rounds 2 --attempts-per-round 100 --training-steps 20000 \
  --development-episodes 40 --selection-episodes 100 --final-episodes 200 \
  --train-seed 1000 --min-original-fraction 0.25 --hours 24 --run-id full_run

# RoboTwin：官方数据基线 -> clean/randomized证据 -> 采集 -> 候选 -> 原生配对评测
autosim research RoboTwin --task beat_block_hammer --controller heuristic \
  --robotwin-training-epochs 6000 --robotwin-evaluation-episodes 20 \
  --run-id robotwin_beat_hammer_seed1000 --hours 24
```

`--controller api|fixed|random|heuristic` 共享同一个 proposal schema、能力白名单、采集器、训练器、seed bank 和评测器。控制组不调用 DeepSeek，也不会被记录成 API 闭环。可显式配置 `--rounds`、`--attempts-per-round`、`--training-steps`、三组评测局数及 `--min-original-fraction`；所有值冻结进 `protocol.json`，恢复时不允许悄然变化。

研究seed不仅控制训练：seed-1000保留历史登记bank，其他seed按`train_seed × purpose`确定性派生开发、选型、最终、每轮采集及导出bank。因此同一研究seed内各控制器可配对比较，不同研究seed也不再静默复用相同episode。

RoboTwin使用项目内部锁定的CuRobo v0.7.8、Warp 1.12和XPolicyLab。其server/client通过`XPOLICYLAB_PYTHON=/path/to/AutoSimSOTA/.venv_robotwin/bin/python`固定到独立解释器；主`.venv`保持RoboSyn/Newton所需的Warp 1.13，避免backend间依赖互相覆盖。官方采集脚本支持可选`max_seed_attempts`，AutoResearch探针必须设置该值，防止专家失败时无界换seed。环境锁与5090兼容构建说明见`RoboTwin/autoresearch_env/`。

RoboTwin的官方专家可解性筛选存在启动非确定性，因此“相同起始seed”不保证最终计数episode相同。执行器先保存基线实际计数的seed bank，候选以`expert_check=false`和只读`seed_bank_file`严格重放；seed顺序/数量、初始化和收据哈希任一不一致都拒绝比较。

RoboSyn的DexSim会在真实解析仓库路径过深时于材质创建阶段触发`DFString.h:94`断言。导出器会把完整输出物理复制到`/tmp/asr_*/r`短路径后运行native smoke；符号链接不能替代物理复制，因为资产路径会被解析回长路径。交付仓库也应复制或解压到类似`/tmp/robosyn`的短路径运行，具体要求记录在`AUTORESEARCH_MANIFEST.json`。

## 闭环语义

1. 解析仓库、任务合同、资产和合法操作，写入 `benchmark_discovery.json`、`asset_manifest.json` 与 `capabilities.json`。
2. 完整审计官方数据；相同内容、任务合同及审计器哈希再次出现时，只有全路径/大小/mtime 校验命中才复用全视频审计。随后原生评测官方 checkpoint，生成带 policy 哈希的开发证据。
3. controller 只从动态能力白名单提出诊断/采集/处理/训练提案，本地校验后才执行。
4. 定向与原分布专家采集均按尝试数记账；实际场景、成功/失败、数据哈希和训练曝光可追溯。定向因素读回不是 `verified` 的数据不得以定向来源进入训练。
5. 每轮候选重新原生评测并重新分析失败。下一轮读取当前候选证据；若采用 policy-prefix 接管，使用当前候选而非旧官方 policy。
6. 独立选型 bank 冻结候选；达到预设门槛后才打开 final bank。官方发布 policy、官方数据续训和候选使用相同 seed 顺序。
7. 以研究评测实际使用的清洁兼容仓库为导出基底，只叠加有哈希记录的数据采集/训练扩展与 checkpoint，再逐级验证独立加载和原生 episode。只有原生 episode 真正执行才置 `export_runtime_verified=true`。

能力初始状态只是 `declared`；必须有真实仿真/训练回执才升级为 `verified`。缺测量保持 `unknown` 或 `unsupported`。

## 决策层与适配器

提示词、提案 schema 与校验集中在 `decision.py`，其中**不含任何 benchmark 名称**：它接收适配器声明的
`OptimizationSpace`，据此生成请求并校验提案。benchmark 的取值、分组、结构校验与耦合规则全部写在
适配器里（`robosyn_adapter.py`、`robotwin_adapter.py`）。接入新 benchmark 只需实现
`adapter_protocol.REQUIRED_METHODS`（`tasks` / `select_task` / `discover` / `task_contract` /
`capabilities` / `optimization_space`），`check_adapter()` 会在运行前报出缺项，而不是中途崩溃。

证据层同样不按任务分支：`task_evidence()` 输出原始测量（逐实体轨迹、成功/失败队列对比、任务自身
`is_task_success` 的源码文本）而不命名类别，阈值与因果判断由控制器自己给出。
`autosim/autosim/skills/*/SKILL.md` 是可增删的方法库，作为参考注入提示词——不参与校验，
控制器可以推翻其中任何一条。

## 公平性与结论边界

- policy 只接收任务合同声明的关节状态和 RGB，相机/对象真值、seed 与 judge 不进入 policy 输入。
- 最终集合在候选冻结前不提供给 controller；开发、选型、最终及采集 seed 由 SQLite 账本隔离。
- 新采集是 benchmark 允许的训练数据优化，不修改官方 success judge、随机化、时限或评测观测；是否符合某个官方提交赛道仍需按该赛道规则确认。
- `completed`、`performance_improved`、`hypothesis_supported` 和 `export_runtime_verified` 分开记录。管线跑完不等于提升，更不自动等于官方 SOTA。
- 当前多 backend、第二 learner、未知仓库自动适配及同预算多研究 seed 的论文级验证仍按唯一计划文件 `plan/autoresearch_progress_feasibility_plan_20260907.md` 推进。

CPU 回归：

```bash
cd /path/to/AutoSimSOTA
PYTHONPATH=autosim .venv/bin/pytest -q autosim/tests
```
