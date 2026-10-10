# 使用指南

从已有 benchmark 仓库启动研究。本文使用当前 `autosota_sim_v1` 入口；历史 `--task`、`--run-id`、`--probe-only` 等参数不能套用。

## 1. 准备运行器

需要 Python 3.10+、可用的 Claude Code CLI、可访问的模型端点，以及支持当前沙箱的 Linux 主机。正式运行在预检中检查 loopback 网关、CLI 与隔离能力；仿真所需的 GPU / 驱动 / 资产另行核验。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ./autosim
.venv/bin/autosim research --help
```

参考根目录 `.env.example` 创建本地 `.env`，填写自己的端点与密钥，设置权限 `chmod 600 .env`。不要上传该文件，不要把凭据放入命令行或报告。采用 Claude Code CLI 不表示必须调用 Claude 模型。

## 2. 分开提供代码与资源

```bash
.venv/bin/autosim research /absolute/benchmark /absolute/new-run 2 '{}' \
  --framework autosota_sim_v1 --isolated-copy --confirm \
  --resource data=/absolute/datasets/task-data \
  --resource pretrained=/absolute/checkpoints/policy
```

`repo` 和 `output` 为必填位置参数；`2` 是研究轮数配置，不是“两次训练必定完成”的承诺。`settings_json` 的键应来自该仓库的评测声明，不要凭空指定通用训练参数。

- 默认复制 Git 已跟踪的**当前工作树**文件，保留未提交的源码修改；`--tracked-copy` 可显式指定。没有 Git 索引时不自动猜测源码边界。
- `--resource RELATIVE_TARGET=/ABSOLUTE/SOURCE` 可重复。左侧是仓库代码实际读取的位置，不一定叫 `data`；右侧是已有资源路径。挂载只读，不能覆盖复制的源码。
- 未跟踪的演示、checkpoint、仿真资产不会默认复制。只复制源码不能证明运行条件完整；Agent 仍要检查读路径、任务和策略兼容性。
- 确需连未跟踪文件一起复制时，可显式 `--full-copy`，仍受复制大小限制；不要以完整复制掩盖资源归属问题。
- 输出目录须与源仓库互不包含。正式框架禁止 `--in-place`；不要直接修改受保护的源仓库。

缺资产并不必然是系统故障；报告应说明缺什么、为什么需要、是否有合法获取方式。

## 3. 设置预算与长期作业

```bash
.venv/bin/autosim research /absolute/benchmark /absolute/new-run 2 '{}' \
  --framework autosota_sim_v1 --budget-scope task \
  --wall-seconds 172800 --agent-turn-budget-usd 4 \
  --agent-total-budget-usd 60 --agent-timeout-seconds 900 \
  --max-actions 80 --max-relaunch 24 --confirm
```

新 AutoSOTA run 默认墙钟 48 小时、每任务 24 GPU 小时、初始回合 $4、整次模型费用 $60。GPU 账本统计受控阶段持有物理设备租约的时间，包含租约内的准备和清理，不是仅统计 CUDA kernel 时间。

Agent 可以提交长期作业、取消停滞任务、在剩余额度内扩大或重分配局部窗口。总 GPU / 墙钟 / 模型费用及实验协议仍是边界。`--max-actions` 是单个监督段动作上限，Monitor 可以在续段和总预算范围内恢复，并非解除所有终止条件。

模型费用依赖用量回执与有效价格卡，是本地估算，不等于服务商账单。价格过期、未知模型、缺少用量不能按免费处理。旧 run 沿用冻结额度，改变默认值不会重置旧账本。更多细节见 [自适应算力](ADAPTIVE_COMPUTE.md)。

## 4. 复用环境，而不跳过核验

```bash
.venv/bin/autosim environments list
.venv/bin/autosim environments register --prefix /absolute/safe-conda-base
.venv/bin/autosim research /absolute/benchmark /absolute/new-run 2 '{}' \
  --framework autosota_sim_v1 \
  --environment-store /absolute/shared-environments
```

新运行默认启用环境池。Agent 看到静态短目录与 ID，决定复用 wheel、安全基础环境或快照；不按 benchmark 名称自动套环境。新运行可加 `--no-environment-reuse` 禁用。

环境池不能与源仓库或运行输出互相包含。续跑不静默更换环境池或复用策略；基础环境与缓存只读，运行副本独立。安装成功之后，仍须核验 GPU、reset / step、渲染、策略加载和原生 rollout。默认池容量 64 GiB，不是预装仿真镜像。

## 5. 阅读与恢复研究

打开输出目录的 `RUN.md` 或 `RUN.html`，查看中文进展、实验对比、长期作业、预算和证据链接。demo 必须来自已记录产物；没有视频 / 图表时，不应该填入虚构结果。

同一输出目录再次运行会使用已有状态与冻结预算，不是创建新预算。不要在原进程仍活动时启动第二个写入者。使用同一源路径、输出路径及兼容配置；必要时可用 `--keep-only` 只沿用已保存的声明、环境与命令，不重新派生。记录不完整时，它不能自动补齐一切。

需要手动刷新记录时：

```bash
.venv/bin/python tools/record.py /absolute/existing-run
```

`--confirm` 请求按留出设置确认最佳候选，但是否完成取决于真实执行证据和剩余资源；不存在有效 baseline / candidate 时不应产生伪造分数。正式计分与筛选记录必须分开。

## 6. CPU 回归

```bash
.venv/bin/python -m pip install pytest
PYTHONPATH=autosim .venv/bin/python -m pytest -q autosim/tests
```

CPU 测试验证框架合同与部分真实本地隔离行为，不验证任意仿真器兼容性，更不是 SOTA 结果。

### 数据生产、保管与实际训练曝光

Agent 可使用 `register_data_version` 登记实际训练文件/目录，并通过已审核训练轴的 `data-version:<id>` 使用它。可选资产卡连接类型、封存 collect/prepare_data 回执、父版本、原生健康计数和处理配方；原始数据不覆盖，处理后的可读结果使用新位置。旧登记不要求补齐元数据，数量未知继续保持未知。

原生代码在真实读写后发 `AUTOSIM_DATA_PROFILE`，loader 在实际 batch 读取后发消费/曝光事件；Supervisor 先看真实卡和精确源码再审核。框架复核身份与计数一致性，不替 Agent 猜字段、安装统一格式或决定采集数量。RUN.md 自动展示中文数据表、真实曝光及对应开发指标；有至少两份全量 episode 计数才画数量图，不把重复读取当作新增数据。

字段、边界和示例见按需 [原生数据合同](../autosim/autosim/skills/curating-simulation-datasets/references/native-data-contract.md)。当前是单 run 的资产/消费链，不是跨 run 数据平台；分片续采、转换缓存、独立备份和数据回收尚未实现，现有 checkpoint 清理授权不包含 raw 数据。

## 7. 串行真实验收

`tools/run_acceptance_queue.py` 是操作者定义的启动队列，不是新的 benchmark 适配器。清单显式提供仓库、独立输出、任务设置、解释器、只读资源和用户服务名；每项仍由完整 AutoSim Main 流程执行，不预写训练/采集/评测代码。需要本机用户 systemd、可用的 Claude Code CLI 与正式运行隔离条件。

```bash
PYTHONPATH=autosim .venv/bin/python tools/run_acceptance_queue.py \
  /absolute/project/autoresearch_runs/matrix/queue_manifest.json
```

清单 `jobs` 中每项字段为 `id`、`repo`、`output`、`unit`、`settings`、`interpreter`、`resources`、`adopt`。`adopt: true` 只监视已经启动且来源身份匹配的原 run，不创建第二个写入者。新项使用独立 task 账本、48 小时墙钟、24 GPU 小时、模型 $60 / 初始局部 $4、一轮候选和留出确认，自动 GC 默认关闭。任务在排队期间不启动自己的预算。

队列只有在前项退出且具有真实 baseline、候选指标、独立留出确认与 Supervisor `real` 审核时才前进；零分不是缺失指标，验收也不表示提升或 SOTA。缺证据或失败时保留原 run，写入中文 `MATRIX.md` / `queue_state.json`，停止启动后续任务；不会自动追加预算、重建实验、修框架或清理文件。排查原 run 后重新启动队列可沿用原清单，不能更改已冻结输入。视频质量与采集→训练消费仍须检查各 run 的实际证据，不能凭排队状态宣称完成。
# 数据研究任务书

新 run 可使用 `--research-brief /absolute/path/study.md` 向主 Agent 提供 UTF-8 研究意图，
最多 6000 字符/24KB，独立于 `settings-json` 和原生训练参数保存。首次启动冻结正文与身份；
续跑可省略该参数，保留原意图。更改意图需要新的研究 run，不能给已初始化 run 偷加任务书。
该文件不授予权限、不改指标或总预算，也不作为实验已完成的证据。

失败驱动数据研究见 [方法与实验协议](FAILURE_DRIVEN_DATA_RESEARCH.md)，
可行性试点任务书见 [PILOT_BRIEF](paper/PILOT_BRIEF.md)。主 Agent 调用
`register_data_campaign` 保存开发失败假设/获取与筛选/targeted-random对照，
原生采集、数据版本、真实训练消费及独立计分后用 `record_data_campaign_arm` 关联结果。
旧数据处理不能当成新的自动采集；已暴露 held-out 的 run 不再注册开发实验。
