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
