<div align="center">

![AutoSimSOTA — 从仓库出发，让实验不断向前](https://raw.githubusercontent.com/2-chen/AutoSimSOTA/main/docs/assets/hero.png?v=editorial-20261003)

### 从仓库出发，让仿真实验不断向前。

面向具身仿真 benchmark 的 Agent 驱动指标优化系统。<br>
理解代码与资源、连接原生采集、训练策略、运行评测；每一次尝试，都留下可追溯的依据。

**Agent-led / Data-first / Evidence-bound**<br>
Experimental · Python 3.10+

[项目理念](#为什么做-autosimsota) · [工作流程](#它如何开展研究) · [可视化预览](#看得见的研究过程) · [开始使用](#开始使用) · [更新日志](CHANGELOG.md)

</div>

## 为什么做 AutoSimSOTA

仿真研究的难点不只是调参。不同仓库有不同的仿真器、数据布局、训练入口、策略接口与评测约束。安装成功不代表环境能运行，训练完成不代表评测加载了正确的权重，跑出一个数字也不代表真正提升。

AutoSimSOTA 希望把这些断开的环节连起来：让主 Agent 阅读真实代码、根据证据选择下一步，让执行器保护源仓库、资源边界和实验协议。技能库提供方法，而不是按 benchmark 名称写死一套答案。

它参考 AutoSOTA 的研究职责划分，但**直接从代码仓库开始**，不需要先从论文寻找复现仓库；专门关注具身仿真里的资源连接、原生 rollout、数据采集和策略身份核验。

| 不只是…… | 更重要的是…… |
| :--- | :--- |
| 启动训练脚本 | 把数据、配置、训练产物与实际加载的策略连接起来 |
| 遇到错误后重试 | 封存错误证据，交给 Agent 分析、修复，再复验原操作 |
| 搜索更高分的参数 | 提出可检验假设，保持评测协议，区分筛选与正式确认 |
| 输出一份最终报告 | 持续记录发生了什么、为什么、证据在哪里 |

## 它如何开展研究

![仓库输入、主 Agent 研究循环与证据输出](https://raw.githubusercontent.com/2-chen/AutoSimSOTA/main/docs/assets/architecture.png?v=editorial-20261003)

**主 Agent 掌握研究上下文。** 它决定读什么、调用哪些技能、安排调查与实验，以及继续修复、换方向还是声明资源不足。Resource、Objective、Init、Monitor、Fix、Ideator、Scheduler、Supervisor 是研究职责，不意味着固定启动八个独立模型进程。

**技能按需阅读。** Agent 先看到短目录，再主动选择具体技能并记录理由。关键词只提供推荐；技能、候选 idea 与运行记忆分别管理。

**执行器守住边界。** 源码在隔离副本中修改；已有数据、资产和权重显式只读连接。长期原生作业可以后台执行，局部时间和费用窗口可在总额度内调整；不靠解除总预算或修改评测来制造“进展”。

**改进必须有证据链。** 训练产物 → 实际策略加载 → 原生 rollout → 指标；候选筛选分数不直接等于最终 best。失败、无提升和缺资源也都是需要如实记录的结果。

## 看得见的研究过程

![AutoSimSOTA 项目介绍短片](https://raw.githubusercontent.com/2-chen/AutoSimSOTA/main/docs/assets/intro.gif?v=editorial-20261003)

<div align="center">

[↓ 下载 24 秒 MP4 介绍视频](https://raw.githubusercontent.com/2-chen/AutoSimSOTA/main/docs/assets/autosimsota-intro.mp4?v=editorial-20261003) · [静态预览](https://raw.githubusercontent.com/2-chen/AutoSimSOTA/main/docs/assets/intro-poster.png?v=editorial-20261003) · [素材说明与重新生成](docs/assets/README.md)

</div>

> 主视觉为 AI 生成的机械臂概念插画；上方是基于该插画与代码动画制作的 **24 秒项目短片**，不是仿真录像，不含实测分数，也不表示某个 benchmark 已完成闭环。

GIF 可直接预览；MP4 请下载后播放，不依赖 GitHub 的文件预览器。

运行时，Recorder 持续维护中文 `RUN.md` / `RUN.html`：当前阶段、实验对比、失败与修复、预算、作业状态，以及可以点击追溯的证据。真实产物可用时，页面可以附上原生仿真视频、图表和数据摘要；没有产物时记录缺口，不拿占位内容冒充 demo。

![RUN.md 研究笔记示意：状态、对比、证据与下一步](https://raw.githubusercontent.com/2-chen/AutoSimSOTA/main/docs/assets/run-preview.png?v=editorial-20261003)

*这是阅读体验示意，不是正在运行的实验截图。“待核验”不会替换成虚构指标。实际报告内容与 demo 是否可用，取决于该次运行产生的证据。*

## 开始使用

需要 Python 3.10+；正式 Agent 运行还需要可用的 Claude Code CLI、配置好的模型端点，以及满足沙箱检查的 Linux 环境。Claude Code 是代码执行运行器，**不等于底层必须使用 Claude 模型**。仿真器、GPU 驱动、数据和权重由目标任务决定，本项目不会将它们一并安装。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ./autosim
.venv/bin/autosim research --help
```

按 [.env.example](.env.example) 在本地配置模型凭据，再提供目标仓库、全新的输出目录和已有资源：

```bash
.venv/bin/autosim research /absolute/benchmark /absolute/new-run 2 '{}' \
  --framework autosota_sim_v1 --isolated-copy --confirm \
  --resource data=/absolute/datasets/task-data \
  --resource pretrained=/absolute/checkpoints/policy
```

`data` / `pretrained` 是示例挂载位置：**必须换成该仓库实际读取的相对路径**。默认复制 Git 已跟踪的当前工作树源码；未跟踪的数据不会自动进入副本。输出目录须与源仓库互不包含。

新 AutoSOTA run 默认墙钟上限 48 小时、每任务 24 GPU 小时，初始模型回合额度 $4、整次模型费用上限 $60。费用是本地用量估算而非服务商账单；局部额度可调整，总边界仍有效。续跑沿用已经冻结的设置。

完整前置条件、资源挂载、续跑和环境复用见 [使用指南](docs/USAGE.md)。

## 当前能力与边界

这是一个**实验性研究框架，不是“任意仓库一键 SOTA”的保证**。

- 已实现 Agent 主导的研究编排、按需技能、隔离资源访问、长期作业、预算账本、证据封存与可视化记录；代码层能力不等于每个仓库都已完成端到端验收。
- RoboSyn 与 RoboTwin 有旧专用管线的真实实验记录；LIBERO 的通用框架运行仍在验证。历史结果不能替代当前通用路径的验证，更不能当作公开榜单 SOTA。
- 缺少必要资产、无法获取的权重、不兼容的驱动或不可用的采集路径，可能使任务无法完成。系统应报告具体证据与边界，而不是编造可运行条件。
- 正式比较必须核验评测协议、实际加载的策略与原生 rollout。环境探针、筛选试验和 CPU 测试均不能代替性能确认。

历史数字、对照条件与限制单独保存在 [历史实验记录](docs/HISTORICAL_RESULTS.md)。

## 深入了解

| 文档 | 内容 |
| :--- | :--- |
| [使用指南](docs/USAGE.md) | 安装、源码隔离、资源连接、预算、续跑与报告 |
| [与 AutoSOTA 的框架对照](docs/AUTOSOTA_COMPARISON.md) | 参考框架与具身仿真适配 |
| [研究记录与证据](docs/PROCESS_RECORD.md) | RUN 文档、产物与可追溯性 |
| [自适应算力](docs/ADAPTIVE_COMPUTE.md) | 长期作业、预算调整与执行边界 |
| [历史实验记录](docs/HISTORICAL_RESULTS.md) | 旧专用管线结果，不代表当前通用验收 |
| [更新日志](CHANGELOG.md) | 功能变化与验证范围 |

```text
autosim/       研究框架、执行器、技能与测试
docs/          使用说明、设计说明与展示素材
tools/         记录、检查与素材生成工具
patches/       仿真相关补丁
attic/         历史实现存档
```

研究运行目录、环境缓存、数据、checkpoint 与 `.env` 不随仓库发布。项目尚未选定开源许可证；第三方代码和资源遵循各自许可。
