# ENPIRE 与 autoresearch-robotics 原生仓库试跑

目的：实际检验两个公开研究循环能否接入 RoboSyn / RoboTwin，完成基线、提出候选、原生评测与保留／回退。先用小规模试跑识别能力与接入边界，再决定是否扩大成论文对照。

运行入口：[实时 RUN.md](../autoresearch_runs/external_systems_20261016_v1/RUN.md)。

## 候选选择与公开实现边界

| 系统 | 固定版本／来源 | 选择理由与接入范围 |
| --- | --- | --- |
| ENPIRE | [NVlabs/ENPIRE](https://github.com/NVlabs/ENPIRE)，`99ee90acf65b5b18957c8382ad580db999528be3` | 已公开 coding-agent 自主研究合同、Environment／Policy 接口、TrialRunner 与产物记录。让 coding agent 自行接入原生任务；不安装无关真机服务／PLD，不称原论文全部系统复现 |
| autoresearch-robotics | [官方仓库](https://github.com/jellyheadandrew/autoresearch-robotics)，`ff30cc481af9343b9c96f448b5ed72b75bfcd562` | 已有简洁的 program.md 与训练—评测—保留循环。原模板为 Fetch，不含这两个仓库；让 agent 创建薄原生入口，记录与原版的差异 |
| Nautilus | [作者项目页](https://yufengjin.github.io/nautilus/) | 2026-10-16 核查主框架仍标 Coming Soon，公开 nautilus-collect 不能替代主研究框架，本轮不编造复现 |

仅凭论文能力描述不能认定已经支持这两个仓库的公开 SOTA。本轮使用明确适配的版本，接入成本也计入研究窗口。

## 图像反馈研究与先行文本诊断

两个系统各接入两仓库，四个图像反馈 arm 独立源码与模型会话，GPU 串行。此前首个 ENPIRE/RoboSyn 文本反馈诊断单独保留；随后新增独立的 `enpire_robosyn_vision`，从同一起点重新研究，不继承文本诊断的路线和结果。因此总计五个记录目录，正式 pilot 比较只使用四个图像条件。

| 输入 | RoboSyn | RoboTwin |
| --- | --- | --- |
| 原仓库 | test/RoboSynChallenge | test/RoboTwin |
| 任务／配置 | click_bell / random | beat_block_hammer / demo_clean |
| 共同起点 | 已有发布 ACT checkpoint | 本机已有 ACT baseline checkpoint，非官方成绩 |
| 现有资源 | 原生数据、权重、EmbodiChain 资产与可用 Python | 原生数据、processed_data、baseline 权重与资产 |
| 开发评测 | 20 episodes，seed2026 | 20 episodes，seed2026，保留专家有效初态检查 |
| 最终确认 | 提交最终选择后，基线与候选各50 episodes，seed2027 | 同左 |

候选路线由各自 agent 决定，包括合法的策略、训练、部署参数与数据操作。不提供预选提分参数，不把 AutoSimSOTA 的 Main／Data／Fix、技能或候选审批链加入外部研究循环。

提供相同的隔离、模型传输与费用记录。模型统一 deepseek-flash。2026-10-16进一步核查[官方视觉文档](https://api-docs.deepseek.com/guides/vision/)确认其支持图像，实际双色测试图经 Anthropic 兼容端点返回正确颜色；证据见[视觉验收](../autoresearch_runs/external_systems_20261016_v1/vision_preflight.json)。旧代理的文本限制已改为显式可启用 inline image，图像请求保守预留完整输入上下文，再按 provider usage 结算；默认文本调用不变。代理与费用相关23项测试通过。

本次为有限试点，统一使用non-thinking请求、单次最多8192输出token；该配置与原论文使用的模型／推理配置不同，不能用本试点没有提分推断其最强配置也无法提分。实际生成策略与费用回执一同保留。

首个文本诊断已发生，费用和全部结果保留，不将其视觉缺失归因于模型。四个后续 arm 允许 agent 实际读取原生视频提取的关键帧并据图提出假设；须保留帧路径与真实图像请求，不能只把媒体链接写进报告便声称看过。它仍是特定模型与适配条件下的 pilot，不能推出其他模型配置的能力上限。

## 配置与操作者干预

1. 复制小型源码到 arm 内；数据、资产、已训练权重只读连接，未复制 RoboTwin 约102GB processed_data。
2. bubblewrap 内隐藏宿主 home，仅开放当前 trial 可写，源仓库／环境／上游参考代码只读；不暴露项目 `.env`，模型密钥只在外部代理持有。
3. 原生评测脚本、任务代码与任务配置冻结。RoboTwin 原脚本把评测数写死100，准备阶段只增加 `test_num` 的20/50/100选择，不变更 seed、任务、初态有效性检查或成功定义；两个系统应用相同改动。
4. 修正复用环境选择：旧 conda 直接解释器加载 PyTorch2.6，能枚举5090但不支持其计算；项目现有环境的 PyTorch2.7.1+cu128 已验证真实 CUDA 张量运算。RoboTwin 使用该环境，不原地升级旧环境。
5. RoboSyn 复用旧 run 的依赖环境；trial 内 sitecustomize 仅移除指向旧 benchmark checkout 的 editable 映射，保留真实依赖绑定，防止新试验实际执行旧实验源码。
6. Agent 负责薄适配、假设、候选、训练／评测、结果与中文 RESEARCH.md。操作者的后续配置修复另记，不能追认成全程无人工接入。
7. 首 arm 自主发现准备复制时误漏 `robosynchallenge.data`，从只读原始包建立运行时桥接并完成仿真。原因是准备器将所有名为 `data` 的目录都排除，已修为仅排除根级数据资源；未启动 arm 原样补齐 Python 子包，按输入源码清单核对无缺项。该工程差异计入 pilot 接入记录，不能把首 arm 的额外修复成本直接当框架差距。

8. RoboTwin补齐独立依赖，保留共享环境；首冒烟发现RT/OIDN报错与空视频。在未形成有效基线前停止该配置尝试，固定本地原生`AUTOSIM_DISABLE_RT=1`路径，两个系统均用同样default/raster条件。旧日志不删除、不记0分；不做跨渲染或官方排行榜等价性推断。
9. RoboTwin沿原模型会话续跑，继承原起点、全部费用与未知预留；由Agent修复视频初始化／帧写入并重新验收，再开始共同基线。操作者通知与恢复记录可追溯，不称无人工接通。

每个 arm 模型上限$5、整个研究窗口75分钟；含先行文本诊断的五个 arm 模型最高合计$25，独立视觉预检最多$0.40，实际 usage 单列。RoboSyn 新试点三arm设备占用保守上限3.75小时，RoboTwin两arm上限2.5小时。RoboSyn准备时旧GPU消耗约19.05小时，合计上限约22.80小时，仍小于24小时；新旧账分别保存，旧模型未知预留不释放。75分钟是本次有限试跑总窗口，不是通用训练命令超时规则。

## 观察与复核

- `RUN.md`：调度状态、费用、原生指标证据和各 arm 的研究记录链接，运行中刷新。
- 各 arm 的 `RESEARCH.md`／`pilot_results.json`：由外部 agent 维护研究过程及选择。
- `logs/agent_*.jsonl`：实际工具与模型轨迹；`model_usage.json`：真实 provider usage 与保守预留。
- `outputs/` 与 `benchmark/eval_result/`：原生训练、评测、视频与数据产物。
- `source_*_before/after.json`：源仓库相关源码前后身份；运行结束检查未被改动。

先判断接入是否完成，再逐项核对基线与候选的原生分数、样本量、所加载权重及保护文件。Agent 自写 `completed` 不能代替外部原生证据复核；partial、异常或缺失不记作0分。开发改善与独立确认分别报告，一次20集 pilot 不宣称公开 SOTA。

启动工具：[external_baseline_trial.py](../tools/external_baseline_trial.py)。已有输出不可隐式重放；继续或修改应保留会话、费用和原始证据，并单独记录依据。
