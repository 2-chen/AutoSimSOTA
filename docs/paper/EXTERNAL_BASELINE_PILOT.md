# 外部系统 baseline：2026-10-16 原生接入试点

已补入英文稿 `External native-port baseline pilots` 小节。这批结果可以作为接入试点、负结果与失败案例，不能作为完整系统优劣排名。原始 Agent 报告保持原样，以下是对原生产物的独立复核。

## 本次实际结果

| 研究流程 | 仓库／反馈条件 | 开发基线 | 候选 | 最终结果 |
| --- | --- | --- | --- | --- |
| ENPIRE 合同指导 | RoboSyn random／文本诊断 | 11/20，55% | 执行窗口25：10/20；窗口10：6/20 | 均回退；原基线 seed2027 确认21/50，42% |
| ENPIRE 原生接口接入 | RoboTwin／图像可用 | 0/20 | 最优验证权重0/20；关闭时域聚合0/20 | 没有提分，保留起点；未确认 |
| autoresearch-robotics 适配 | RoboTwin／图像可用 | 0/20 | 开启聚合0/20；重训0/20；重训＋聚合0/20 | 没有提分；未确认 |
| autoresearch-robotics 适配 | RoboSyn random／图像可用 | 未取得有效分数 | 主协议下未评测 | 场景构建失败；不能填0% |
| ENPIRE 合同指导 | RoboSyn random／图像可用 | 未取得有效分数 | 主协议下未评测 | 场景构建失败；不能填0% |

开发集为20集、seed2026；确认集为50集、seed2027。每一行只有一次研究试跑，不是多次独立 Agent 重复，也不能把 episode 数当研究重复数。

### 不混入主任务的诊断数据

- autoresearch-robotics 在 RoboSyn **clear** 配置下：基线12/20；窗口25候选3/20；`act_step=5`候选12/20；另一seed的基线32/50。
- ENPIRE 图像可用的 RoboSyn arm 在 **clear** 下仅完成1集冒烟，1/1；不能表述成100% benchmark成绩。
- ENPIRE 文本诊断在确认开始后另试时域集成，7/20；不把该探索追认成冻结候选，也不替换原确认选择。

这些数值均经原生JSON逐集成功数／聚合数复核。clear 与 random 是不同场景条件，不做跨配置提分比较。

## 身份、条件和实际执行

- 上游：[ENPIRE](https://github.com/NVlabs/ENPIRE/tree/99ee90acf65b5b18957c8382ad580db999528be3)，[autoresearch-robotics](https://github.com/jellyheadandrew/autoresearch-robotics/tree/ff30cc481af9343b9c96f448b5ed72b75bfcd562)。这是明确适配的 native-port pilot，不是原论文完整系统复现。
- ENPIRE/RoboTwin 实际使用 `TrialRunner`、`ArtifactStore`；ENPIRE/RoboSyn 使用研究合同指导原生批评测，不能写成完整 ENPIRE 运行时复现。
- 模型统一 DeepSeek Flash，non-thinking，单请求最多8192输出token；不能据此推断其他模型与原论文配置的能力上限。
- 首个文本诊断单列；后续四arm开放图像。实际图像请求回执数依次为：ENPIRE/RoboTwin 52、autoresearch/RoboSyn 60、autoresearch/RoboTwin 53、ENPIRE/RoboSyn 0。开放图像不等于实际获得图像。
- RoboSyn 起点是发布 ACT 权重；RoboTwin 起点是本机已有 ACT 训练产物，不是经本次复现的官方 baseline。
- RoboTwin 均使用原生 default/raster 渲染开关，不能冒充默认RT条件；两个agent的初始时域聚合设置也不同，不能视为冻结配方下公平排名。
- ENPIRE/RoboTwin 的60集逐集记录均为400步正常截断，不是适配器异常记0。autoresearch/RoboTwin 的四份原生计数均完整到20集，实际加载日志均为全部权重键匹配；原生结果文件与日志末次成功率一致。
- autoresearch/RoboTwin 完成一次400 epoch、356.3秒的原生重训，使用已有50条示范，无新增采集。报告写候选加载 `policy_best.ckpt`，但实际日志明确加载 **`policy_last.ckpt`**。按真实加载身份记录，不把最优验证loss附会到部署模型。其“行为质变”“充分拟合”等Agent判断未作为实验证明写入论文。
- RoboTwin 专家有效初态筛选导致部分评测的实际接受seed序列不同；不能仅因顶层seed相同就宣称物理初态配对。

## 当前运行情况与仍需处理的问题

本轮监督进程在 **2026-10-16 21:28:51（北京时间）** 正常退出，服务为 `inactive/dead`、退出码0。检查时5090利用率0%，显存约1066MiB；没有继续运行的本研究 Agent／评测进程。这表示队列执行完毕，不表示每个arm都完成了目标。

1. **RoboSyn random 的运行环境差异尚未解释。** 两个图像可用arm在 distractor mesh 的原生材质初始化中出现断言或挂起；但早先文本arm成功完成random。证据只支持“这两次运行受阻”，不支持“RoboSyn注定不支持”或“引擎一定无法修复”。Agent关于修改环境必然等于修改任务协议的判断过强：保持任务／物理／成功定义不变的配置或依赖修复仍可能成立，需要单独复验。
2. **历史报告发现缺陷，现已修复。** autoresearch/RoboSyn已在 `benchmark/RESEARCH.md`、`benchmark/pilot_results.json` 交付结果，但当时启动器只读trial根目录，最终显示 `needs_followup`，并多追加了总结回合。现支持两个明确报告根，冲突／坏格式交审核而非重跑实验；保留原调度状态，不追改历史费用或结果。
3. **历史指标索引缺项，现已修复。** 此前未索引 ENPIRE `summary.json` 和 RoboSyn `research_out`；现已只读重建RUN索引，排除无效目录和资源软链接，并明确文件发现不等于有效计分。论文始终直接复核底层产物，未依赖旧计数。
4. **现有对照还不能支撑方法领先。** 没有外部系统取得主任务改善，没有完成匹配协议下的完整系统对照，也没有验证新数据获取→真实batch消费→增益；本轮不新增论文中的SOTA或数据方法有效性结论。

本次只读审核未重启实验、修改候选策略、补写原始Agent报告或释放旧预算预留。

## 费用与证据

五arm provider-priced 模型用量合计 **$0.499296852**，另有独立视觉预检约$0.000075；首文本arm仍保留 **$0.14256774** 未结清预留，不算成已结算费用，也不释放。这里不是硬件、人工等全成本。两个原仓库在记录覆盖的源码范围内前后身份一致。

- [独立机器证据快照](evidence/external_native_port_20261016.json)：17项原生measurement、路径／SHA256、逐集／聚合核对、实际加载身份、用量与限制。
- [试跑说明](../EXTERNAL_SYSTEM_PILOT_20261016.md)：适配方式、预算、模型条件。
- [原始RUN.md](../../autoresearch_runs/external_systems_20261016_v1/RUN.md)与[操作者干预清单](../../autoresearch_runs/external_systems_20261016_v1/OPERATOR_INTERVENTIONS.md)。
- [autoresearch/RoboSyn实际报告](../../autoresearch_runs/external_systems_20261016_v1/autoresearch_robotics_robosyn/benchmark/RESEARCH.md)，弥补RUN索引缺项。

复核／重新导出（不调用LLM、不运行仿真、不更改源证据）：

```bash
.venv/bin/python tools/export_external_pilot_evidence.py
```

机器证据不替代正式研究准入审核。下一轮应先解释环境差异、统一起点和评测适配，冻结后再做独立研究重复；不能把此处负结果直接用作我们的系统领先证据。
