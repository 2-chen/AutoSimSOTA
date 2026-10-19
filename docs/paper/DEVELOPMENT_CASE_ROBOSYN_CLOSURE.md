# RoboSyn 前瞻轨道：产物恢复、局部开发提升与数据供给试点

更新：2026-10-13。这是可核验的开发证据，不是正式FDAS主表或无人工干预trial。

## 1. 已完成的有效评测

研究：[robosyn_data_closure_20261012_v2](../../autoresearch_runs/robosyn_data_closure_20261012_v2/RUN.md)。任务click_bell；原生random开发协议、20 episode、评测seed标签0。

| 测量 | 学习/策略关系 | 开发成功数 | 原生评测时间 | 回执核验 |
| --- | --- | ---: | ---: | --- |
| baseline | 本轨道完成40,000更新的本地ACT；部署执行16个预测动作 | 4/20（20%） | 192.3秒 | consistent，issues=[] |
| round_2_handoff_correction | 第2轮完成训练的独立producer；仅恢复原模型包并评测 | 4/20（20%） | 192.3秒 | consistent，issues=[] |
| round_3_handoff_correction | 第3轮完成训练的独立producer；仅恢复原模型包并评测 | 1/20（5%） | 213.5秒 | consistent，issues=[] |
| round_4_handoff_correction | 第4轮已经完成40,000更新的独立producer；仅恢复完整模型包并评测，无重新训练 | 4/20（20%） | 190.3秒 | consistent，issues=[] |
| round_5 | **复用baseline同一权重**；Agent只将部署执行长度16→8，没有新训练或新数据 | 2/20（10%） | 211.4秒 | consistent，issues=[]；frozen parent/identity核验通过 |
| round_6_handoff_correction | Agent选择更新边界计数候选，局部训练2,000步/batch8；同producer完整bundle恢复，没有再次训练 | **6/20（30%）** | 179.9秒 | consistent，issues=[]；Main随后显式adopted |

这些行不能合并成多个独立训练seed。第2/3/4轮是实际不同的训练producer，均为同一个训练seed标签下的见证插桩开发候选，其策略身份不同；只有baseline与round_5是同一冻结权重的部署比较。相同seed标签不证明实际初态配对；本轮未做显著性、独立确认或公开SOTA结论。旧24/100→32/100是另一轨道，不与这里的20 episode结果直接计算增益。

新30%是同开发评测口径下的局部观察，比本轨道20%基线高10个百分点，已被系统保留。该候选同时变动插桩源码与局部学习量，不能将提升归因于插桩、较短训练或新数据，也不是等学习量的数据方法对照。采样和实际消费尚未认证；结果用于开发进展与运行成本案例，不填写FDAS主表。

### 原生身份与证据索引

- baseline：train `5ed99d12749a4d219de3e8604961b49f`，evaluate `25ad33587f014e51a7d215ab9f8f059f`，策略目录SHA `73937330606bf2a59a7f8c37ee2f1c5c9316b0e2f02ff5cb3f589069c37c6a88`。
- 恢复第4轮：原train `6565927a4edd431e8f8935f778232aca`，新evaluate `157f92f7da84483bb71e4c8fbc0495e3`，策略目录SHA `764618a7ee7d1f17c3d4c1d56f36a874fd08f3f2fefa39b43bd6e1e94d477263`。先前3077.5秒的完成训练保留，新增的是190.3秒评测，不是重训后的新producer。
- 恢复第2轮：train `47a20f66bcc54162aa6e5cadf35762c4`，evaluate `b157074684b542bfa456b9fbfbda09ff`，策略目录SHA `7788b2d8d4dc1a22fec0e0d6067e5949bddc35ea0e4ab65d58786d9bb65781a2`。
- 恢复第3轮：train `e89adb4c69be407ab2388c2f2814fb2d`，evaluate `0bb18c3cdcf5419aba2b7a1a48e19082`，策略目录SHA `97afb94105c1648c13f562b0a75db01f25991463344b2558eeebca2b6368b446`。该5%负结果保留，不能只报告另外两项20%。
- round_5：复用baseline策略，evaluate `57ee5bbe2f2c4fc0a54687c68da16a46`；实际部署idea为`replan-horizon-8-deployment-reuse`。原生加载、完成rollout与结构化metric身份一致；controller history记录`undone=true`、best仍baseline。
- 第6轮：train `2fdbdcd9a31c4d039a3a2f5725506333`，evaluate `9a9caa688a274ccca7a8b2d20d96ad08`，策略目录SHA `05da96ca6cdc1c1c2e03d0119cc822ac161e1fa3b09dad173feee511ee3f30da`。原round_6产物形状失败保留；新增补充分数6/20，13:22 Main通过adopt_linked_measurement显式保留，best.score=0.3、原baseline不改。
- 六份有效measurement均在该run的`research/derived/measurements/`；逐阶段receipt、原生封存log、result archive、source snapshot与handoff proof保留。原第2/3/4/6轮失败没有改写，补充测量另存。

[机器可读证据摘要](evidence/robosyn_closure_20261013.json)从上述六份measurement和原生回执只读提取，并逐项通过receipt_verifier。另列实际model.safetensors的字节hash，避免将包含配置的目录hash与权重身份混为一谈；只导出聚合计数/成本/身份，不复制视频或原始数据。未可靠取得的原2k训练过程时长记null，不填0或推算成本收益。

可写出的观察：更频繁重规划并非这个策略的稳定提分技巧；这次单seed、20回合的开发尝试得到负结果并被回退。不能据此推广“执行长度16普遍优于8”，也不能将第4轮相同20%解释成见证插桩改善了学习。

## 2. 已完成的数据供给试点，不等于数据收益

系统原生专家collect `0cc6135b20ee4705ae5ba4429baae992` 正常完成：native日志记录20次尝试保存20条成功示范；20个episode、1480个时间步、三相机共60段视频，整个采集进程1025.1秒。

操作者只读检查实际parquet数值/形状/时间、全部视频解码/帧数/PTS/FPS，通过；当前池有20种exact-action序列。它不证明20个独立场景族、物理成功复验、失败定向覆盖或已经被训练消费。保存成功是专家采集产出，不是学习策略的20/20成功率。

本批为普通random/reset专家，不是MimicGen/DAgger实测。目前这些计分measurement的数据版本消费均未正式认证；上述新增数据尚未形成新的数据候选得分。故可写“原生供给与可读性试点”，不能写“高质量定向数据带来性能提升”。

2k新训练的只读日志诊断发现29条并发JSON拼接坏行，可解析记录缺阶段/reader字段且setting仍是场景标签。训练/评测结果可核验，不代表这些日志能认证实际版本到training_batch的消费。原数据、原日志不修改，具体连接缺口交由Data/Fix继续修复。

## 3. 自主性和干预

开发者修复历史源码准入、完整bundle复用及重复状态审核，重载原服务，并按用户授权将原账本模型上限80→90美元。GPU仍24h、墙钟原起点/期限和全部消耗/未知预留保留。干预详见[原记录](../../autoresearch_runs/robosyn_data_closure_20261012_v2/OPERATOR_INTERVENTIONS.md)。

后续又补前瞻数据选择封存及同producer恢复的数据绑定，并受控重载；这不改变上述旧unknown消费状态或已计分结果。GPU数据arm的端到端验收仍未完成。

修复后Main自行选完整bundle、首次传错输入label后自行读框架证据修正，完成第4轮计分；随后自行选择部署长度8的候选、评测并回退。第6轮ABI缺口被审核拒绝后，Main自行CPU读源码/调用点，改选2k局部工作，完成原生训练、同producer恢复计分并保留30%。开发者未代选checkpoint、训练配方/采集目标、启动这些评测或填写成绩。这是“维护后的自主实验”，不回溯包装为冻结零干预trial。

## 4. 仍需新增的核心实验

先完成真实数据登记/实际batch曝光和一个完整新数据候选，再在同一生成器/质量底线/初始化/学习量/评测下完成原数据续训、随机补采、失败定向补采。需要新科学对照，不把当前部署消融或恢复试点填入FDAS主表。
