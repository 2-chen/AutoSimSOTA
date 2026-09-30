---
name: adding-more-can-subtract
description: 新增数据后退步时检查质量、分布、采样和训练预算；更多数据可能有害也可能有益，需同口径对照。
scope: general
confidence: methodological-or-borrowed — applicability must be verified in the current run
evidence: Reviewed 2026-09-29; literature links and limitations are stated in the method body. No new benchmark experiment was run.
---

# 新增数据可能降低效果

借鉴证据：[RoboCasa365 v1 表 4](https://arxiv.org/html/2603.04356v1) 中，10% 目标数据条件下，Human300 的平均成功率为 40.0%，加 MG60 后为 35.9%。这是特定预训练混合与下游协议的结果，不证明合成数据普遍有害，也不独立证明退步由质量导致。

比较原数据继续训练与加入新数据；明确采样、训练更新量、归一化、模型和评测是否同时改变。检查成功判定、动作/观测一致性、重复度、分布偏移与旧任务遗忘。

最小试验先隔离一种因素：原数据、同成本的新数据混合；如需解释机制，再增加过滤或比例对照。保留新数据身份及 loader 消费证据。若效果不明确，报告不确定性，不把单次下降写成因果结论。

本技能回答“为何混合后退步”；quality-needs-reweighting 处理质量干预，weight-the-small-set 处理实际曝光，不要三者自动叠加。
