---
name: generator-success-is-not-data-value
description: 采集器成功率与训练数据价值分开评估；用于选择生成配置、比较产出率和下游策略收益。
scope: general
confidence: methodological-or-borrowed — applicability must be verified in the current run
evidence: Reviewed 2026-09-29; literature links and limitations are stated in the method body. No new benchmark experiment was run.
---

# 采集成功不等于策略变强

[MimicGen 官方项目](https://mimicgen.github.io/) 展示了由少量源演示生成新 reset/物体/机器人条件的数据并训练策略，说明数据生成和策略学习是两个需分别验证的环节；不能把生成成功率直接作为最终目标。

分别记录尝试数、接受数、成功依据、有效样本质量/覆盖、采集成本；再测训练后同协议的策略表现。成功率高可能只是任务容易；低产出可能有新覆盖，也可能纯粹浪费，不能提前认定价值。

在采集与后续训练预算都可比的条件下小规模测试候选，保留失败尝试计数，确认实际 loader 使用新数据。只在开发集收益/信息量支持时扩量。视频证明采集行为，不证明下游提升。
