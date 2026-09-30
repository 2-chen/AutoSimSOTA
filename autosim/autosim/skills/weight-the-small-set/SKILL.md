---
name: weight-the-small-set
description: 少量新增数据混入大数据集后无效果时，检查实际抽样曝光，再比较合法采样比例；不预设固定混合权重。
scope: general
confidence: methodological-or-borrowed — applicability must be verified in the current run
evidence: Reviewed 2026-09-29; literature links and limitations are stated in the method body. No new benchmark experiment was run.
---

# 检查小数据集的真实曝光

方法性指导；过去 RoboSyn 的数值不作为通用比例建议。历史材料仅保留在 benchmark-scoped 的 robosynchallenge-measurements 中，复用数值前需核对原始回执。

按实际采样单位计算比例：episode、frame、sequence 或 source。自然频率不一定等于 episode 数量占比；长度、过滤、replacement、distributed sampler 和 horizon weighting 都会改变它。读取 loader 的样本身份/计数验证，而不是只看 manifest。

比较原数据、自然混合和一个有理由的增强曝光候选，尽量保持 sampler 的其余行为与训练成本一致。不能统一设 0.5，也不能在协议不允许时新增权重机制。

曝光太少可能没有可测影响，曝光太多可能过拟合新集合、降低覆盖或遗忘旧任务。报告开发效果、实际比例、训练更新量与不确定性。无收益时先排除未消费/欠训练，再考虑比例是否值得继续搜索。
