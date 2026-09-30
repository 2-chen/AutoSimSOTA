---
name: control-same-budget
description: 区分超过发布基线与解释改进原因；因果归因需要计算/数据成本清楚的对照，不强求同时匹配所有预算口径。
scope: general
confidence: methodological-or-borrowed — applicability must be verified in the current run
evidence: Reviewed 2026-09-29; literature links and limitations are stated in the method body. No new benchmark experiment was run.
---

# 选择回答研究问题的对照

方法性建议。发布 checkpoint 可以是合法基线，但“比它更好”不自动证明提升来自新方法，而非更多训练或更多数据。

要解释数据改进，可比较原数据继续训练与数据候选，尽量匹配更新量、模型和评测；要解释计算效率，应比较 GPU 时间/总成本下的效果。不同模型每 step 成本不同，通常无法同时严格匹配 steps 和 GPU 时间，应预先选择主要口径并同时报告其他成本。

连续多轮训练需计入累计预算、筛选成本和初始化历史，不能只算最后一次。额外数据采集也有成本。必要时保留发布策略、同预算继续训练、最终候选三者；预算不足时明确缺少哪种因果对照，仍可报告合规本地提升，不夸大机制。
