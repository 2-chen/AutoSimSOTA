---
name: make-your-prior-earn-it
description: 要声称 LLM 推理、针对性采集或启发式优于普通搜索时，加入合法的随机/简单对照；不是每轮增分的前置门槛。
scope: general
confidence: methodological-or-borrowed — applicability must be verified in the current run
evidence: Reviewed 2026-09-29; literature links and limitations are stated in the method body. No new benchmark experiment was run.
---

# 验证启发式是否真的带来额外价值

方法性建议。针对失败选择一个方向并获得提升，只支持该候选优于所用基线；未必证明“选择得聪明”优于任何合理改动。

预算允许且研究目标包含推理价值时，以同成本的合法随机/简单策略作为对照，配对开发条件并报告不确定性。随机对照也要遵守所有协议与资源限制。

差异不显著不等于等效或推理无价值；小样本只能支持有限结论。没有该对照仍可继续优化，但不要声称已证明 LLM 选法优于普通搜索。历史某次 p 值不能作为新任务放弃推理的依据。
