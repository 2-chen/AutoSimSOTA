---
name: quality-needs-reweighting
description: 混合质量或来源的数据表现不稳定时，比较原样训练、过滤与重加权；质量标签和收益都必须验证。
scope: general
confidence: methodological-or-borrowed — applicability must be verified in the current run
evidence: Reviewed 2026-09-29; literature links and limitations are stated in the method body. No new benchmark experiment was run.
---

# 验证质量干预，而非默认必须重加权

robomimic 研究了不同质量的人类演示与算法选择的关系，提示质量和设计选择都会影响结果；不支持“混合质量数据不能用于普通 BC”的绝对结论。[原始研究](https://arxiv.org/abs/2108.03298v2)

先检查质量依据：任务成功不等于轨迹没有停顿/错位，短轨迹不必更好，稀有难例不必更差。把损坏或语义不兼容与可学习但次优的数据分开。

最小对照是同训练预算的原样混合与一种过滤/权重策略，保留纯原数据参照。有条件时控制来源、数量和覆盖，避免过滤掉全部难场景后仅在简单开发集上变好。权重设计依据训练/开发数据，不能来自最终留出分数。

优先尝试可解释的来源/质量分组；昂贵的数据价值模型需要额外训练与验证预算，不能仅因论文有效就默认部署。无稳定质量依据或没有改善时，保留简单 BC 路线。记录被删样本、有效曝光、旧任务保持和不确定性。
