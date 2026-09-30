---
name: hold-one-variable
description: 识别数据、采样器、归一化与训练配置共同变化的混杂；需要解释单项效果时设计消融，不禁止组合优化。
scope: general
confidence: methodological-or-borrowed — applicability must be verified in the current run
evidence: Reviewed 2026-09-29; literature links and limitations are stated in the method body. No new benchmark experiment was run.
---

# 把隐含改动算进实验

方法性建议。记录候选相对基线的实际 diff：数据、采样路径、normalization、augmentation、更新量、模型初始化、评测与 harness 行为。

最小实验尽量只改变一个可解释因素。组合改进可以保留，只能声称组合有效；要归功于数据、推理或某种规则，另做隔离相应机制的对照。没有预算做完整消融就报告限制，不能把它当作停止所有研究的理由。

关注启用某 manifest 后是否同时切换 temporal sampler、缓存或数据统计量。检查实际 consumer，不只看 Agent 宣称改了什么。每个候选保存代码、配置、数据与评测身份。历史单仓库数值不能给新仓库估计混杂大小。
