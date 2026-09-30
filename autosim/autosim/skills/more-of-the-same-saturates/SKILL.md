---
name: more-of-the-same-saturates
description: 增加同分布数据收益趋缓时，区分数据饱和、欠训练、采样稀释与评测噪声，再决定是否扩采。
scope: general
confidence: methodological-or-borrowed — applicability must be verified in the current run
evidence: Reviewed 2026-09-29; literature links and limitations are stated in the method body. No new benchmark experiment was run.
---

# 判断是否真的出现数据收益递减

借鉴的是数据规模研究中的条件性现象，不是“几百条就饱和”的规律。来源及真实/仿真范围见 [文献调研](../choosing-simulation-improvements/references/literature-review.md)。

先核对新增样本是否被读取、曝光是否足够、训练量是否随数据扩展而不足。固定 steps 与固定 epochs 回答不同问题，都应报告累计计算量。

用少量递增数据规模与足够训练的对照估计开发集收益/成本，并附不确定性。一次平坦结果可能是低分辨率、欠训练或随机波动；不能据此宣布更多数据永远无用。

如果边际收益相对成本很小，可优先改变覆盖、质量、控制或表征；如果仍欠拟合，则考虑延长训练。输出下一批数据或下一轮训练的预计信息价值及停止依据，而不是绝对样本阈值。
