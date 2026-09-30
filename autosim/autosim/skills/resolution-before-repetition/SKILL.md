---
name: resolution-before-repetition
description: 测量差异不明确时区分噪声、样本不足、错误指标和真实小效应；分布重叠不代表增加样本无用。
scope: general
confidence: methodological-or-borrowed — applicability must be verified in the current run
evidence: Reviewed 2026-09-29; literature links and limitations are stated in the method body. No new benchmark experiment was run.
---

# 判断下一次测量能回答什么

方法性建议。旧版“分布重叠则更多样本无用”不成立；重叠分布仍可能有可检出的均值或成功率差异。

先检查指标是否真的对应问题、样本是否独立/可配对、实际模型是否加载，再估计当前差值与不确定性。预期效应小而样本少时，更多独立 episode 或训练 seed 可能有价值；指标饱和或错误时，扩大同一种测量可能无效。

输出预算内下一测量的目的：提高差值精度、增加独立训练重复、按预定义开发切片定位失败，或添加合法辅助信号。没有明确效果也可报告“当前证据无法区分”，不等于没有合法实验或应结束研究。

不要改正式评分来制造分辨率；任何低成本代理先验证与目标关系，最终仍按冻结协议确认。
