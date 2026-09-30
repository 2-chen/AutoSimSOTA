---
name: progress-beats-binary
description: 正式成功率过于稀疏时，使用合法的分阶段进度辅助诊断和筛选；不能保证更少样本，也不能替换官方指标。
scope: general
confidence: methodological-or-borrowed — applicability must be verified in the current run
evidence: Reviewed 2026-09-29; literature links and limitations are stated in the method body. No new benchmark experiment was run.
---

# 用进度解释失败，保留正式指标

方法性建议，不保留旧版“统计上普遍更省 rollout”的绝对结论。子目标完成、接触阶段、距离等可能帮助区分失败，也可能与最终成功反向或受奖励投机影响。

先查原生定义和使用权限；验证信号的稳定性与对目标的相关性。仿真内部状态可被协议允许的 evaluator 用于计分/诊断，但不能因此传入 policy；“内部状态指标一律无效”也是错误的。

在开发实验中同时报告正式指标与辅助进度，明确后者仅作诊断还是已校准的筛选信号。进度提高不等于任务成功率提高。不得据此改官方 success、horizon、任务权重或跳过最终确认。

记录样本数、来源和失败切片，有真实数据时画进度图并说明局限。信号不稳定或与最终目标不一致时，停止用它排名而非硬调新指标。
