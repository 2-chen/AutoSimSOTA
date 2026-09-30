---
name: report-paired-and-intervaled
description: 比较候选成功率时保留 episode 对应关系、报告差值不确定性，并区分开发选优与独立确认。
scope: general
confidence: methodological-or-borrowed — applicability must be verified in the current run
evidence: Reviewed 2026-09-29; literature links and limitations are stated in the method body. No new benchmark experiment was run.
---

# 报告配对结果与不确定性

方法性统计指引，具体方法的假设需要核对。相同 seed 不必然产生相同状态；固定初态、任务、环境版本与随机源控制须有证据。配对可降低部分方差，不会消除全部初态或策略随机性。

保存每 episode 的样本身份、结果与失败/截断原因。成功率注明成功数/有效总数；异常退出不能无依据记失败或剔除。聚合要遵守任务权重，并区分 episode 不确定性、训练 seed 变化和跨任务差异。

单臂二项比例可用适用的 Wilson/精确区间；比较时关注差值区间。配对二元结果可用不一致配对表及适当的配对检验，或按实验独立单位构造 bootstrap。存在任务/训练 seed 聚类时，不能把所有帧或相关 episode 当独立样本。

两条单臂区间重叠不等于差异不显著；p 值不显著也不证明没有效果。多轮搜索和反复查看结果会增加选择偏差，预定确认规则或采用有效的序贯方法，不反复运行固定样本检验直到过线。

开发集用于研究；最终留出只按协议作独立确认。没有能力估计可靠区间时，报告原始计数、比较口径与局限，不制造精确显著性或声称 SOTA。
