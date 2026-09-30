---
name: diversity-before-volume
description: 采集预算有限时比较新场景、物体、初态或任务覆盖与重复演示；适用于允许改变训练分布的协议。
scope: general
confidence: methodological-or-borrowed — applicability must be verified in the current run
evidence: Reviewed 2026-09-29; literature links and limitations are stated in the method body. No new benchmark experiment was run.
---

# 比较覆盖与重复数量

借鉴证据：[Data Scaling Laws v4](https://arxiv.org/abs/2410.18647v4) 在真实机器人模仿学习中研究环境/物体多样性及每配置演示数量，观察到增加后者的边际收益递减。它不是所有仿真任务的最优配额证明；RoboCasa365 的仿真任务多样性研究见 [调研](../choosing-simulation-improvements/references/literature-review.md)。

先定义实际覆盖维度和训练许可范围：物体/布局、初态、目标组合、接触/恢复状态。seed 数量不是覆盖度本身，同 seed 或不同 seed 的效果由环境实现决定。不能以测试集失败作为采集分布标签。

在相近采集与训练预算下比较“重复已有配置”和“扩展一种覆盖”。同时记录成功产出率、质量、有效样本和各开发切片得分，避免把难配置低产出误认为低价值。

如果已有数据不足以学会基本动作，增加同分布演示仍合理；如果新随机化超出任务定义、改变动力学或使专家失效，缩小范围。没有统一的场景数、每场景演示数或必胜顺序。
