---
name: spending-a-budget-across-ideas
description: 当完整实验昂贵时按可验证的低保真信号分配预算、续训和淘汰；不得用跨模型不可比的 loss 草率排名。
scope: general
confidence: methodological-or-borrowed — applicability must be verified in the current run
evidence: Reviewed 2026-09-29; literature links and limitations are stated in the method body. No new benchmark experiment was run.
---

# 在多个假设之间分配预算

借鉴 [ASHA](https://arxiv.org/abs/1810.05934) 的逐级筛选思想；这是超参数调优方法，不保证早期信号能预测当前具身任务的最终成功率。

主 Agent 先测吞吐与最小有效学习量，选择少量可比候选。可按训练量逐级晋升，但冷启动/慢收敛候选应有足够观察窗口；只验证一轮能运行不等于有效筛选。没有可信低保真信号时，宁可少做几项完整实验。

训练 loss 的数值依目标、归一化和模型而异，不能跨算法直接排名；同算法 loss 下降也不保证 rollout 成功。先利用已有学习曲线或有限标定，判断早期开发指标能否预测后期排序，并保留不确定性。

## 正确计算续训成本

若十个候选各到 5 epoch，两个晋升到 20，一个到 50，能完整恢复时总训练量是 10×5 + 2×15 + 1×30 = 110 epoch-equivalents，而不是 50。若晋升重新训练则为 10×5 + 2×20 + 1×50 = 140。不同模型/数据量的 epoch 成本还不同；应使用实测资源估算，并加上评测、采集和修复。

可以调整局部时限、晋升比例与候选数，不把某个 rung 表写死为运行器规则。取消停滞任务前检查真实进度和阶段，长时作业使用现有提交/轮询/取消接口，总预算不可突破。

有意义的停止依据是信息与收益不足、明确资源不可行或合法终态，不是“本轮 loss 没变”。最终赢家仍需完整产物身份与独立确认，低保真分数不进入正式最佳指标。
