---
name: budget-attempts-not-episodes
description: 采集有失败、过滤或内部重试时，分别限制尝试数、接受数和实际时间，估计可承担的有效数据产出。
scope: general
confidence: methodological-or-borrowed — applicability must be verified in the current run
evidence: Reviewed 2026-09-29; literature links and limitations are stated in the method body. No new benchmark experiment was run.
---

# 预算不能只数成功 episode

方法性指引，不使用过去某仓库的产出率作为默认值。读取原生命令：请求 N 表示尝试次数、成功条数、seed 数还是任务数？内部是否一直规划直到凑够成功数？

先做有墙钟/资源边界的试采，记录尝试、接受、失败类别和总成本。用本次产出率及波动估计下一批，而不是统一乘 1.5 或 2。零成功时先诊断专家/配置/资源，不能无限重试。

约束应包含接受目标、失败尝试上限、墙钟/GPU 等总资源和可取消条件；原生参数表达不了的边界交受控执行器/主 Agent 处理，不凭空发明 flags。CPU 规划也耗时，GPU 小时不覆盖全部成本。

训练、转换、存储与评测同样计入实验估算。超局部额度可以申请调整，但不能超仓库总硬限；小批有效产物可否续用取决于协议与数据完整性，不把部分完成默认为全量成功。
