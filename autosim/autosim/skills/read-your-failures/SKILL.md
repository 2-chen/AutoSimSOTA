---
name: read-your-failures
description: 失败后先读稳定证据并判断根因与可复验修复；区分瞬时错误、配置错误、资源上限和未知崩溃。
scope: general
confidence: methodological — requires current source and runtime verification
evidence: Reviewed 2026-09-29; historical incidents motivate the method but do not establish universal defaults.
---

# 让失败改变下一次决策

方法性指引。先读取原 attempt 的 stable evidence ID、命令/配置、日志尾部与因果链、退出码或信号、资源/阶段遥测。不能把进程死亡直接诊断为显存上限，也不能从截断摘要猜根因。

已知配置错误按源码与 consumer 修复；明确资源不足时在本机约束内调整；网络等瞬时故障可作有界重试；未知崩溃先增加最小诊断，必要时重放以验证可重复性，而不是随机改一个参数。

把问题、证据、已尝试修复与待验证条件交 Scheduler/Fix。修补后复验原失败操作，只有新回执能说明恢复；候选评分仍走原生评测和身份链。

相同错误反复出现且无新信息时改调查方向或说明具体边界，不无限重试。暂时失败不应永久删掉模型家族/参数：记录平台、版本、资源与失败条件，条件改变后可以重新研究。
