---
name: finding-how-a-benchmark-runs
description: 查找当前版本原生采集、转换、训练与评测入口，并核对 wrapper、模型家族与实际调用；不从文件名猜命令。
scope: general
confidence: methodological — requires current source and runtime verification
evidence: Reviewed 2026-09-29; historical incidents motivate the method but do not establish universal defaults.
---

# 找到实际运行入口

方法性指引。优先读 README/指南、脚本说明、配置和 wrapper；文档可能过时，最终要追当前源码与小试。一个仓库有多个 train.py，文件名不能决定哪个匹配当前任务和权重。

逐一记录入口、解释器/cwd、参数/配置解析、必要资源、动作提供者、输入产物与输出位置。wrapper 可能同时采集、转换和清理；说明覆盖的阶段，避免重复转换或误删资源。

对 vendored 代码沿调用链判断是否真正参与当前运行，既不能仅因在仓库内就认作入口，也不能仅因第三方名字就排除其用途。检查发布模型/数据属于哪个策略家族，但不将发布内容当唯一允许研究的模型。

输出具体文件与可验证的调用，不只给目录；CLI help 若会初始化设备也须按受控操作执行。少量前向/加载/rollout 验证支持程度，完整训练另行预算。

需要人工遥操作可记为“有人参与的采集能力”，不能写成“没有采集入口”；没有无人专家时说明缺口。unknown 与 unsupported 分开，下一步调查由主 Agent 决定。
