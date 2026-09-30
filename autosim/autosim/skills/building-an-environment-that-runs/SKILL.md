---
name: building-an-environment-that-runs
description: 依赖已安装仍不能运行时，检查版本、硬件与真实执行；用隔离环境和最小验证修复，不套固定版本表。
scope: general
confidence: methodological — requires current source and runtime verification
evidence: Reviewed 2026-09-29; historical incidents motivate the method but do not establish universal defaults.
---

# 环境要能执行，而不仅是安装成功

方法性指引。检查当前 Python、依赖锁、原生扩展、驱动/GPU 架构、仿真/渲染与资源权限。过去某仓库可用的版本组合只可作线索，不是跨仓库默认值。

用最小真实操作验证：GPU 张量计算、环境 reset/step/读取一帧、模型 forward 或训练导入。is_available、包能 import、命令 exit 0 各只支持有限结论，不证明整条链路可用。

读取完整错误或稳定证据 ID，区分缺包、二进制 ABI/版本不兼容、缺设备/显示/资产、权限与资源不足。先检查官方兼容说明和本机证据，再决定 pin、升级或局部补丁。构建日志中的 workaround 也要核对适用范围与副作用。

只改 run-local 环境；重建环境可能丢失已有配置，应保存解析依赖与构建过程并重新验证相关路径。不要修改系统驱动、共享环境或源仓库来临时掩盖错误。安全隔离导致的问题应按权限边界处理，不以关闭隔离为通用修复。

每次修复说明假设、变更与原失败操作复验；重复同因错误先读历史，避免无新证据循环安装。保留版本、平台、执行证据和可迁移的步骤，不把机器绝对路径固化为通用技能。
