---
name: making-a-stage-run
description: 原生入口运行失败时按证据定位参数、环境、资源与消费者连接；修复后复验原操作，不靠猜通用配置名。
scope: general
confidence: methodological — requires current source and runtime verification
evidence: Reviewed 2026-09-29; historical incidents motivate the method but do not establish universal defaults.
---

# 让原生阶段真正跑起来

方法性故障分类，不限定只有三种错误。执行时保存完整日志、原命令/cwd/config、退出或超时原因、产物和稳定 evidence ID；摘要保留 traceback 的因果链和尾部，必要时按 ID 读正文。

- 参数解析失败：读当前 CLI/help/config schema；使用完整 flag，注意 argparse 缩写可能把名字错误表现成值错误。
- import/动态库/渲染/设备失败：检查实际解释器、环境和权限，不靠换数据参数修复。
- 路径/配置错误：沿真实 consumer 找到 resolved 值、cwd、资源绑定与缓存；不要猜 data_root 或 output_dir。
- 训练/执行已开始后失败：检查数据样本、模型兼容性、数值、内存、作业信号与内部阶段，不把所有退出归为超时。

路径按 shell 与配置解析器各自的规则传递并小试。非 ASCII 路径不必然不可用；只有实际复现解析问题时才考虑安全别名。相对路径可以合法，只须绑定 cwd；不要把宿主绝对路径塞进不同命名空间。

HDF5 句柄在 spawn/pickle 路径下可能失败，num_workers=0 可作为诊断/临时修复；独立 worker 内延迟打开文件也可能是合适方案，不把零 worker 变成统一永久规则。隔离所需的环境变量不能为省事随意撤销。

缺数据先检查授权资源绑定和官方来源；不假设数据必在邻居目录。输出位置可能由代码生成，追实际构造函数；只能写 run-local 位置，不重定向回源仓库。输入 dict 的缺失、空值、0 和 False 要按 schema 区分。

修复形成可审计差异，重试原失败阶段并比较新回执；后续还须验证实际消费者，文件产生不等于训练/评分成功。候选已回滚时通过新候选复验，不能把 Fix 的说明或探针当正式得分。
