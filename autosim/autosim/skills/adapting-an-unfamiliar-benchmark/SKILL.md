---
name: adapting-an-unfamiliar-benchmark
description: 从当前仓库与实际资源推导可核验的能力图；用于新仓库接入，区分未知、缺条件和已验证。
scope: general
confidence: methodological — requires current source and runtime verification
evidence: Reviewed 2026-09-29; historical incidents motivate the method but do not establish universal defaults.
---

# 读陌生仓库，交付可检验的结论

方法性指引。先读当前版本文档、配置、入口 wrapper，再沿调用链追到生产者和消费者。文档是线索，不是不会出错的权威。

任务列表从 registry/config/文件或原生 API 推导；数据 shape、相机、时序既查真实样本也查部署环境。记录冲突，不把数据或代码任何一方当作永远正确。配置 path 字段只放路径，解释另写。

代码在隔离 checkout 修改，已有资产通过显式资源绑定访问；不要复制整个大仓库，也不要认为复制源码保留了数据和外部依赖。只在授权的资源位置调查，不凭兄弟目录同名就认领数据。

每个结论绑定版本、源码位置、配置或回执。区分未知、发现能力但缺资源、已完成小试和消费者验证通过。“尚未找到”不等于不支持；说明哪次读取或小试能消除未知。

输出任务/目标、环境/资源、采集/转换/训练/评测连接、协议约束、可变参数和待验证问题。benchmark-shapes 提供维度，connecting-native-data-collection 处理数据供给，其他技能按当前缺口选择。
