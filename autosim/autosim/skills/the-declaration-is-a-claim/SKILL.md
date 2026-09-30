---
name: the-declaration-is-a-claim
description: 验证声明的能力和参数是否对应真实 consumer、合法配置组及可运行值；候选失败后可修复，不静态封死搜索空间。
scope: general
confidence: methodological — requires current source and runtime verification
evidence: Reviewed 2026-09-29; historical incidents motivate the method but do not establish universal defaults.
---

# 声明是待验证的能力主张

方法性分类。源代码可证明名称/映射/默认值存在，有些错误通过阅读就能发现；运行验证补上动态依赖与行为证据，不宣称所有错误只有运行才能发现。

查清 YAML 文件名、配置组名、注册类名和字段值的区别。改变算法组可能需要伴随参数、模型头或 checkpoint；单改一个字符串不能证明支持该算法。

先 resolved config / import / batch forward，按必要程度做短训练与原生评测。一次短试只证明该配置能走通，不证明已收敛。没有完整评测能力的候选标为待修/未评分，交主 Agent 决定修复、预算或换路线，不静态认定永远不可用。

声明的采集/转换/训练参数必须追到实际消费者；新数据输入和缓存失效尤其需要验证。运行失败时撤销受影响的已验证结论，保存证据及复验条件；新值来自源码/实验探索而非仓库专用硬编码。
