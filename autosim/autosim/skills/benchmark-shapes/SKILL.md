---
name: benchmark-shapes
description: 调查仿真仓库的任务、指标、观测、控制、数据与资源差异；所有维度都需要当前源码与运行证据，不预设通用默认值。
scope: general
confidence: methodological-or-borrowed — applicability must be verified in the current run
evidence: Reviewed 2026-09-29; literature links and limitations are stated in the method body. No new benchmark experiment was run.
---

# 仿真仓库有哪些不同

方法性分类，不是穷尽枚举。旧版把“固定成功谓词、双视角 RGB、按 horizon 结束”等写成 universal，现撤销：它们都不能作为新仓库默认值。

| 维度 | 需要查明 |
| --- | --- |
| 任务与优化目标 | registry/config/数据元信息如何指定任务；单任务、多任务、持续学习或泛化协议 |
| 官方指标 | 成功率、回报、子任务进度等；计算位置、聚合方式、有效 episode 和异常处理 |
| 终止与初态 | terminated/truncated、horizon、固定状态或采样分布、seed 实际影响哪些随机源 |
| 数据供给 | 发布数据、脚本/规划专家、人工演示、生成式转换、策略 rollout 或原生 RL |
| 观测 | RGB/深度/点云/状态/语言/历史；传感器权限、相机名称、时序、标定 |
| 控制 | 关节/eef、绝对/增量、单位、夹爪约定、动作频率；不能从维度猜含义 |
| 数据消费 | 文件格式与字段、转换、归一化、split、真实 loader 的配置根与缓存 |
| 策略资源 | 模型源码、匹配权重、训练预算、是否能从头训练或恢复 |
| 执行环境 | 仿真引擎、驱动/渲染、设备、资产、许可、CPU/GPU/内存/存储与并发 |
| 有效比较 | 评测版本、数据权限、checkpoint 选择、任务权重、配对状态与预算口径 |

每项注明源码/配置/样本/回执来源与未验证之处。文档和源码都可能过时，实际 consumer 及运行结果负责验证。数据样本描述已有数据，当前环境描述部署条件，冲突应调查而非任选其一。

成功检查可以合法读取仿真内部状态；这不自动允许策略读取同样信息。一个终止标志也不必然表示成功。即使成功率同名，任务分布、选择规则或物理设置不同也不能直接比数值。

交付最小能力图与缺口，不生成仓库名称→命令/默认维数的硬编码表。需要接通采集时读 connecting-native-data-collection，需要策略部署核对时读 aligning-policy-inputs-and-actions。
