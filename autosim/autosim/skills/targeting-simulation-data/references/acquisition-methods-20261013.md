# 自动采集：实现审计与技术路线（2026-10-13）

本页是技术对照，不是新增API或自动注入Agent的常驻上下文。Agent选读三个自包含正文：generating-adapted-demonstrations、collecting-expert-corrections、auditing-demonstration-quality。剩余验收见[总规划](../../../../../plan/MASTER_PLAN.md)，字段以[原生合同](../../curating-simulation-datasets/references/native-data-contract.md)为准。

## 1. 实现到底到了哪里

现有链路：Scheduler委派只读Data → 开发失败获取提案 → Init/Fix接原生接口 → 主控提交collect/prepare_data → manifest/可逆选择 → 数据版本 → 实际training_batch → 原生计分。LLM决定补什么，工具核验身份/权限/字节/计数；Data不是第二调度器。

已经有通用作业、工作量、source-reviewed collection_controls、失败部分保留、原生格式检查、版本/消费与对照基础。**没有一个已验收的框架层通用MimicGen后端或DAgger专家；文字提到方法不等于已接入。**

2026-10-13 RoboSyn新增20episode、1480帧、60视频，全部CPU解码；普通random/reset专家，非失败定向或状态专家。与官方1000条无完全相同动作序列；首条与单条探针动作RMS差约1.57e-7，图像却不同，不能等同整条重复。raw与源仓库未改。

本run的collect builder读episodes/collection_controls，没有消费通用i["seed"]。原生parser已有collection_seed，controls可路由；Main撤去seed后完成默认小试，尚未实证跨批独立seed。Agent可先核查并走已路由控制及精确引用，或修builder，不在通用内核硬编码别名。未启用collection_manifest，缺逐条seed/成功/条件谱系不能事后补造。

当前检查到的训练仍绑定既有数据根，不能归因新增数据收益。两轮已完成训练未完成模型目录交接计分，研究效率仍受系统连接影响；优先恢复付费产物，不连续重训替代闭环。

## 2. 最新可核验方法与采用条件

| 技术 | 环节 | 本项目采用条件 |
| --- | --- | --- |
| [MimicGen](https://mimicgen.github.io/docs/tutorials/datagen_custom.html) | 源分段、对象坐标适配、环境执行 | 示范/对象帧/分段/控制器核验；失败目标选择由上游Agent提供 |
| [DexMimicGen](https://dexmimicgen.github.io/) | 双臂灵巧生成 | 保留协作与来源关系，不独立拼接破坏接触/时序 |
| [AutoData/SkillGen](https://isaac-sim.github.io/AutoData/main/pages/concepts/algorithms.html) | 模块化生成和规划桥接 | 本地任务/embodiment/规划资源匹配，不为所有仿真器默认安装 |
| [DAgger](https://proceedings.mlr.press/v15/ross11a.html) | 学习者状态专家标签与聚合 | 状态专家真实可用，历史/多步标签/mask匹配loader |
| [AIM（2025）](https://proceedings.mlr.press/v267/cai25e.html) | 选择性干预门控 | 有查询/风险证据；人的监督不算免费自动专家 |
| [RESample v4，2026-08-05](https://arxiv.org/abs/2510.17640v4) | 覆盖引导探索与恢复 | 实际coverage/分支/恢复可用，不等同DAgger专家 |
| [WM-DAgger，2026预印本](https://arxiv.org/abs/2604.11351) | 世界模型纠正合成/过滤 | 已有匹配模型与校验；合成视频不充当真实仿真 |
| [DataMIL v2，2026-06-19](https://arxiv.org/abs/2505.09603v2) | 策略相关选择 | 有目标代理和可承担成本；启发式清洗不是其复现 |

来源本次可访问，main/develop文档会变，执行绑定本地revision/许可。MimicGen和DAgger并非新方法；最新应体现可核验扩展及适用性，不只是方法名。

## 3. 分清供给

场景级：从开发失败定义训练初态/对象/视角区域，原生专家或轨迹适配生成完整示范；参数要改变实际条件，保留普通分布。只有结果性按压不足而无对应控制时，先调查或随机池后选。

状态级：冻结learner访问训练状态，查询可靠expert target再聚合。在线查询不要求离线恢复；回放另需恢复影响继续执行的状态。reset脚本、成功检测器和LLM解释不自动提供动作。

策略恢复/自举：训练状态上有可靠成功/恢复才有供给。零成功不能无限筛选；探索、学习者动作、专家纠正分别记来源。世界模型的成本和真实性另列。

## 4. 小试而非方法名

选一种供给和一种目标变化；Data用现有提案说明正确动作、可控/可观察量、来源、工作量、上下文、反例和退路。缺边交Init/Fix，Scheduler决定执行，不新增平行角色/账本。

保存producer/config、实际seed/scene/source-demo/子任务、尝试/规划失败/结果/接受、数据与条件、标签来源/接管、converter/loader。缺字段保持unknown，不把完整物理状态塞进LLM上下文。

ACT单步查询不能生成其余K-1步专家目标。核验同状态续演、有效mask或完整suffix；prefix可作上下文，未重标的位置不直接当BC目标。MimicGen父片段/DAgger策略版本保留谱系，split按组。

## 5. 质量、扩量与贡献

Q0是完整/同步/控制/监督有效，Qt是目标缺口覆盖，下游是真实学习收益。新增检查覆盖嵌套数值/矩阵shape、FPS/PTS；exact-action重复只诊断。曝光/遮挡/近重复按原生语义检查，不统一裁动作或过滤静止。

格式通过不认证专家；源族、动作近重复、合法视觉变化分别统计。检查完小批先登记/消费/计分；未消费时修连接，不增采。扩量看边际覆盖、产出、重复、全成本与旧分布保持，不默认倍增。

固定生成器、初始化、学习量、Q0和评测，比较原数据续训、随机、规则及Agent目标；等时间/尝试和等接受量分开。换供给、目标、筛选、混合或学习配方分别消融。贡献在失败到可控条件/供给选择/数据价值决策的迁移，不在重新发明MimicGen或将生成成功叫SOTA。

尚待真实验收：源适配后端、状态纠正、跨仓库迁移及自主消费/收益；技能/CPU测试不填这些实验空格。
