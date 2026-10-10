---
name: accelerating-policy-experiments
description: 实验吞吐低、每个候选都重训或评测比训练还贵时，定位关键路径，复用合规产物并比较缓存、批处理、精度和评测开销；不降低正式计分要求。
---

# 让同一预算产生更多有效实验

方法建议，不是执行权限或默认加速开关。先判断当前时间花在等待数据、计算、仿真、加载、保存还是重复规划，再选一个可逆改动；不要先给每个仓库装新训练框架。

依据：2026-10-06 原始来源调研及工程推断，尚无本机增分/提速测量。论文或文档展示的能力必须在当前版本复验。

## 先复用，再加速

环境与资源连接已通过且指纹未变时，复用验收证据，不重复安装。训练已完成时先核对完整产物目录、所选训练终点与实际加载，不能因为交接失败重新训练。合法的执行 chunk、支持的采样步数、已有 EMA/raw 或开发侧 checkpoint 选择可复用权重比较；改变网络、预测长度、传感器或归一化则通常不是零训练候选。按需读 [训练身份](../validating-policy-training/SKILL.md) 和 [动作反馈](../tuning-action-chunks-and-feedback/SKILL.md)。

相同实验的复用身份至少包含代码/配置、训练数据与预处理、初始化/训练产物、评测实现/任务/初态及随机性设置。仅路径相同不能复用；两个随机重复实验也不能合并成一次。无正式确认时不把开发选优结果升级为正式最佳。

主 Agent 的正式候选接口：若 `research_options.frozen_policy_trials` 提供已有本地训练测量，可显式选择 `run_the_loop` 的 `reuse_policy`，传入该测量的 `label` 和 `measurement_sha256`，并省略 `training_settings` / `job_id`。仅 code 类部署候选；执行器重验原训练进度/产物选择、冻结权重与协议，在新候选源码快照上只执行一次新评测，不按仓库名或关键词自动决定跳过训练。训练、数据或权重变化应走普通训练路径；图执行缺输入边、数据版本复用缺绑定时明确报告边界。普通“训练＋评测”即使配方相同也会生成新权重，不能声称与 incumbent 的字节完全相同；该能力的上线效果以实际回执为准。

## 找到真正的瓶颈

用已有日志或有界 profiling 区分冷启动与稳定运行，记录 data wait、forward/backward、env step/render、加载/保存、评测的时间和峰值 RAM/显存。GPU 异步操作会让未同步的计时失真；计时边界要合理，不能每一步强制同步反而拖慢整个作业。[PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

| 实测瓶颈 | 小试候选 | 必须复验 |
| --- | --- | --- |
| I/O、图像解码或数据等待 | 原生低维/图像缓存、少量 loader worker、pinned memory、异步传输 | RAM/worker 放大、实际样本 ID、更新步吞吐；新增数据使旧缓存失效 |
| 模型计算 | 实现支持的 AMP、静态形状编译，必要时调整 micro-batch | 有限 loss/梯度、实际更新和动作误差、开发 rollout、冷启动/编译回本 |
| rollout/推理 | 原生并行环境、跨环境批量推理、eval/no-grad、同一次决策内共享视觉编码 | reset/历史独立、episode 身份、环境后端与物理语义；不跨状态复用过时特征 |
| 保存/媒体/评测 | 开发评测频度与预定少量 demo；原生保存频率/保留策略 | 终点、best、续训点与正式证据完整；不删必需媒体或跳过最终确认 |

缓存模式以本地源码和内存估算为准。robomimic 的公开配置展示低维/全量缓存及训练内 rollout/save 控制，这是能力例子，不是所有仓库都支持这些参数。[官方配置](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/config/base_config.py)

worker 数、并发环境数和 batch 不设跨仓库常数；从可完成的小试寻找吞吐拐点，而非以显存占满为目标。单 GPU 同时训练和仿真可能竞争显存/渲染，不默认并发更快。ManiSkill 支持 GPU 向量化，但不能据此把其他物理后端直接替换为 ManiSkill。[官方接口](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html)

AMP/编译可能不适合接触敏感控制或短任务。明确学习率、有效 batch、更新数、采样曝光及 RNG；增大 batch 后不能仍说训练语义完全相同。遵循本机版本支持范围，不为加速盲目升级整个环境。[PyTorch 性能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

## 用最少评测回答当前问题

开发侧可以先做接口小试，再在预定小样本上排除明显无效候选，然后扩大样本并确认赢家。小样本通过只表示值得继续，不代表显著提升。初态配对须有状态身份；相同 seed 不一定相同状态。不得减少正式 episode 数、只测简单任务、改 horizon 或剔除失败来获得“提速”。按需读 [配对与不确定性](../report-paired-and-intervaled/SKILL.md)。

预留赢家复评、对照和所需 demo 成本。新加速能力若执行接口不支持，可提出隔离候选补丁与验证，但不能虚构工具已具备它，也不能擅自清理资源。

## 保留什么证据，何时停止

交付：瓶颈证据 ID、候选/对照身份、实际更新量、总耗时与冷启动成本、RAM/显存、开发指标/原始计数、数值或协议风险、回退点。纯工程提速比较同工作量；若用于训练更多步，另记总计算和增分，不能把额外训练收益归给缓存。

回本估计：一次性成本 / 每次节省时间；预计剩余重复次数不足以回本、吞吐改善不可靠、动作/分数退步或证据链破坏时回退。吞吐提升与成功率提升分别报告。完整依据与候选清单见 [增量调研](../choosing-simulation-improvements/references/score-efficiency-20261006.md)；正文已可独立使用，链接不自动加载。
