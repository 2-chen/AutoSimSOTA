---
name: where-successes-come-from
description: 需要新训练轨迹时判断动作来自专家、规划器、人工或现有策略；成功判据、录制能力和自主采集能力分开验证。
scope: general
confidence: methodological-or-borrowed — applicability must be verified in the current run
evidence: Reviewed 2026-09-29; literature links and limitations are stated in the method body. No new benchmark experiment was run.
---

# 谁能产生训练轨迹

方法性分类。成功判据回答“这次是否完成任务”，不提供执行动作；env.step/视频录制接口也不等于自动演示采集器。

查当前任务的动作提供者：脚本专家、运动规划器、人类遥操作、已有策略，或重组源演示的生成器。分别核对任务覆盖、资源、初始化条件、观测/动作格式和输出消费者。只有人工接口时不能声称无人采集；已有策略零成功时，成功过滤自举通常无法启动。

专家标准 reset 可用不证明它能恢复任意偏离状态；生成器还可能需要源轨迹、分段标注、物体状态与适配接口。固定初态并不妨碍采集，只限制分布覆盖；done/terminated/truncated 不能直接当成功。

BC 常用成功演示，但在线/离线 RL 等也可能利用失败经验。决定保存什么应来自训练算法与协议，不要先把所有非成功数据删掉。

输出生产者与录制/转换/loader 的证据链、已验证能力、未知项及最便宜验证。具体接通方法见 connecting-native-data-collection；没有实现证据时返回未知，不把“尚未找到”改写为“不支持”。
