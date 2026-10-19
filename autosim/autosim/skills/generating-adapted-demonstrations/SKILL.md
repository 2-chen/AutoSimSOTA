---
name: generating-adapted-demonstrations
description: 有少量合法成功示范时，调查并连接 MimicGen 类对象坐标/子任务轨迹适配；适用于对象布局扩增或缺原生脚本专家，不把任意视频、关节数组复制当成新示范。
---

# 用少量示范生成可执行的新轨迹

正确行为来自源示范与可核验的几何适配，不来自LLM猜动作。Data调查、Init/Fix实现隔离桥接、Main通过既有原生阶段执行；不另建采集调度器。依据：[MimicGen环境接入](https://mimicgen.github.io/docs/tutorials/datagen_custom.html)、[环境接口](https://mimicgen.github.io/docs/modules/env_interfaces.html)。

## 先查最小接口

从源码找成功源episode及许可、对象位姿/坐标系、末端目标与夹爪、子任务分段、目标位姿到控制动作的转换，以及原生成功判据/记录器。源视频或仅joint actions不足以直接做对象坐标变换；有FK/IK或原生控制器时先核验转换，而不是给关节角加物体平移。

已有MimicGen接口优先复用 `MG_EnvInterface`、`TaskSpec` 和源数据准备链。没有接口时，提出最小桥接的源码落点和原生小试；装包成功不等于适配成功。特权对象位姿可以供生成器使用，不擅自增加机器人策略的评测观测。

## 适配与执行

设子任务起点的源/新对象位姿为 `T_O_src`、`T_O_new`，源控制器末端目标为 `T_E_src(t)`；一致世界坐标下，候选目标为 `T_O_new @ inverse(T_O_src) @ T_E_src(t)`。这是刚体适配思路，不保证接触正确。核对目标/实测位姿、绝对/相对动作、旋转约定、控制频率和夹爪时序。

选择匹配当前对象状态的源片段，保持子任务顺序；连接动作经过真实控制器/规划器执行，再由原生判据接受。记录新状态、图像和执行动作，不把源观测贴到新动作上。接触关系已经改变时旧片段可能失效，不能称为任意失败状态的恢复专家。

双臂任务核对两臂片段来源与协调约束，可复用 [DexMimicGen](https://dexmimicgen.github.io/)。连接段碰撞且已有规划底座时，调查 [AutoData的SkillGen路线](https://isaac-sim.github.io/AutoData/main/pages/concepts/algorithms.html)，不是盲增插值步数。AutoData/Isaac Lab Mimic仅是条件性后端，版本、接口、GPU/规划器资源需本地确认；不为其他仿真器强制安装Isaac Sim。

## 小试接到数据候选

用独立训练初态和有限尝试验证“源片段→变换→真实执行→成功→记录→native loader”。保存源demo/subtask ID、变换/配置、实际seed与场景、生成器版本、拒绝原因及数据身份；同源子孙保持同组，不能跨训练/验证制造独立性。

现有提案使用 `supervision.kind=demo_adaptation`，在 `data_context` 说明帧、桥接与分段；controls只填已证明的输入。目标来自开发失败，不使用测试seed。登记后验证真实batch和同协议分数；生成成功率不代替策略收益。

先比较同生成器的随机与目标布局，保持共同Q0、源示范、初始化和学习量。若换生成器，注明供给变化，不能只归因于目标选择。缺源片段/动作转换、低产出或消费不通时，修最短缺边、退回原生专家/随机池筛选或说明边界；不无限重生成。
