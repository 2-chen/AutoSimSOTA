# 正式搜索后的独立验证

本模块位于已冻结的 `research/` 核心之外，不改变运行中的训练、任务环境或裁判。CPU 测试不代表真实训练/评测通过。

## 已实现的入口

在 `autosim/` 目录使用 `../AutoSimSOTA/.venv/bin/python`：

```bash
python -m autosim.research_validation.cli compare-initializations --first EVAL_A --second EVAL_B
python -m autosim.research_validation.cli determinism --task click_bell --checkpoint CHECKPOINT
python -m autosim.research_validation.cli lock-repeats --task click_bell
python -m autosim.research_validation.cli train-repeats --task click_bell
python -m autosim.research_validation.continuation
```

默认研究目录为 `output/robosyn_general_20260905`；可显式传 `--workspace`、`--output`。GPU 操作使用现有 suite/GPU 互斥，不与生产训练并跑。

## 一致性检查

同 checkpoint 在独立进程上重复同一预声明的 5 局开发 bank；逐局检查初始允许观测哈希、成功与步数，并量化关节、物体平移和旋转矩阵差异。哈希匹配只是小样本证据，不保证所有隐藏物理/RNG 状态一致。缺失记录为未知；没有事后编造的容差及“通过”结论。已有合并哈希不能单独推断 RGB 差异。

实际 pilot 的按铃、抽屉数值检查见输出目录 `pilot_initialization_numeric_audit.json`。它不是同 checkpoint 复跑，也不是最终测试。

## 三个训练 seed

正式两轮开发完成、最终测试尚未开始时，锁定官方数据受控重训、自动选择、随机补采对照三个训练配方。源 seed 为 1000，各追加 1001、1002，共六份新模型；没有重复搜索或补采，也不从源 seed 的最终 policy 权重初始化。

保留源训练分段：若源模型先 20k 初筛再续训至 80k，追加 seed 也从头训 20k，再仅从该 seed 自己的 checkpoint 续训至 80k。不能将直训 80k 偷换成相同执行配方。锁定实际数据内容 ID、混合清单、每段配方、权重和模型配置；输出所有 seed，不按最终分数挑选。

每任务额外训练进程预算 48 小时，**另计于原每任务 40 小时研究预算之外**。失败及中断保留，不无条件重试。完成训练不等于验证提升；最终层级统计和语义验收仍需实现/执行。

## 后台队列

`continuation` 等生产队列完成或明确停止后，先调度已完成正式研究任务的一致性复跑，再追加训练 seed。生产尚未完成、仅 pilot 通过的任务不进入重复训练。队列最多存活 1,000 小时（包含等待与全部任务执行），不是实际消耗承诺；最终 500 局测试不会自动打开。

现用用户级 transient systemd 服务：

- `robosyn-general-20260905.service`：生产研究。
- `robosyn-assets-20260905.service`：主资产下载。
- `robosyn-assembly-assets-20260905.service`：装配优先下载，单独状态，固定官方 revision。
- `robosyn-validation-20260905.service`：后续一致性/重复训练队列。

这些服务不依赖当前交互会话保持连接；未配置机器重启后的自动启动。服务重启策略不等于允许重跑已执行的失败评测局。

## 仍未交付

十任务全部性能改善、多训练 seed 最终 500 局与层级统计、代表任务数据/处理归因、完整任务语义视频审核、DP 训练适配、第二 benchmark 迁移验证和官方 SOTA 均未宣告完成。
