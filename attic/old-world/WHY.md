# 旧世界(2026-09-23 移出)

从系统里移出 **191 个文件 / 42,012 行**。包从 58,155 行降到 15,785 行。

判据一条:**为某一个 benchmark 而设的文件、函数或流程不留在系统里。**
覆盖一大类的可以留;泛化由 LLM 完成,不由按名字写死的分支完成。

## 移出了什么

| 是什么 | 行数 | 为什么走 |
|---|---|---|
| `research/repository_autoresearch.py` + 9 个 `robotwin_*` | ~4,000 | 第二套研究循环,而且 `autosim research` 跑的是它。CLI 的命令行里带着一个 benchmark 的任务名、epoch 数、适配器 |
| 根目录 9 个 `robosyn_*.py` | ~3,700 | 一个 benchmark 的采集、混合、选择、对比、恢复 |
| 旧任务 CLI 那一套(`adapters/`、`scene/`、`mock/`、`optimize/`、`onboard/`、`export/`、`ideas/`、`mcp_client`) | ~6,800 | Isaac Sim 和 ACT 各一个适配器,场景生成器,闭合回路优化器 —— 换一个 benchmark 都不能用 |
| `research/registry.py` | ~180 | 一张写死的任务表:`click_bell` → 关节/物体/族,十个任务名 |
| `experiment_validation/`、`experiment_system/`、`research_validation/` | ~7,000 | 一个 benchmark 的评测与实验编排 |
| `research/` 里只被旧循环可达的 ~50 个模块 | ~8,000 | 设备探针、采集 worker、score_push 系列、恢复 agent 等 |

## 移出前确认过的事

**可达性分析,不是估计。** 从通用入口(`research`/`scout`)出发做静态可达性,
不可达的才动。分析脚本的两次修正值得记下来:

1. 第一版解析相对导入时把包根算错了,把 99 个模块误报成不可达。
2. 第二版漏了 `from . import a, b` 这种形式 —— 模块名在 `names` 里,不在 `module` 里。

**静态分析漏了一个子进程依赖。** `contract_codegen` 是通过路径去跑
`contract_runner.py` 的,不是 import。它被移走之后 `test_contract_codegen` 才报出来,
已经搬回来了。教训:可达性分析覆盖不了按路径调用的东西。

**移出前没有任何测试引用过那一整套。** `adapters/`、`scene/`、`optimize/` 等 6,817 行
一个测试都没有。

## 怎么取回来

    cd <repo>
    cp -r attic/old-world/autosim/* autosim/
    cp -r attic/old-world/tools/* tools/

`cli.py` 里那六条命令、以及 `research/repository_autoresearch` 的分发,已经删掉了,
需要从 git 历史取回。确认不需要之后,整个 `attic/` 可以直接删。

## 一件事没做

包代码里 benchmark 名字 **0 处**;但**文档字符串里还有 39 处**。那些是解释一条通用规则
为什么存在的历史(「某次碰撞让一个阶段烧掉了整轮预算」)。删掉它们会让规则失去依据,
重写 39 段是另一件事,收益也小。要清的话按同一判据来:**名字出现在分支、默认值或
路径里**才必须走。
