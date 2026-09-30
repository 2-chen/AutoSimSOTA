# 旧任务 CLI 那一套(2026-09-23)

移出 16 个文件,6817 行。

## 为什么

`autosim` 的 CLI 曾经有六条命令:`init`、`main`(裸 `autosim --repo ... --task ...`)、
`sessions`、`inspect`、`demo`、`scene`。它们通向一个和"研究循环"平行的世界:
Isaac Sim 的 MCP 客户端、ACT 适配器、场景生成器、mock 模拟器、闭合回路优化器。

那是**为具体的模拟器和具体的策略族写的**:`act_adapter.py`(758 行)是关于 ACT 这一个
策略族的,`isaac_adapter.py`(882 行)和整个 `scene/`(1602 行)是关于 Isaac Sim 这一个
模拟器的。换一个 benchmark,这些都不能用。

## 移出去之前,谁引用它们

只有 `autosim/cli.py` 里那六条命令,以及 `tools/` 下的几个脚本。没有任何 research
路径上的模块引用它们 —— 这一点是用可达性分析确认的,不是估计的。

## 怎么取回来

    mv attic/legacy-task-cli/autosim/autosim/* autosim/autosim/

命令行那部分(`cmd_run`/`cmd_system`/`cmd_main`/`cmd_demo`/`cmd_scene`)已经从
`cli.py` 删掉了,可以从 `<commit>` 之前的版本取回。
