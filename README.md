# AutoSimSOTA

面向具身仿真 benchmark 的可审计自动研究系统。核心入口接收 benchmark 仓库路径，发现任务与能力，由研究控制器提出实验，再由本地执行器完成数据采集、质检、policy训练、原生评测、候选选择与导出。DeepSeek负责研究提案，不负责执行GPU训练。

## 仓库边界

- `autosim/`：Python包、CLI、适配器、研究执行器和测试；也保留早期实验模块。
- `patches/`：外部仓库的本地兼容修改与基准版本记录，不包含完整上游代码或资产。
- `docs/`：发布检查与当前可移植性限制。
- `.env.example`：无凭据配置模板。

第三方RoboSynChallenge、RoboTwin、EmbodiChain不是本项目核心代码。虚拟环境、模型、数据、缓存、运行报告和本地历史plan不进入Git；忽略规则不删除这些本地文件。实际维护的总计划目前位于本仓库外，不是运行依赖。

## 安装与CPU检查

从本仓库根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ./autosim pytest
.venv/bin/autosim research --help
.venv/bin/python -m pytest
```

这只安装核心依赖，不安装仿真器、CUDA、ACT训练依赖或资产。真实研究需要按对应benchmark配置独立环境及数据。项目根目录的`.env`用于本地API配置；不要把密钥放进提交、报告或命令示例。

```bash
# 仅发现能力，不启动API或GPU训练；路径由使用者提供
.venv/bin/autosim research /absolute/path/to/RoboSynChallenge --probe-only
```

完整流程见[研究入口说明](autosim/autosim/research/README.md)。其中历史机器路径和结果是开发记录，不是便携部署承诺；发布前请阅读[发布检查](docs/RELEASE_READINESS.md)。

## 当前验证范围

ClickBell已有真实API闭环及本地提升证据；Water/Handle只有小规模本地先导，RoboTwin已完成单轮启发式闭环但没有性能提升。尚未证明API决策优于等预算随机/固定对照，也没有自动多卡或任意benchmark零样本适配能力。结果不是官方隐藏赛道或SOTA认证。

## 许可与发布

尚未确定本仓库整体许可证，请维护者在公开发布前确认核心代码来源并选择许可证。第三方组件各自遵循原许可证；兼容补丁不改变其权利归属。不要因早期README的许可证徽章而推断本仓库整体已获MIT授权。
