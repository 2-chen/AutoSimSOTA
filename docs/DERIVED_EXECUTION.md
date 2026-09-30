# 无后端执行：让系统自己找出 benchmark 怎么跑

系统原来每接一个 benchmark 就要写一个执行后端文件：一个文件写明 RoboSyn 的训练器、评测器、采集器和转换器，第二个文件写明 RoboTwin 的。接 LIBERO 本来会是第三个。

现在这条路是派生的：**survey 报告仓库里有什么 → 模型选读哪些文件 → 模型给出每个阶段的入口点和调用方式 → 生成一个把系统输入映射到该 benchmark 命令行的函数 → 运行它，程序自己决定命令对不对**。

当前入口是 `.venv/bin/autosim research <repo> <output-dir> [rounds] [settings-json]`
（或 `python -m autosim.research.derive_and_run`）；研究循环在
`autosim/research/derived_research.py`，命令生成与纠错在
`autosim/research/execution_derive.py`。旧 `tools/research_derived.py` 已不存在。
`--keep-only` 只运行已保存的接入记录，`--rederive` 明确丢弃当前输出目录中的派生记录。

## 现在能做到什么（LIBERO，无手写后端）

- **识别出 LIBERO 只有 `train` 和 `evaluate`**，`collect` 不可用（唯一采集脚本要人操作机械臂），并据此得出研究循环可用阶段——这是读出来的，不是配置里写死的。
- **`evaluate` 能派生出一条跑得通的命令，且稳定复现**（最近四次运行四次通过）：从 usage line 读出真实 flag（`--benchmark/--task_id/--algo/--policy/--seed/--load_task/--device_id`），自行得出 `PYTHONPATH={repo}`，自行选定 `--task_id`。
- **`train` 能独立找到数据集**。LIBERO 的 loader 把数据路径拼进自己的 checkout（`libero/libero/../datasets`），那里从来没有数据集；数据在 checkout 的兄弟目录里。仓库里没有任何文字说明这件事。系统扫描 checkout 附近的目录，找到了 `/home/wbc/下载/autoresearch/test/datasets/libero`。
- **`train` 能独立发现两个 loader 都必须零 worker**：`h5py objects cannot be pickled` → 数据集持有文件句柄不能跨进程 → `train.num_workers=0` 和 `eval.num_workers=0`。
- **`train` 能独立发现首 import 会提问**，并把答案作为 `stdin` 声明出来。

`train` 还没能稳定通过验证，剩下的是一步之遥，而且是**同一步的两半**：正确的调用是 `folder=".../datasets/libero"` —— 键要叫 `folder`（不是 `data_root`、不是 `data.data_path`），值要加引号（hydra 的 override 文法不接受未加引号的非 ASCII 路径）。两个错误各自都犯过、也都各自被诊断对过，但没有一次在同一份草稿里同时做对。这不是结构问题，是收敛速度问题。

## 失败分三类，修法不同

这是这一轮最大的收获，写进了 `autosim/skills/making-a-stage-run/SKILL.md`。

| 类别 | 症状 | 该改什么 |
|---|---|---|
| 程序拒绝了参数 | `usage:` / `unrecognized arguments` | 重写命令。环境无关 |
| 程序没走到读参数 | `ModuleNotFoundError` / `cannot open shared object` / display 错误 | 改**调用方式**：工作目录、环境变量 |
| 程序跑了但缺东西 | 找不到数据文件、路径打不开 | 改**取值**。参数可能完全正确 |

把第三类当第二类修，或者把第二类当第一类修，是这一轮浪费尝试的主要方式。最糟的一种是告诉模型"参数被接受了，去改值"——而程序根本没走到读参数那一步。代码现在会先分类再生成反馈（`is_contract_error` / `is_environment_error`）。

## 系统自己的缺陷（都已修，都有测试）

这一轮暴露的问题里，**大部分不是 benchmark 的，是系统自己的**：

1. **提示词自相矛盾。** 一处说"参数必须从 `i` 里按名字读"，另一处说"参数是字面量写进 argv"。生成函数按前者写，调用方按后者给，于是 `KeyError: 'benchmark_name'`。
2. **嵌套 traceback 的原因在末尾。** hydra 会 re-raise，前 14 行只有库内部帧，真正的异常在三十行之后。而限长是从右边截断的——把"取头部"这个错误又犯了一遍。
3. **null 被当成肯定的回答。** `str(None).strip()` 是 `"None"`，为真。模型用 `null` 回答"这项没问题"，代码读成了"有障碍"。
4. **模型说"这是命令的问题不是环境的问题"时，结论被丢弃。** 那句话是关于命令的发现，而写命令的那部分还在循环里。现在它会作为指引回到下一轮生成。
5. **一次修订把 `invocation` 置成 `None`。** 提示词说"如果也不对就改"，所以不变时返回 `null`；合并进去之后下一轮从字符串 `"None"` 构造命令。
6. **四个独立的崩溃，每个都能终止整轮派生**：草稿丢了解释器（`PermissionError`）、诊断函数走到读不了的文件（`PermissionError` on `/etc/xrdp/key.pem`）、异常处理器引用了失败前不会绑定的变量（`UnboundLocalError`）、模型回了一段散文（不是 JSON）。现在每一轮都包了兜底：一轮抛异常就是一轮没产出，预算有限。

第 6 条值得单独说：**失败路径必须比成功路径更结实**，因为它运行的时候已经出事了。

## 两个设计边界

**输出目录：benchmark 可能根本没有这个参数。** LIBERO 的 `create_experiment_dir` 从配置值和工作目录算出 `./experiments/{benchmark}/{algo}/{policy}_seed{seed}/run_N`，而 `cfg.experiment_dir` 是这个函数**赋值**的，不是从配置读的——所以 `output_dir=`、`+output_dir=`、任何猜的名字都会被 hydra 正确拒绝。系统花了若干轮猜这个名字，才有人去读那个构造路径的函数。

由此得两条：读构造输出路径的函数，别猜它的 key；路径是相对的时候，**工作目录就是控制手段**——import 由搜索路径解决，所以阶段可以从一个调用方选的 scratch 目录跑，benchmark 自己的目录树就落在它下面。后者已经实现为 `artifact_beside`：声明的产物没出现在 output 下时，到命令运行的地方再找一次，并且**按修改时间界定**——checkout 里躺着之前每一次运行的产物，一个能被昨天的 checkpoint 满足的检查不是检查。

**调用方给的路径必须是 benchmark 能表达的。** 系统自己选输出目录，就得选一个 benchmark 的参数文法能接受的。checkout 在一个非 ASCII 名字的目录下（就是这台机器的情况），输出路径里的 `下载` 会让 hydra 的 override lexer 在等号处失败——而错误指向的是**系统自己的目录**。修订器把那个 traceback 读得完全正确，仍然无法动手，因为出问题的值不归 benchmark 管。现在阶段拿到的是纯 ASCII 路径；新运行的 ASCII 路径是指向 run 目录内真实输出的别名，旧运行的反向别名只读兼容。

## 显式执行图（2026-09-25 增量）

上述四阶段仍是兼容能力问题，不再是唯一可声明的阶段名。接入答案可以增加有源码证据的额外阶段，并提供 `execution_graph.nodes`、`depends_on`、`bindings`、`score_target`。图在执行前检查缺节点、循环依赖、非法绑定和未派生命令；执行中按节点保存 attempt 回执，上游 checkpoint 在传给下游评测前冻结。图的 evaluate-role 节点无论叫什么名字，都要经过同一协议冻结检查。

合成合同测试已跑通“先生成数据 → 更新策略 → 评测”及无 checkpoint 的源码控制器；后一种还在一轮源码补丁后重新测出更高的显式回报指标。**这仅证明新执行路径可达**：尚无真实仓库泛化、独立原生复评或图结构自动优化验收。

首次测量尝试会在 `research/<run-id>/comparison_protocol.json` 冻结评测节点定义、指标合同、任务合同，以及设置中的通用评测键（如 `task`、`seed`、`episodes`、`horizon`、`eval.*`）。仓库使用非标准名称时，应在 `research_goal.protocol_keys` 显式列出；训练参数如 `train.n_epochs` 不随之冻结。后续测量若改变冻结字段，会在启动训练前拒绝，回执核验也会复核设置哈希与该协议。**覆盖范围仍只是已声明和通用键**；命令内部隐式读取的配置、随机初始状态是否真正相同、独立复评与统计置信度还没有得到证明。

## 这一轮又修掉的三处（都是「验证通过 ≠ 跑得起来」）

1. **只留了 source，丢掉了 revision。** 派生循环在本地 `row` 上累积修订——工作目录、环境变量——但驱动只拿走了生成的函数和参数。于是**每个「验证通过」的阶段在真正运行时都没有那条让它跑起来的 `PYTHONPATH`**，在第一个 import 上就死：`ModuleNotFoundError: No module named 'libero'`，来自一条刚刚被证明可用的命令。这是「验证过了却跑不起来」的根因，`make_runnable` 现在连修订后的 stage row 一起返回。

2. **runner 把 `dataset` 猜成了 checkout 路径。** `_inputs` 里写的是 `str(self.repo)`。调用方的输入优先级高于派生确定的取值，所以系统**已经发现并记录下来的正确 `folder` 被一个没有数据的路径静默替换掉了**。命令看起来完全正确——键对、引号对——指向的目录是空的。runner 现在只提供它确实知道的东西，这里是什么都不知道。

3. **失败的测量被丢弃了。** `measure` 在 train 不通过时提前返回，结果不落盘，报告只说「没有跑到结束」。三次都得靠手工复现才知道程序说了什么。现在失败也写入 `measurements/`，`said` 进报告。

另外把 `device` 拆成了 `device`（框架设备字符串，`cuda`/`cpu`）和 `device_index`（序号）。LIBERO 做 `torch.device(cfg.device)`，传整数 0 是 `TypeError`——而 `--device_id 0` 要的正是整数。同一个词曾经同时表示这两件事。

## 通用性边界

- **源码级不兼容派生不出来。** LIBERO 有六处需要改源码才能跑（`np.bool`、`persistent_workers`、`torch.load` 的 `weights_only`、`cfg.folder` 默认值、hydra 路径要加引号、`libero.lifelong.main` 里那个 `NameError`）。派生能找出**怎么调用**，改不了 benchmark 自己的代码。这类不兼容系统目前只能报出来——`patches/libero.patch` 记的就是这些。
- **模型读同一个事实有方差。** `evaluate.py` 的 `--load_task` 有的轮次读对有的读错。轮次预算有限，读错的那次就把预算用完了。这不是机制问题，但它决定了成功的概率。
- 这一轮没有验证 `collect`——LIBERO 没有自动采集能力，系统如实报告了这一点，这正是设计要的行为。
