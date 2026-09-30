# 过程可视化与阶段总结：改造规划

给系统加一层"人能看懂它做了什么、以及为什么"的东西。规划分现状、技术依据、设计、实施顺序、以及边界。

---

## 一、现状：原材料已经很丰富，缺的是装配

一次运行留下的记录比想象的多。按可用的成熟度排：

| 已有 | 位置 | 说明 |
|---|---|---|
| **"为什么"的文本** | `proposal.hypothesis`(上千字)、`expected_validation`、`resolution.note`、`declaration.evidence`、`strategy.reasoning`、`execution.json` 的 `parameters[*].evidence`、`stages[*].level`、`events.json` 的 `device_why`、`observed[*].detail` | 这些不是日志，是**论证**，逐条带文件路径 |
| 阶段/轮次/结果四层 | `events.json`、`research_report.json`、`measurements/*.json`（派生管线）；`rounds/round_N/{proposal,round_result}.json`、`selection.json`、`run_state.json`（全流程） | 决策与结果都在，只是分散在多个文件 |
| **逐步骤轨迹** | `telemetry.jsonl` —— 实测 1047 行，每行 `{step, seed, entities{物体:{pose:4x4}}}` | 一次 40 局评测的每步刚体位姿。**这是唯一能离屏重建场景的源** |
| 资源视图 | `budget.json`、`schedule.json`（含 `next.waiting[].reason`）、`telemetry.json` | "为什么这个作业还没开始"就在 `waiting[].reason/detail` |
| 视频 | `autoresearch_runs` 下**1155 个 mp4**；RoboTwin 原生评测按 `task_config` 录像；LeRobot 数据集自带 | 已有，但没有被任何东西索引或展示 |
| 报告生成范式 | `milestone_dashboard.py`、`robosyn_milestone.py`、`research_comparison.py`、`checkpoint_audit.py` 等 8 处 | 都是「JSON + `render_markdown()` + `--report` 落 .md」的同一模式，**可直接照搬** |
| 唯一的人读入口 | `tools/observe.py` | 读 `running.json`/`measurements/`/`events.json`，输出"哪里对不上"。纯文本 |

**四个真正的缺口**（都是缺写入点或缺装配，不是缺数据）：

1. **派生管线丢掉了提案原文。** `derived_research.py:482` 写的是 `content, _ = client.chat_with_metadata(...)` —— 下划线丢掉了 provider 元数据，而 `content` 只用过一次（解析成 `hypothesis`），**原始响应没有落盘**。所以这条管线上"控制器看到了什么、说了什么"无法回放。
2. **全流程管线只留哈希。** `rounds/round_N/proposal.json` 里有 `response_sha256`，**没有响应正文和 prompt 正文**。全仓库只有 scout 的 `draft_*.json` 是完整的（含被拒草稿，那正是最有信息量的东西）。
3. **没有跨运行的索引。** `autoresearch_runs/<benchmark>/<run_id>` 只能靠列目录枚举。没有 `runs.json`。
4. **研究管线硬编码不录像。** `evaluation.py:335`、`safe_evaluation.py:154`、`safe_evaluation_v2.py:38` 都是 `eval_video_log=False`。所以自己跑的评测没有 demo，只有 RoboTwin 的原生评测和数据集里现成的。

**明确不要重建的**：不要做 dashboard 服务。这套系统一贯的纪律是"产物就是记录"——起一个服务等于造第二个真相源，而第一个已经足够。

---

## 二、技术依据

**① 记录的标准形状：[W3C PROV](https://www.semanticscholar.org/reader/a847ab0537e30010588fd086e9d56c5fc99819d9) 映射到 agent 执行**

| PROV | 对应 |
|---|---|
| Entity | 证据产物 —— 文档、工具输出、**主张** |
| Activity | 执行单元 —— 推理步骤、工具调用 |
| Agent | 模型、工具、**人** |
| 关系 | `used` / `wasGeneratedBy` / `wasDerivedFrom` / `wasInformedBy`（步骤间） |

一个实现细节值得抄：**把扁平日志变成溯源图，需要把一条记录拆成"调用、参数、结果"三部分**。我们的 `measurements/*.json` 恰恰是扁平的。

**② 最该警惕的先例：[ADR-015](https://raw.githubusercontent.com/fworks-tech/agenthood/477980dd62844fb3570adfc12312b5923d3c09ee/docs/adr/ADR-015-decision-intelligence-and-provenance.md)**

> ADR 和 git 历史记录了**人**的决策，但 **agent 运行期间的决策是短暂的** —— `DecisionLog` 这个 schema 存在**却从来没被写入过**，tracer 是个 no-op，于是"**系统为什么做了 X**"是无法回答的。

**这套系统已经有同一个形状。** 这一轮里我自己撞到同一件事两次：能力写好没接线、`recipe.json` 写 1 处读 0 处。**决策的推理写进了模型响应，而记录只留下结果。** 所以规划的第一层不是可视化，是**把决策连结果一起写下来**。

**③ 给人读的那一层**：[Springer 那篇](https://rd.springer.com/content/pdf/10.1007%2F978-3-319-16462-5_13.pdf?pdf=core)从 W3C PROV **自动生成 notebook**（结果表、图、溯源可视化、指向代码和数据的链接），例子就是 ML 实验。它的立场：**溯源应当是可读文档的主入口，不只是审计功能。** 相邻做法有 Burrito、Research Objects。另有用户研究显示，**用溯源数据的人理解系统行为显著更好**。

**④ 证据要分层**（"breakable receipts" 那份）：**哈希能验证的、能重放的、仍然需要人来看的** —— 而且有些层无论密码学验证与否都需要人工复核。这给了"佐证材料"一个诚实的框架，**分不清哪层是哪层，才是"给了证据"和"看起来像给了证据"的区别**。

**⑤ 遥测标准**：[OTel GenAI semconv](https://uptrace.dev/blog/opentelemetry-ai-systems) 的 span 树是 `root → invoke_agent → chat/execute_tool`，最小可用信号是**时长、token、错误态**。值得借鉴的是**结构**，不是传输——我们不需要 collector，需要的是"每个执行单元有父、有时长、有具名属性"。同一份材料提醒：AI 负载产生的遥测是传统服务的 10–50 倍，**别把一切都记下来**。

---

## 三、设计

五层，从"记录"到"呈现"。**关键原则：算出来的部分和写出来的部分分开并且标明**——时间线、数字、产物路径由记录生成（不可能错），总结由模型写（带引用，并标为"写出来的"）。

### 第 1 层：决策记录（缺的结构）

每次运行一个 `decisions.jsonl`，追加式：

```json
{"at": "...", "activity": "vary train.loss_scale=2.0",
 "by": "model", "agent": "deepseek", "used": ["evidence/round_1.json"],
 "why": "控制器原话，或 tool 的判定理由",
 "produced": ["rounds/round_1/proposal.json"],
 "outcome": null}
```

三个要点：

- **`outcome` 事后回填。** 一条决策没有结果就是日记，不是记录——这正是 ADR-015 那个没被写入的 `DecisionLog`。回填点在每轮结束时。
- **`by` 区分模型 / 工具 / 人。** 人给的 seed、人手工改的路径、人回的那句话，都要能看出来。Cognizant 那份的"agent 写的三元组要可查询地区分于人类断言"是同一个要求。
- **补上四个缺口的三处**：派生管线把 `content` 与 provider 元数据落盘；全流程管线把响应正文与 prompt 正文落盘（不只 sha256）。

### 第 2 层：叙事文档（主交付物）

每次运行一份 `RUN.md`，**生成而非手写**，可重复生成：

```
# <benchmark> / <run_id>
## 摘要              ← 模型写，带引用，标 WRITTEN
## 时间线            ← 生成，来自 events.json + measurements/ + process.json
## 决策与结果        ← 生成，来自 decisions.jsonl
## 数字              ← 生成，来自 measurements/*.json 的 readings
## 产物与证据        ← 生成，每个文件 + 它证明了什么
## 未解决            ← 生成，来自 observations.json + 未被诊断的失败
## 复现方法          ← 生成，来自 derived_stages.json + recipe.json + budget.json
```

**每一节标明它是算出来的还是写出来的。** 这一条是整个设计里最重要的：一个模型写的总结如果和算出来的时间线长得一样，读者就分不清哪句可以查证。

沿用现成的 `render_markdown()` + `--report` 范式（8 处已有）。

### 第 3 层：证据与 demo

- **证据索引**：每个产物一行，写清"它证明了什么"和"它不能证明什么"。后者是"breakable receipts"那份的要点。
- **demo**：把 `eval_video_log` 变成**提案可以变的轴**（现在硬编码 False），并把已有 mp4 收进索引。RoboTwin 原生评测的录像已经在，只是没人展示。
- **轨迹动画**：`telemetry.jsonl` 是唯一能离屏重建场景的源。可选，但它是"具身仿真"这一类最贴题的展示形式——**一行断言比不上一段回放**。

### 第 4 层：可视化

- **Mermaid 画在 MD 里**：零依赖，GitHub 直接渲染。溯源图、决策链、时间线（Gantt）都用它。
- **时间线**：阶段与作业的甘特图，来自 `events.json` 的时间戳与 `process.json` 的 `elapsed_seconds`。
- **自包含 HTML**：单文件、无 JS 依赖、可邮件发送。不是服务。`metrics.py:plot_curve()` 是死代码，可以复活成图表生成。

### 第 5 层：跨运行索引（现在完全没有）

`autoresearch_runs/runs.json`：每次运行一行，含 benchmark、时间、状态、关键数字、`RUN.md` 路径、证据目录。**这是"人能了解系统整个运行过程"的前提**——现在只能列目录。

---

## 四、实施顺序

按"解锁下游"排序，每一步独立可用：

| # | 做什么 | 为什么先做 | 成本 | 状态 |
|---|---|---|---|---|
| 1 | **决策记录 + `outcome` 回填**；补三处丢失的写入 | 不做这个，后面所有层都只能展示结果、展示不了推理 | 小 | **已完成** |
| 2 | **`RUN.md` 生成器**（算/写分离 + 引用） | 主交付物；范式已有 | 中 | **已完成** |
| 3 | **证据索引 + `eval_video_log` 成为提案轴 + 收集已有 mp4** | 用户明确要的 demo 部分 | 小 | **已完成** |
| 4 | **Mermaid 溯源图 + 甘特时间线**（画进 `RUN.md`） | 零依赖，边际成本极低 | 小 | **已完成** |
| 5 | **自包含 HTML 视图** | 便于分享 | 中 | **已完成** |
| 6 | **`runs.json` 跨运行索引** | 让"整个运行过程"可枚举；也解锁跨运行对比 | 小 | **已完成** |
| 7 | 轨迹动画（可选） | 最贴题的 demo，但成本最高 | 大 | **已完成**（见下面的说明：做成了轨迹图，不是动画） |

### 已完成部分落在哪里

全部七步都做了。逐层：

- **第 1 层**：`autosim/research/decisions.py`。写入点在 `derived_research.py`（每轮一条 `by="model"` 的提案决定，跑完回填 outcome）与 `compute_decision` 的设备决定（`by="tool"`）。`exchanges/round_N.json` 保存控制器被喂的和答的原文（含 provider 元数据）；`repository_autoresearch.py` 的 `proposal.json` 现在也保存 prompt 与响应正文，不只 sha256。测试 `tests/test_decisions.py`。
- **第 2 层**：`autosim/research/run_record.py`，人读入口 `tools/record.py`。装配两种运行形状（派生管线的 `events.json`/`measurements/`，全流程管线的 `rounds/`+`process.json`）。摘要由模型写，其数字逐个在记录里查出处，查不到的列出来。生成点：`derived_research.run_stage`（每阶段，不调模型）与 `run()`（每轮，调模型）；`repository_autoresearch` 在每轮结束、正常结束、被中断、以及**异常处理器里**各生成一次。测试 `tests/test_run_record.py`。
- **第 3 层**：`evaluation.py` 新增 `--video-log`（默认关）；`runtime.evaluate(video=)` 一路带到非分片与分片两条路径，并把 `video_log` 记进 `evaluation_request.json`（记「被要求了什么」，与被各分片比对的 `protocol.json` 分开）；RoboSyn 声明新增 `resolution.video_log`（**optional**，因为在这条轴出现之前写的提案意思是「不录」，把那些判为不合法是读错了它们）；`repository_autoresearch.evaluate(video=)` 在缓存命中时**拒绝**而不是静默返回（复用键不含录像这一维）。`RUN.md` 新增「录像与 demo」一节，按证明力分类列出全部录像。
- **第 4 层**：`autosim/research/mermaid.py`（纯函数，两个渲染器）；甘特图画在「时间线」下、溯源图画在「决策与结果」下。**用真解析器验证过**：把全部运行生成出来，22 张图逐张过 Mermaid 自己的 parser，再拿对抗性标签（`cuda:0`、`train.n_epochs=1`、带引号和方括号的散文、中文标点、`a/b\c<d>&e#f`）重测，全部通过。解析器收在 `tools/mermaid_parse.mjs`（需要一个 node 依赖树，所以不进测试运行）；测试断言的是从它身上读出来的不变式，`MERMAID_PARSER` 指过去时会跑真解析器。
- **第 5 层**：`autosim/research/report_page.py`，生成 `RUN.html`。零脚本、零样式表、零外链、零 URL，全部断言在 `tests/test_report_page.py` 里。**图不是 Mermaid**：Mermaid 是 JavaScript，而一个要靠脚本才能显示图表的页面在邮件客户端里什么也不显示，所以甘特、溯源、曲线全部改画成内联 SVG。测试把每个 SVG 当 XML 解析，标签里塞 `<`、`>`、`&`、引号 —— 浏览器拿到坏 SVG 是**静默不显示**，不是报错。曲线复活了 `metrics.py:plot_curve()` 那个写完从没被调用的东西，但没有复活它对 matplotlib 的依赖。
- **第 6 层**：`run_record.find_runs()` / `one_line_about()` / `runs_index()`，落盘 `autoresearch_runs/runs.json`。**这一层撞出了规划里没有的第三种运行形状**：`scouting/` 下的目录是 benchmark 上手运行（`autosim/research/scout.py`），记录是 `draft_*.json` / `verification.json` / `declaration.json`，**不在任何索引里** —— 这台机器上大半的工作对任何读运行的东西都是不可见的。`read_scout()` 把它们并了进来，索引因此从 42 次变成 63 次。索引里还留了 `directories_with_no_records`：17 个目录什么都没产出，**点名而不是跳过**，因为「试过但什么都没有」和「没试过」不是一回事。
- **第 7 层**：`autosim/research/trajectory.py`。97 个 `telemetry.jsonl`、3.5 万行位姿，此前一行都没被读过。做成了**俯视轨迹图**而不是动画，理由见下。

`RUN.md` 的最终节序是：摘要（写出来的）→ 时间线（含甘特图）→ 决策与结果（含溯源图）→ 数字（含曲线）→ 录像与 demo → 轨迹 → 产物与证据 → 未解决 → 复现方法。除第一节外全部标注 `算出来的`。

**代码分在六个模块里，一个模块一件事**（`run_record.py` 一度长到 1300 行装了四件事，拆开了）：

| 模块 | 一件事 |
|---|---|
| `run_record.py` | 把一次运行的记录装配成文档（读五种形状、八个 section、渲染） |
| `claims.py` | **写出来的那一半，以及对它数字的检查** —— 单独成模块，因为这是文档里唯一会错的部分，要能被单独审计 |
| `run_index.py` | 一次很多运行：`runs.json` |
| `report_page.py` | 同一个文档作为单文件 HTML，含内联 SVG 图表 |
| `trajectory.py` | 从位姿记录读出轨迹，含按输入的 stat 做键的缓存 |
| `mermaid.py` | 两张 Mermaid 图，纯函数 |

### 第 7 层：做成了轨迹图，而不是动画 —— 以及为什么

规划写的是「轨迹动画」。做出来的是**位姿在跨度最大的两个轴上的投影，每条路径按那一局的成败着色**。两条理由，第二条更重要：

1. 动画需要真实场景几何才能渲染，而系统里没有 —— 只有位姿。给位姿做动画就是给点做动画，成本换来的是一段会动的点。
2. **平面按数据选，不是按「哪个轴朝上」的约定选**：取所有局里跨度最大的两个轴。一个把世界摆得不一样的 benchmark，用约定的仰视图画出来就是一条直线。

**这条图最关键的一处纪律：着色必须被测量过。** 一张按成败着色的图会诱导读者看出差别；如果记录里没有差别，那着色就是伪装成发现的装饰。所以 `by_outcome()` 把差别算出来印在图的旁边。实测（`full_run` 的 selection 评测，100 局）：成功的 48 局里 bottle 走的路程中位数 **0.382**，失败的 52 局是 **0.198** —— **1.93 倍**；终点与杯子的距离中位数 0.221 对 0.306。**差别是真的，着色有依据**，而且这条差别只有位姿记录给得出来 —— 成功率那一栏里这 100 局只是两个数。

图上写着：**这是轨迹，不是回放。** 没有东西被渲染，没有相机被模拟；它显示东西在哪里，不显示看起来什么样，更不显示策略当时看到了什么。当成渲染图来读是这张图会主动诱导的错误，所以把话说在图旁边，而不只是写在代码注释里。

### 各层实施中撞到的具体问题

**第 2 层：检查器一开始什么都没查出来。** 剥离标识符用的 `\w*\d\w*` 把纯数字也一起剥了，于是一段数字全编的摘要在检查器眼里干净得很。剥离模式必须要求 token 里既有字母又有数字。**百分号的精度方向还是反的**：`116.7%` 字面量是 1 位小数、除以 100 后是 3 位，按字面量取精度，`1.2` 就成了它的出处。**数字还藏在字符串里**：空闲显存在 `device_why` 里是散文、损失在 `said` 里，只走 JSON 结构会把这些全判成查不到出处 —— 而一个总在虚报的检查是没人看的检查。

**第 3 层：三处硬编码，只有一处是真的。** 规划说 `eval_video_log=False` 硬编码在三处，这对；说三处都该变成轴，错。`safe_evaluation.py` 和 `safe_evaluation_v2.py` 的文档第一行自己写着 **"deliberately not a ranking evaluator"** 和 **"Single-episode, fixed-seed form of the non-ranking safe evaluator"** —— 它们是数值安全探针，单回合、固定种子，用途是把策略引发的非有限状态变成一次显式失败，而不是产生证据。**没人会去看一次 NaN 探测的录像。** 这两处保持硬编码。（这是同一个错误的小号版本：把一处常量出现三次，当成三处需要同一件事。）

**第 4 层：不臆造边和条，转义是替换不是删除。** 甘特图里没有可用时间戳的条目**不画** —— 一个条的位置就是它主张的东西，画错位置的条是一句看起来像测量结果的假话。溯源图只画记录里已有的 `used`/`produced`，一条没有依据的决定就是没有入边，那个空缺是发现而不是格式问题。转义同理：`cuda:0` 和 `cuda 0` 是两块不同的卡，把冒号删掉会让它们印成同一个词，那比两个都不印更糟。图的节点 id 取自内容的哈希而不是位置 —— 未变的运行两次读出来的图必须一样，否则它没法和自己比对，而那是图的主要用途。

**第 5、6、7 层：文档把自己列进了索引。** 产物索引扫运行目录，于是 `RUN.md` 出现在其中，而它的**大小随每次生成变化** —— 两次读同一次未变的运行会得到不同的文档，没有不动点。`RUN.html` 更糟：它由那次扫描生成，每重新生成一次就比上一次长一点。两个都排除了。同类的还有一处：产物索引里每个文件都显示同一个数字，因为那一列印的是**该类的平均值**而不是单个文件的大小 —— 一列写着「大小」的地方放平均值，就是一个不是任何文件大小的数。

**这三件事是同一个形状**，值得记下来：检查器全过、着色没有依据、列里放平均值 —— **一个总是通过的检查，和一个总是正确的检查，从输出上分不出来。**

**还有一处是性能上的同一个毛病。** 第一版只在**画图**上做了缓存，那一节仍然每次都把 14 个 telemetry 文件全解析一遍（5 万次 JSON 解析，占总耗时 2.5 秒里的 2.2 秒），而这份文档在每个阶段边界都会重新生成。修法是把两者统一到一个 sidecar（`trajectory.json`，键是输入的大小与修改时间），**冷 1.9 秒、热 0.11 秒**。教训不是「要缓存」，而是：**改了一半的优化常常看起来像没改** —— 我修了画图那条路径，测出来的数字降了一点，就以为修完了。

阶段总结的触发点：**第 2 层的生成接在 `awareness.observe()` 已经跑的地方** —— 每个阶段结束时那边已经在读记录了，让它顺手重新生成文档。不需要新的定时器。

---

## 五、边界与风险

- **模型写的总结会错。** 缓解不是"让它更准"，是**把算的部分和写的部分在文档里分开并标明**，且写的部分必须带可查的路径。这一层做不好，整个文档的可信度不如一个 `ls`。
- **别把一切都记下来。** OTel 那份提醒 AI 负载的遥测是传统服务的 10–50 倍。记录的单位应该是**决策**和**阶段**，不是每个 token。
- **凭证要擦。** 现有代码已有 `redact()` 与密钥擦除（`runtime.py` 的 `environment()`），新增的落盘点必须走同一条路——尤其是补写 prompt/响应正文之后。
- **不做的事**：不建 dashboard 服务、不引入 OTel collector、不引入数据库。已有的产物 + 生成的文档 + 单文件 HTML，够用，且不会造出第二个真相源。
- **一件要说清的**：第 1 层让运行变慢了吗——不会，多写几个文件而已，但它让"为什么"从不可回答变成可回答，而这是这一轮反复付代价的东西。
