# Editorial Robotics / 展示素材

暖纸白、炭黑与朱红。用机械臂雕塑感、编辑式大字和留白替代默认蓝色 dashboard。
所有素材用于项目介绍，不是实验成果；不含真实轨迹、实测分数、密钥或 run 数据。

| 文件 | 来源与用途 |
| :--- | :--- |
| `robotics-cover.png` | 内置 imagegen 生成的机械臂概念插画，原始输出保留，不是仿真截图 |
| `hero.svg` / `hero.png` | 插画与代码排版组成的项目封面 |
| `architecture.svg` / `architecture.png` | 原生可编辑概念图：主 Agent、资源、技能、实验与证据 |
| `run-preview.svg` / `run-preview.png` | 中文研究笔记设计示意，指标均标待评测，非运行截图 |
| `intro-poster.png` / `intro.gif` | 视频静态封面 / README 可播放预览 |
| `autosimsota-intro.mp4` | 24 秒、1280×720、24fps、无声 H.264 项目短片 |
| `IMAGE_PROMPT.md` | 生图最终提示词、来源和视觉边界 |

## 图片重渲染

```bash
/usr/bin/python3 tools/render_readme_images.py
```

需要 PyGObject、Rsvg 2.0、Lato 与 Noto Sans CJK 字体。`hero.svg` 相对引用原始
`robotics-cover.png`，须保留在同一目录；架构图与预览不依赖生成模型。
首页使用最终 PNG 而不是 SVG，避免不同 Markdown 阅读器的 SVG 限制。

## 视频重渲染

```bash
python tools/render_intro.py --output docs/assets
```

需要 Pillow、FFmpeg（libx264）、中文字体及保存在仓库中的概念插画。
Python 绘制独立字体/几何动画层；FFmpeg 为插画添加平缓镜头移动并合成。
仅用 CPU，两条编码线程，不修改研究环境，不使用实验 GPU，不重新调用生图模型。
可用 `--font`、`--ffmpeg`、`--art` 指定资源。`--poster-only` 仅生成预览封面。

四章依次介绍仓库驱动、数据优先、产物身份与评测证据链、中文研究记录。
流程线的动画是概念说明，不是训练进度、性能曲线或机械臂运动仿真。

README 使用原始文件 URL；GIF 可内嵌，MP4 提供下载播放链接，不依赖 GitHub 文件页视频预览。
所有生成素材都保存在项目目录，不能只依赖模型工具的个人缓存路径。
