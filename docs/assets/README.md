# 首页展示素材

本目录素材用于解释项目，不是实验成果。

| 文件 | 用途 |
| :--- | :--- |
| `hero.png` / `hero.svg` | 项目横幅：PNG 展示，SVG 编辑源 |
| `architecture.png` / `architecture.svg` | 研究职责与证据流概念图 |
| `run-preview.png` / `run-preview.svg` | 人类可读研究记录的界面示意，非真实运行截图 |
| `intro.gif` | GitHub README 可直接展示的循环预览 |
| `autosimsota-intro.mp4` | 24 秒无声流程介绍视频，非仿真录像 |

动画全部由代码绘制，不使用模型生成的仿真画面，不含实测分数、真实轨迹、密钥或 run 内数据。首页使用 PNG 配图与 GIF 动画，避免依赖 SVG 文件预览器；图像采用绝对原始文件 URL，兼容不解析仓库相对路径的 Markdown 阅读器。MP4 链接仅提供下载后播放，不承诺 GitHub 文件页在线播放。

## 重新生成 PNG 配图

系统 Python 需要 PyGObject 与 Rsvg 2.0（Linux 常见包为 `python3-gi`、`gir1.2-rsvg-2.0`），并准备中文字体。示例使用系统 Python，不改动研究任务环境：

```bash
/usr/bin/python3 tools/render_readme_images.py
```

脚本将本目录三个 SVG 编辑源重新渲染为同名 PNG；仅使用 CPU。

## 重新生成动画

在独立的素材工具环境安装 Pillow，准备 FFmpeg（含 libx264 编码器）和中文字体；不要为生成素材更改正在运行研究任务的环境。脚本仅使用 CPU，默认编码线程为 2，不调用模型 API 或 GPU。

```bash
python tools/render_intro.py --font /absolute/NotoSansCJK-Regular.ttc \
  --ffmpeg /absolute/ffmpeg --output docs/assets
```

输出 MP4、GIF 和用于检查的首帧 `intro-poster.png`。可以仅生成一帧来检查中文与排版：

```bash
python tools/render_intro.py --font /absolute/NotoSansCJK-Regular.ttc \
  --output /tmp/autosim-intro-preview --poster-only
```

静态 SVG 无脚本、无外部素材依赖，可直接编辑与审阅。
