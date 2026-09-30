"""Render a CPU-only, illustrative README animation; never consumes run data.

Requires Pillow, a CJK font and FFmpeg with libx264. No benchmark footage,
model API, GPU renderer or fabricated metric is used.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT, FPS, SECONDS = 1280, 720, 15, 24
BG = "#0e182b"
PANEL = "#18263d"
TEXT = "#ecf2ff"
MUTED = "#9baac4"
MINT = "#73e2c6"
BLUE = "#79c9ff"
PURPLE = "#b5a4ff"
AMBER = "#f3b86b"

SCENES = [
    ("从仓库出发", "代码隔离，资源显式连接。", MINT),
    ("让 Agent 做研究", "根据真实代码与证据，决定下一步。", BLUE),
    ("让证据说话", "训练完成，不等于评测了正确的策略。", PURPLE),
    ("让人看懂过程", "持续记录进展、对比、demo 与边界。", MINT),
]


class Renderer:
    def __init__(self, font_path: Path):
        self.fonts = {
            size: ImageFont.truetype(str(font_path), size)
            for size in (13, 14, 16, 18, 20, 22, 26, 34, 44, 48)
        }

    def render(self, time: float) -> Image.Image:
        scene = min(int(time // 6), 3)
        progress = (time % 6) / 6
        image = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(image)

        def text(x, y, value, size=20, color=TEXT):
            draw.text((x, y), value, font=self.fonts[size], fill=color)

        def box(x, y, w, h, fill=PANEL, outline=None, radius=16):
            draw.rounded_rectangle(
                (x, y, x + w, y + h), radius, fill=fill, outline=outline, width=2
            )

        def arrow(start, end, color=BLUE):
            draw.line((start, end), fill=color, width=3)
            angle = math.atan2(end[1] - start[1], end[0] - start[0])
            points = [end]
            for delta in (-0.55, 0.55):
                points.append((end[0] - 11 * math.cos(angle + delta), end[1] - 11 * math.sin(angle + delta)))
            draw.polygon(points, fill=color)

        for x in range(0, WIDTH, 40):
            draw.line((x, 0, x, HEIGHT), fill="#152139")
        for y in range(0, HEIGHT, 40):
            draw.line((0, y, WIDTH, y), fill="#152139")
        text(48, 27, "AutoSimSOTA", 22)
        text(241, 33, "AUTONOMOUS SIMULATION RESEARCH", 13, MUTED)
        box(874, 26, 358, 34, "#322e2a", radius=17)
        text(892, 33, "流程介绍动画 · 非仿真录像 · 无实测分数", 14, AMBER)
        draw.line((48, 83, 1232, 83), fill="#334565")
        title, subtitle, accent = SCENES[scene]
        text(48, 114, f"0{scene + 1}", 26, accent)
        text(106, 104, title, 48)
        text(108, 174, subtitle, 22, MUTED)

        if scene == 0:
            box(48, 266, 320, 243, outline="#334565")
            text(74, 290, "已有 benchmark 仓库", 22, MINT)
            text(74, 342, "源码 / 配置 / 原生入口", 20)
            text(74, 390, "数据 / 资产 / 权重", 20)
            text(74, 453, "不默认复制整个大仓库", 16, MUTED)
            box(464, 254, 346, 266, outline=MINT)
            text(490, 280, "独立运行环境", 26, MINT)
            text(490, 345, "源码副本：允许受控修改", 20)
            text(490, 394, "已有资源：显式只读挂载", 20)
            text(490, 461, "原仓库保持受保护", 18, MUTED)
            arrow((375, 374), (453, 374), MINT)
            box(907, 266, 325, 243)
            text(931, 290, "Agent 进一步核验", 22, BLUE)
            for index, label in enumerate(["实际读取路径", "环境 reset / step", "策略加载与原生 rollout"]):
                text(933, 351 + 46 * index, label, 18)
            arrow((821, 374), (896, 374))
            text(48, 553, "复制成功只是开始；资源连接和运行条件需要真实证据。", 20, MUTED)
        elif scene == 1:
            nodes = [(48, 263, "理解仓库", "读代码 / 查资源"),
                     (677, 263, "提出假设", "参考技能 / 设对照"),
                     (677, 439, "运行实验", "后台作业 / 原生评测"),
                     (48, 439, "分析与修复", "封存证据 / 复验原操作")]
            active = min(int(progress * 4), 3)
            for index, (x, y, label, note) in enumerate(nodes):
                box(x, y, 554, 136, outline=accent if index == active else "#334565")
                text(x + 25, y + 22, label, 26, accent if index == active else TEXT)
                text(x + 25, y + 75, note, 18, MUTED)
            arrow((613, 330), (666, 330))
            arrow((954, 404), (954, 432))
            arrow((667, 506), (613, 506))
            arrow((325, 432), (325, 405))
            text(48, 599, "主 Agent 掌握全局；技能提供方法，执行器保护边界。", 20, MUTED)
        elif scene == 2:
            labels = [("训练产物", "checkpoint / 配置"), ("实际加载", "权重与策略身份"),
                      ("原生 rollout", "真实环境与轨迹"), ("正式指标", "冻结评测协议")]
            for index, (label, note) in enumerate(labels):
                x = 48 + 307 * index
                box(x, 283, 263, 180, outline=PURPLE if progress * 4 >= index else "#334565")
                text(x + 20, 314, label, 26, PURPLE)
                text(x + 20, 387, note, 18, MUTED)
                if index < 3:
                    arrow((x + 269, 373), (x + 296, 373), PURPLE)
            box(48, 501, 1184, 94, "#27253d")
            text(73, 522, "筛选分数 ≠ 正式提升    安装成功 ≠ 仿真通过", 22, PURPLE)
            text(73, 559, "没有可信链路，就不声明改进；缺资源和零结果也如实记录。", 18, MUTED)
        else:
            box(48, 251, 742, 357, outline="#334565")
            text(74, 275, "RUN.md / RUN.html", 26, MINT)
            text(74, 326, "发生了什么 · 为什么 · 证据 · 接下来", 20)
            draw.line((74, 369, 761, 369), fill="#334565")
            text(74, 385, "实验", 18, MUTED)
            text(408, 385, "正式指标", 18, MUTED)
            text(614, 385, "身份核验", 18, MUTED)
            text(74, 435, "Baseline", 20)
            text(408, 435, "待评测", 20, AMBER)
            text(614, 435, "待核验", 20, AMBER)
            text(74, 484, "Candidate", 20)
            text(408, 484, "未确认", 20, AMBER)
            text(614, 484, "待核验", 20, AMBER)
            text(74, 557, "示意界面：不虚构分数，不冒充真实运行。", 18, MUTED)
            box(822, 251, 410, 159, "#211f3b")
            text(846, 275, "Demo 与数据可视化", 22, PURPLE)
            text(846, 327, "有真实产物时附视频与图表", 18)
            text(846, 369, "没有产物时说明缺口", 18, MUTED)
            box(822, 435, 410, 173)
            text(846, 458, "目标：可读、可追溯、可复验", 22, BLUE)
            text(846, 511, "不承诺任意仓库一键 SOTA", 18)
            text(846, 558, "让每一步研究都留下依据。", 18, MUTED)

        for index, (_, _, color) in enumerate(SCENES):
            x = 48 + 300 * index
            box(x, 660, 280, 5, "#334565", radius=2)
            fraction = 1.0 if index < scene else progress if index == scene else 0
            if fraction > 0:
                box(x, 660, max(3, 280 * fraction), 5, color, radius=2)
        text(48, 680, "READ THE CODE. RUN THE EXPERIMENT. KEEP THE EVIDENCE.", 13, MUTED)
        text(1081, 680, f"{min(int(time), 23):02d} / 24 sec", 13, MUTED)
        return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    parser.add_argument("--poster-only", action="store_true")
    args = parser.parse_args()
    if not args.font.is_file():
        parser.error("provide --font pointing to an existing CJK font")
    ffmpeg = shutil.which(args.ffmpeg)
    if not args.poster_only and ffmpeg is None:
        parser.error("FFmpeg not found; provide --ffmpeg")
    args.output.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(args.font)
    renderer.render(0).save(args.output / "intro-poster.png")
    if args.poster_only:
        return
    video = args.output / "autosimsota-intro.mp4"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2",
               "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(video)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for frame in range(FPS * SECONDS):
            process.stdin.write(renderer.render(frame / FPS).tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("FFmpeg video encoding failed")
    except BaseException:
        process.kill()
        process.wait()
        raise
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                    "-filter_complex_threads", "2", "-filter_complex",
                    "fps=5,scale=800:-1:flags=lanczos,split[a][b];"
                    "[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                    "-loop", "0", str(args.output / "intro.gif")], check=True)
    print(f"Created {SECONDS}s illustrative intro in {args.output}")


if __name__ == "__main__":
    main()
