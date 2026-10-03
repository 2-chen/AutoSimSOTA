"""CPU-rendered editorial motion film; concept artwork, never benchmark footage.

Pillow draws authored typography/geometry. FFmpeg animates the persisted AI cover
and composites the motion layers; no GPU, live run data or fabricated metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS, SECONDS = 1280, 720, 60, 24
PAPER, INK, RED, MUTED = "#f3eee5", "#252520", "#cc452e", "#817d72"


class Renderer:
    def __init__(self, font: Path):
        self.font = font
        self.latin = Path("/usr/share/fonts/truetype/lato/Lato-Heavy.ttf")
        self.fonts = {}

    def face(self, size, latin=False):
        key = size, latin
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(str(self.latin if latin and self.latin.is_file() else self.font), size)
        return self.fonts[key]

    def render(self, seconds, *, transition=True):
        chapter = min(int(seconds / 6), 3)
        phase = seconds % 6
        ease = 1 - (1 - min(phase / .85, 1)) ** 3
        image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        dark = chapter == 1
        fg = PAPER if dark else INK
        secondary = "#b8b0a2" if dark else MUTED
        if chapter in (1, 2):
            draw.rectangle((0, 0, WIDTH, HEIGHT), fill=INK if dark else PAPER)
        else:
            draw.rectangle((0, 0, WIDTH, 76), fill=(243, 238, 229, 235))

        def text(x, y, value, size=24, color=None, latin=False):
            draw.text((x, y), value, font=self.face(size, latin), fill=color or fg)

        def line(x1, y1, x2, y2, color=RED, width=2):
            draw.line((x1, y1, x2, y2), fill=color, width=width)

        shift = int(24 * (1 - ease))
        text(54, 31, "AutoSimSOTA", 20, latin=True)
        text(966, 36, "AUTONOMOUS SIMULATION", 14, secondary, True)
        line(54, 76, 1226, 76, secondary, 1)
        if chapter == 0:
            text(54, 142 + shift, "AUTONOMOUS / SIMULATION", 16, RED, True)
            text(49, 191 + shift, "AutoSim", 94, latin=True)
            text(49, 281 + shift, "SOTA.", 94, RED, True)
            text(57, 412 + shift, "从仓库出发。", 31)
            text(57, 459 + shift, "让实验不断向前。", 31)
            text(57, 524, "读代码 / 采数据 / 训策略 / 验证提升", 18, secondary)
        elif chapter == 1:
            text(54, 111, "01 / DATA FIRST, NOT GUESSWORK", 16, RED, True)
            text(49, 156 + shift, "Better data.", 78, latin=True)
            text(55, 264 + shift, "让 Agent 找到原生采集路径，针对失败提出数据策略。", 26)
            labels = [("READ", "理解仓库"), ("COLLECT", "采集数据"), ("TRAIN", "训练策略"), ("EVALUATE", "原生评测")]
            line(102, 416, 1174, 416, "#5d5b54", 1)
            active = min(phase / 5, 1)
            line(102, 416, 102 + 1072 * active, 416, RED, 3)
            for i, (label, cn) in enumerate(labels):
                x = 102 + 357 * i
                draw.ellipse((x - 7, 409, x + 7, 423), fill=RED if active >= i / 3 else secondary)
                text(x - 47, 459, label, 21, latin=True)
                text(x - 47, 498, cn, 21, secondary)
            text(54, 588, "技能提供方法。主 Agent 决定下一步。执行器保护原仓库。", 21, secondary)
        elif chapter == 2:
            text(54, 111, "02 / KEEP THE EVIDENCE", 16, RED, True)
            text(51, 159 + shift, "Trust the chain.", 76, latin=True)
            text(57, 260 + shift, "训练完成，不等于评测加载了正确的策略。", 26)
            stages = [("01", "训练产物", "checkpoint / config"), ("02", "实际加载", "policy identity"),
                      ("03", "原生 rollout", "environment / episodes"), ("04", "正式指标", "frozen protocol")]
            for i, (number, label, note) in enumerate(stages):
                x = 56 + 308 * i
                color = RED if phase >= 1 + i * .8 else secondary
                text(x, 357, number, 45, color, True)
                line(x, 419, x + 248, 419, color)
                text(x, 449, label, 27)
                text(x, 496, note, 16, secondary, True)
            text(56, 588, "连接训练产物、实际策略与原生评测，让每一次改进都有依据。", 22, secondary)
        else:
            # A paper reading surface gives text a quiet background over the cover.
            draw.rectangle((0, 77, 702, 637), fill=PAPER)
            text(54, 121, "03 / THE RESEARCH JOURNAL", 16, RED, True)
            text(49, 175 + shift, "Show the work.", 65, latin=True)
            text(56, 279 + shift, "别只看最终分数。", 32)
            text(56, 333 + shift, "看懂系统为什么继续、为什么修复。", 23)
            line(56, 400, 636, 400, secondary, 1)
            for i, value in enumerate(("RUN.md / 中文进度与实验对比", "真实 Demo / 视频、曲线与数据摘要", "证据链接 / 产物身份与原始回执")):
                text(56, 432 + 49 * i, value, 22)
            text(56, 596, "From repository to measurable progress.", 16, secondary, True)

        line(54, 647, 1226, 647, secondary, 1)
        line(54, 647, 54 + 1172 * seconds / SECONDS, 647, RED, 3)
        text(54, 669, "AGENT-LED / DATA-FIRST / EVIDENCE-BOUND", 14, secondary, True)
        text(1119, 669, f"0{chapter + 1} / 04", 14, secondary, True)
        # Crossfade adjacent designed scenes. Fading each scene to the artwork
        # used to flash the robot between dark/light chapters and feel like stalls.
        if transition and chapter > 0 and phase < .45:
            previous = self.render(chapter * 6 - 1 / FPS, transition=False)
            fraction = phase / .45
            fraction = fraction * fraction * (3 - 2 * fraction)
            image = Image.blend(previous, image, fraction)
        return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    parser.add_argument("--art", type=Path, default=Path("docs/assets/robotics-cover.png"))
    parser.add_argument("--poster-only", action="store_true")
    args = parser.parse_args()
    ffmpeg = shutil.which(args.ffmpeg)
    if not ffmpeg or not args.font.is_file() or not args.art.is_file():
        parser.error("FFmpeg, CJK font and persisted robotics-cover.png are required")
    args.output.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(args.font)
    # ffmpeg animates the generated artwork; Python only draws code-native layers.
    filters = ("[1:v]scale=2560:-1,"
               "zoompan=z='1.035+0.015*sin(on/240)':"
               "x='iw/2-iw/zoom/2+20*sin(on/180)':y='ih/2-ih/zoom/2':"
               "d=1:s=1280x720:fps=60,setsar=1[art];"
               "[art][0:v]overlay=0:0:shortest=1,format=yuv420p[out]")
    video = args.output / "autosimsota-intro.mp4"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "1280x720", "-r", str(FPS), "-i", "pipe:0",
               "-loop", "1", "-framerate", str(FPS), "-i", str(args.art),
               "-filter_complex_threads", "2", "-filter_complex", filters, "-map", "[out]"]
    if args.poster_only:
        command += ["-frames:v", "1", str(args.output / "intro-poster.png")]
    else:
        command += ["-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
                    "-profile:v", "baseline", "-level:v", "3.2", "-tune", "fastdecode",
                    "-crf", "22", "-maxrate", "2M", "-bufsize", "4M",
                    "-g", "60", "-keyint_min", "60", "-sc_threshold", "0", "-bf", "0",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        assert process.stdin
        for index in range(1 if args.poster_only else FPS * SECONDS):
            seconds = .8 if args.poster_only else index / FPS
            process.stdin.write(renderer.render(seconds).tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("FFmpeg rendering failed")
    except BaseException:
        process.kill()
        process.wait()
        raise
    if args.poster_only:
        return
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-threads", "2"]
    subprocess.run(base + ["-ss", "0.8", "-i", str(video), "-frames:v", "1", str(args.output / "intro-poster.png")], check=True)
    subprocess.run(base + ["-i", str(video), "-filter_complex_threads", "2", "-filter_complex",
        "fps=20,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=3", "-loop", "0", str(args.output / "intro.gif")], check=True)
    print(f"Rendered {SECONDS}s editorial concept film (CPU only)")


if __name__ == "__main__":
    main()
