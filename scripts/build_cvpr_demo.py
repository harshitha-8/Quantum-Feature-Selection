from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "cvpr_demo_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "cvpr_demo"
VIDEO_SIZE = (1920, 1080)
CARD_MARGIN = 72
CAPTION_HEIGHT = 228
TITLE_COLOR = "#f8fbff"
TEXT_COLOR = "#d6e5f5"
ACCENT_COLOR = "#72d0ff"
BG_TOP = (11, 22, 40)
BG_BOTTOM = (26, 59, 93)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a captioned CVPR-style demo storyboard and MP4 from project figures."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seconds-per-slide", type=float, default=4.5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if "slides" not in data or not isinstance(data["slides"], list):
        raise ValueError(f"Manifest {path} is missing a 'slides' list.")
    return data


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        font_path = Path(candidate)
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def resize_with_padding(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(target_width / image.width, target_height / image.height)
    resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (target_width, target_height), "#08111d")
    offset = ((target_width - resized.width) // 2, (target_height - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def make_gradient_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    bg = Image.new("RGB", size, BG_TOP)
    draw = ImageDraw.Draw(bg)
    for y in range(height):
        ratio = y / max(1, height - 1)
        color = tuple(
            int(BG_TOP[idx] * (1 - ratio) + BG_BOTTOM[idx] * ratio)
            for idx in range(3)
        )
        draw.line([(0, y), (width, y)], fill=color)
    return bg


def render_slide(slide: dict[str, Any], index: int, total: int, output_path: Path) -> None:
    width, height = VIDEO_SIZE
    background = make_gradient_background(VIDEO_SIZE)
    draw = ImageDraw.Draw(background)

    title_font = load_font(44, bold=True)
    body_font = load_font(29)
    meta_font = load_font(24)

    image_box_width = width - (CARD_MARGIN * 2)
    image_box_height = height - CAPTION_HEIGHT - (CARD_MARGIN * 2) - 24
    image_path = ROOT / slide["image"]
    if not image_path.exists():
        raise FileNotFoundError(f"Missing slide image: {image_path}")

    panel = resize_with_padding(Image.open(image_path), image_box_width, image_box_height)
    background.paste(panel, (CARD_MARGIN, CARD_MARGIN))

    caption_top = CARD_MARGIN + image_box_height + 24
    draw.rounded_rectangle(
        [(CARD_MARGIN, caption_top), (width - CARD_MARGIN, height - CARD_MARGIN)],
        radius=30,
        fill="#0d1827",
        outline="#1c3655",
        width=2,
    )

    draw.text((CARD_MARGIN + 28, caption_top + 28), slide["title"], font=title_font, fill=TITLE_COLOR)
    body_y = caption_top + 90
    for line in wrap_text(draw, slide["caption"], body_font, image_box_width - 56)[:4]:
        draw.text((CARD_MARGIN + 28, body_y), line, font=body_font, fill=TEXT_COLOR)
        body_y += 38

    footer = f"{index}/{total}  |  {slide.get('source_label', 'Project figure')}"
    draw.text((CARD_MARGIN + 28, height - CARD_MARGIN - 42), footer, font=meta_font, fill=ACCENT_COLOR)
    background.save(output_path, quality=95)


def html_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_html(manifest: dict[str, Any], output_path: Path) -> None:
    slides_html = []
    for idx, slide in enumerate(manifest["slides"], start=1):
        slides_html.append(
            f"""
            <section class="slide">
              <div class="meta">{idx} / {len(manifest["slides"])} · {html_escape(slide.get("source_label", "Project figure"))}</div>
              <h2>{html_escape(slide["title"])}</h2>
              <div class="image-wrap">
                <img src="../{html_escape(slide["image"])}" alt="{html_escape(slide["title"])}">
              </div>
              <p>{html_escape(slide["caption"])}</p>
            </section>
            """
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_escape(manifest.get("title", "CVPR Demo"))}</title>
  <style>
    :root {{
      --bg: #08111d;
      --panel: rgba(13, 24, 39, 0.92);
      --line: rgba(114, 208, 255, 0.16);
      --text: #edf6ff;
      --muted: #9db7d4;
      --accent: #72d0ff;
    }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top left, rgba(114, 208, 255, 0.15), transparent 30%), linear-gradient(180deg, #0b1627 0%, #0e2540 100%);
    }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 40px 24px 72px; }}
    h1 {{ font-size: 2.6rem; margin-bottom: 10px; }}
    .intro {{ color: var(--muted); line-height: 1.7; max-width: 900px; margin-bottom: 30px; }}
    .slide {{ background: var(--panel); border: 1px solid var(--line); border-radius: 28px; padding: 24px; margin-bottom: 28px; box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22); }}
    .meta {{ color: var(--accent); font-size: 0.95rem; margin-bottom: 10px; }}
    .image-wrap {{ background: #050b14; border-radius: 22px; padding: 14px; margin: 18px 0; }}
    img {{ display: block; width: 100%; border-radius: 14px; }}
    p {{ line-height: 1.75; color: #d9e8f8; font-size: 1.06rem; }}
  </style>
</head>
<body>
  <main>
    <h1>{html_escape(manifest.get("title", "CVPR Demo"))}</h1>
    <p class="intro">{html_escape(manifest.get("description", ""))}</p>
    {''.join(slides_html)}
  </main>
</body>
</html>
"""
    output_path.write_text(html)


def build_markdown(manifest: dict[str, Any], output_path: Path) -> None:
    lines = [f"# {manifest.get('title', 'CVPR Demo')}", "", manifest.get("description", ""), ""]
    for idx, slide in enumerate(manifest["slides"], start=1):
        lines.extend(
            [
                f"## Slide {idx}: {slide['title']}",
                "",
                f"![{slide['title']}](../{slide['image']})",
                "",
                slide["caption"],
                "",
            ]
        )
    output_path.write_text("\n".join(lines))


def build_concat_file(frame_paths: list[Path], seconds_per_slide: float, output_path: Path) -> None:
    lines: list[str] = []
    for frame in frame_paths:
        lines.append(f"file '{frame.resolve()}'")
        lines.append(f"duration {seconds_per_slide:.3f}")
    if frame_paths:
        lines.append(f"file '{frame_paths[-1].resolve()}'")
    output_path.write_text("\n".join(lines) + "\n")


def build_video(frame_paths: list[Path], seconds_per_slide: float, fps: int, output_path: Path) -> None:
    concat_path = output_path.parent / "slides.txt"
    build_concat_file(frame_paths, seconds_per_slide, concat_path)
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_path),
            "-vf", f"fps={fps},format=yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    output_dir = args.output_dir
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    total = len(manifest["slides"])
    for idx, slide in enumerate(manifest["slides"], start=1):
        frame_path = frames_dir / f"slide_{idx:02d}.png"
        render_slide(slide, idx, total, frame_path)
        frame_paths.append(frame_path)

    build_html(manifest, output_dir / "index.html")
    build_markdown(manifest, output_dir / "storyboard.md")

    summary = {
        "title": manifest.get("title", "CVPR Demo"),
        "slide_count": total,
        "seconds_per_slide": args.seconds_per_slide,
        "estimated_video_seconds": round(total * args.seconds_per_slide, 2),
        "outputs": {
            "html": str((output_dir / "index.html").resolve()),
            "markdown": str((output_dir / "storyboard.md").resolve()),
            "frames_dir": str(frames_dir.resolve()),
        },
    }
    if not args.skip_video:
        video_path = output_dir / "cvpr_demo.mp4"
        build_video(frame_paths, args.seconds_per_slide, args.fps, video_path)
        summary["outputs"]["video"] = str(video_path.resolve())

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
