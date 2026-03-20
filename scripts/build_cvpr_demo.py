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
VIDEO_SIZE = (2560, 1440)
CARD_MARGIN = 80
CAPTION_HEIGHT = 268
TITLE_COLOR = "#111111"
TEXT_COLOR = "#2c3640"
ACCENT_COLOR = "#5c6670"
BG_COLOR = "#ffffff"
LINE_COLOR = "#d2d8df"


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
        "/System/Library/Fonts/Supplemental/Tahoma Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Tahoma.ttf",
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


def make_background(size: tuple[int, int]) -> Image.Image:
    return Image.new("RGB", size, BG_COLOR)


def render_title_slide(slide: dict[str, Any], index: int, total: int, output_path: Path) -> None:
    width, height = VIDEO_SIZE
    background = make_background(VIDEO_SIZE)
    draw = ImageDraw.Draw(background)

    title_font = load_font(24, bold=False)
    subtitle_font = load_font(18, bold=False)
    meta_font = load_font(23)

    title = slide["title"]
    subtitle = slide.get("caption", "")
    title_lines = wrap_text(draw, title, title_font, 620)
    line_gap = 16
    title_heights: list[int] = []
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        title_heights.append(bbox[3] - bbox[1])
    total_title_height = sum(title_heights) + max(0, len(title_heights) - 1) * line_gap
    title_y = (height - total_title_height) / 2 - 36

    current_y = title_y
    for line, line_height in zip(title_lines, title_heights):
        title_box = draw.textbbox((0, 0), line, font=title_font)
        title_width = title_box[2] - title_box[0]
        title_x = (width - title_width) / 2
        draw.text((title_x, current_y), line, font=title_font, fill=TITLE_COLOR)
        current_y += line_height + line_gap

    if subtitle:
        subtitle_box = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        subtitle_width = subtitle_box[2] - subtitle_box[0]
        subtitle_x = (width - subtitle_width) / 2
        subtitle_y = current_y + 16
        draw.text((subtitle_x, subtitle_y), subtitle, font=subtitle_font, fill=TEXT_COLOR)

    footer = f"Slide {index}/{total}   |   {slide.get('source_label', 'Project overview')}"
    draw.text((CARD_MARGIN, height - CARD_MARGIN - 32), footer, font=meta_font, fill=ACCENT_COLOR)
    background.save(output_path, quality=95)


def render_slide(slide: dict[str, Any], index: int, total: int, output_path: Path) -> None:
    if slide.get("layout") == "title":
        render_title_slide(slide, index, total, output_path)
        return

    width, height = VIDEO_SIZE
    background = make_background(VIDEO_SIZE)
    draw = ImageDraw.Draw(background)

    title_font = load_font(48, bold=True)
    body_font = load_font(30)
    meta_font = load_font(23)

    image_box_width = width - (CARD_MARGIN * 2)
    image_box_height = height - CAPTION_HEIGHT - (CARD_MARGIN * 2) - 24
    image_path = ROOT / slide["image"]
    if not image_path.exists():
        raise FileNotFoundError(f"Missing slide image: {image_path}")

    panel = resize_with_padding(Image.open(image_path), image_box_width, image_box_height)
    background.paste(panel, (CARD_MARGIN, CARD_MARGIN))

    draw.rectangle(
        [(CARD_MARGIN - 1, CARD_MARGIN - 1), (width - CARD_MARGIN + 1, CARD_MARGIN + image_box_height + 1)],
        outline=LINE_COLOR,
        width=2,
    )

    caption_top = CARD_MARGIN + image_box_height + 34
    draw.line((CARD_MARGIN, caption_top - 16, width - CARD_MARGIN, caption_top - 16), fill=LINE_COLOR, width=2)
    draw.text((CARD_MARGIN, caption_top + 8), slide["title"], font=title_font, fill=TITLE_COLOR)
    body_y = caption_top + 86
    for line in wrap_text(draw, slide["caption"], body_font, image_box_width - 56)[:4]:
        draw.text((CARD_MARGIN, body_y), line, font=body_font, fill=TEXT_COLOR)
        body_y += 40

    footer = f"Slide {index}/{total}   |   {slide.get('source_label', 'Project figure')}"
    draw.text((CARD_MARGIN, height - CARD_MARGIN - 32), footer, font=meta_font, fill=ACCENT_COLOR)
    background.save(output_path, quality=95)


def html_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_html(manifest: dict[str, Any], output_path: Path) -> None:
    slides_html = []
    for idx, slide in enumerate(manifest["slides"], start=1):
        image_html = ""
        if "image" in slide:
            rel_image = Path(slide["image"]).relative_to("results")
            image_html = f"""
              <div class="image-wrap">
                <img src="../{html_escape(str(rel_image))}" alt="{html_escape(slide["title"])}">
              </div>
            """
        slides_html.append(
            f"""
            <section class="slide">
              <div class="meta">{idx} / {len(manifest["slides"])} · {html_escape(slide.get("source_label", "Project figure"))}</div>
              <h2>{html_escape(slide["title"])}</h2>
              {image_html}
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
      --bg: #ffffff;
      --panel: #ffffff;
      --line: #d2d8df;
      --text: #111111;
      --muted: #5c6670;
      --accent: #27313b;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--text);
      background: var(--bg);
    }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 40px 24px 72px; }}
    h1 {{ font-size: 2.6rem; margin-bottom: 10px; }}
    .intro {{ color: var(--muted); line-height: 1.7; max-width: 900px; margin-bottom: 30px; }}
    .slide {{ background: var(--panel); border: 1px solid var(--line); padding: 24px; margin-bottom: 28px; }}
    .meta {{ color: var(--accent); font-size: 0.95rem; margin-bottom: 10px; }}
    .image-wrap {{ background: #ffffff; padding: 14px; margin: 18px 0; border: 1px solid var(--line); }}
    img {{ display: block; width: 100%; }}
    p {{ line-height: 1.75; color: #2c3640; font-size: 1.06rem; }}
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
        lines.extend([f"## Slide {idx}: {slide['title']}", ""])
        if "image" in slide:
            rel_image = Path(slide["image"]).relative_to("results")
            lines.extend([f"![{slide['title']}](../{rel_image})", ""])
        lines.extend([slide["caption"], ""])
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
