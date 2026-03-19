from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = ROOT / "results" / "cvpr_demo_assets"
PRE_IMAGE = Path("/Volumes/T9/ICML/Part_one_pre_def_rgb/DJI_20250929100235_0457_D.JPG")
POST_IMAGE = Path("/Volumes/T9/ICML/205_Post_Def_rgb/DJI_20250929124641_0175_D.JPG")


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def resize_cover(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    target_w, target_h = target_size
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    left = max(0, (resized.width - target_w) // 2)
    top = max(0, (resized.height - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h))


def vegetation_response_map(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    exg = np.clip((2.0 * g) - r - b, -1.0, 1.0)
    return (exg + 1.0) / 2.0


def dryness_response_map(image: Image.Image) -> np.ndarray:
    return 1.0 - vegetation_response_map(image)


def colormap(values: np.ndarray) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)
    red = np.clip((v - 0.25) / 0.75, 0.0, 1.0)
    green = np.clip(1.0 - np.abs(v - 0.5) * 2.2, 0.0, 1.0)
    blue = np.clip((0.75 - v) / 0.75, 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def add_title_bar(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    width, height = image.size
    canvas = Image.new("RGB", (width, height + 120), "#0b1627")
    canvas.paste(image, (0, 120))
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 26), title, fill="#eef7ff", font=load_font(42, bold=True))
    draw.text((36, 76), subtitle, fill="#9fc1de", font=load_font(24))
    return canvas


def save_normal_visual(source: Path, title: str, subtitle: str, output_name: str) -> None:
    with Image.open(source) as img:
        visual = resize_cover(img, (1600, 900))
    add_title_bar(visual, title, subtitle).save(ASSET_DIR / output_name, quality=95)


def save_heatmap_visual(source: Path, title: str, subtitle: str, output_name: str, mode: str) -> None:
    with Image.open(source) as img:
        base = resize_cover(img, (1600, 900))
    metric = vegetation_response_map(base) if mode == "vegetation" else dryness_response_map(base)
    metric_label = "Vegetation response heatmap" if mode == "vegetation" else "Defoliation response heatmap"
    heat = Image.fromarray(colormap(metric), mode="RGB")
    overlay = Image.blend(base.convert("RGB"), heat, alpha=0.42)

    board = Image.new("RGB", (1600, 900), "#08111d")
    board.paste(base.resize((790, 900), Image.LANCZOS), (0, 0))
    board.paste(overlay.resize((790, 900), Image.LANCZOS), (810, 0))
    draw = ImageDraw.Draw(board)
    badge_font = load_font(28, bold=True)
    draw.rounded_rectangle((28, 28, 244, 78), radius=20, fill="#0d2238")
    draw.text((46, 40), "Original view", fill="#eef7ff", font=badge_font)
    draw.rounded_rectangle((838, 28, 1240, 78), radius=20, fill="#3b0f17")
    draw.text((856, 40), metric_label, fill="#ffe8ea", font=badge_font)
    add_title_bar(board, title, subtitle).save(ASSET_DIR / output_name, quality=95)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    save_normal_visual(PRE_IMAGE, "Pre-Defoliation Normal Image", "Raw UAV frame with intact green canopy before defoliation.", "pre_normal.png")
    save_heatmap_visual(PRE_IMAGE, "Pre-Defoliation Heatmap", "Heatmap emphasizes strong vegetation response across the canopy.", "pre_heatmap.png", mode="vegetation")
    save_normal_visual(POST_IMAGE, "Post-Defoliation Normal Image", "Raw UAV frame after defoliation with exposed bolls and drier field texture.", "post_normal.png")
    save_heatmap_visual(POST_IMAGE, "Post-Defoliation Heatmap", "Heatmap emphasizes defoliated, low-vegetation regions after treatment.", "post_heatmap.png", mode="dryness")
    print(str(ASSET_DIR.resolve()))


if __name__ == "__main__":
    main()
