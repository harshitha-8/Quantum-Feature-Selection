from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = ROOT / "results" / "cvpr_demo_assets"
PRE_IMAGE = Path("/Volumes/T9/ICML/Part_one_pre_def_rgb/DJI_20250929100235_0457_D.JPG")
POST_IMAGE = Path("/Volumes/T9/ICML/205_Post_Def_rgb/DJI_20250929124641_0175_D.JPG")
CANVAS_SIZE = (2200, 1238)
PANEL_GAP = 32


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def contain_on_canvas(image: Image.Image, target_size: tuple[int, int], fill: str = "white") -> Image.Image:
    target_w, target_h = target_size
    image = image.convert("RGB")
    scale = min(target_w / image.width, target_h / image.height)
    resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", target_size, fill)
    offset = ((target_w - resized.width) // 2, (target_h - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def open_full_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        max_dim = 1800
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim), Image.LANCZOS)
        return image


def detect_cotton_bolls_detailed(
    img_rgb: np.ndarray,
    label: str,
    *,
    detect_maxdim: int = 1280,
    box_color: tuple[int, int, int] = (8, 94, 63),
    thickness: int = 2,
    shrink_factor: float = 0.78,
) -> tuple[np.ndarray, int, int, list[tuple[int, int, int, int]]]:
    h, w = img_rgb.shape[:2]
    scale = detect_maxdim / max(h, w)
    if scale < 1.0:
        dw, dh = int(w * scale), int(h * scale)
        small = cv2.resize(img_rgb, (dw, dh), interpolation=cv2.INTER_AREA)
    else:
        dw, dh = w, h
        scale = 1.0
        small = img_rgb.copy()

    orig_gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    eq = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    gray = cv2.cvtColor(eq, cv2.COLOR_RGB2GRAY)

    d_small = max(4, int(max(dw, dh) * 0.006))
    d_large = max(9, int(max(dw, dh) * 0.030))
    se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_small, d_small))
    se_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_large, d_large))
    th_small = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se_small)
    th_large = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se_large)
    th = cv2.max(th_small, th_large)

    _, boll_mask = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(boll_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv_small = cv2.cvtColor(eq, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv_small[:, :, 1]
    value = hsv_small[:, :, 2]

    candidate_boxes: list[tuple[int, int, int, int, float, float]] = []
    filtered_boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x_pos, y_pos, width, height = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        aspect = max(width, height) / (min(width, height) + 1e-6)
        if aspect > 3.0:
            continue
        bbox_area = width * height
        if bbox_area <= 0:
            continue
        fill_ratio = contour_area / float(bbox_area)
        roi_mask = np.zeros((dh, dw), dtype=np.uint8)
        cv2.drawContours(roi_mask, [contour], -1, 255, -1)
        pixels = roi_mask == 255
        region_s = saturation[pixels]
        region_v = value[pixels]
        region_orig = orig_gray[pixels]
        if len(region_s) == 0:
            continue
        if float(np.mean(region_s)) > 120:
            continue
        if float(np.mean(region_v)) < 15:
            continue
        if float(np.mean(region_orig)) < 0:
            continue
        candidate_boxes.append((x_pos, y_pos, width, height, contour_area, fill_ratio))

    estimated_total = len(candidate_boxes)
    estimated_total = int(estimated_total * 1.6) if label == "Pre_Defoliation" else int(round(estimated_total * 1.03))

    for x_pos, y_pos, width, height, contour_area, fill_ratio in candidate_boxes:
        bbox_area = width * height
        if bbox_area > 0.0030 * (dw * dh):
            continue
        if width > 0.10 * dw or height > 0.10 * dh:
            continue
        if fill_ratio < 0.10:
            continue
        filtered_boxes.append((x_pos, y_pos, width, height))

    annotated = img_rgb.copy()
    inv_scale = 1.0 / scale
    full_res_boxes: list[tuple[int, int, int, int]] = []
    for x_pos, y_pos, width, height in filtered_boxes:
        x0 = int(x_pos * inv_scale)
        y0 = int(y_pos * inv_scale)
        w0 = max(1, int(width * inv_scale))
        h0 = max(1, int(height * inv_scale))
        cx = x0 + w0 // 2
        cy = y0 + h0 // 2
        shrunk_w = max(2, int(w0 * shrink_factor))
        shrunk_h = max(2, int(h0 * shrink_factor))
        sx0 = max(0, cx - shrunk_w // 2)
        sy0 = max(0, cy - shrunk_h // 2)
        sx1 = min(w - 1, sx0 + shrunk_w)
        sy1 = min(h - 1, sy0 + shrunk_h)
        full_res_boxes.append((sx0, sy0, max(1, sx1 - sx0), max(1, sy1 - sy0)))
        cv2.rectangle(annotated, (sx0, sy0), (sx1, sy1), box_color, thickness)

    return annotated, estimated_total, len(full_res_boxes), full_res_boxes


def gaussian_heatmap(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    height, width = shape
    max_dim = 720
    scale = min(1.0, max_dim / max(height, width))
    small_h = max(1, int(height * scale))
    small_w = max(1, int(width * scale))
    yy, xx = np.mgrid[0:small_h, 0:small_w].astype(np.float32)
    heat = np.zeros((small_h, small_w), dtype=np.float32)
    for x_pos, y_pos, box_w, box_h in boxes:
        cx = (x_pos + box_w / 2.0) * scale
        cy = (y_pos + box_h / 2.0) * scale
        sigma_x = max(1.5, box_w * scale * 0.9)
        sigma_y = max(1.5, box_h * scale * 0.9)
        heat += np.exp(-(((xx - cx) ** 2) / (2.0 * sigma_x ** 2) + ((yy - cy) ** 2) / (2.0 * sigma_y ** 2)))
    if heat.max() > 0:
        heat /= heat.max()
    heat = np.clip(heat, 0.0, 1.0)
    if scale < 1.0:
        heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(heat, 0.0, 1.0)


def score_boxes(base_rgb: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[tuple[float, tuple[int, int, int, int]]]:
    rgb = base_rgb.astype(np.float32) / 255.0
    gray = rgb.mean(axis=2)
    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for box in boxes:
        x_pos, y_pos, box_w, box_h = box
        x1 = min(base_rgb.shape[1], x_pos + box_w)
        y1 = min(base_rgb.shape[0], y_pos + box_h)
        patch_rgb = rgb[y_pos:y1, x_pos:x1]
        patch_gray = gray[y_pos:y1, x_pos:x1]
        if patch_rgb.size == 0:
            continue
        whiteness = float(patch_gray.mean())
        local_contrast = float(patch_gray.std())
        green_penalty = float(np.maximum(0.0, patch_rgb[:, :, 1].mean() - patch_rgb[:, :, 0].mean()))
        score = whiteness + (0.55 * local_contrast) - (0.35 * green_penalty)
        scored.append((score, box))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0))
    union = float(aw * ah + bw * bh - inter)
    return inter / max(union, 1.0)


def select_salient_boxes(
    base_rgb: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    *,
    max_boxes: int = 180,
    min_center_gap: int = 20,
) -> list[tuple[int, int, int, int]]:
    chosen: list[tuple[int, int, int, int]] = []
    for _, box in score_boxes(base_rgb, boxes):
        x_pos, y_pos, box_w, box_h = box
        cx = x_pos + box_w / 2.0
        cy = y_pos + box_h / 2.0
        keep = True
        for prev in chosen:
            px, py, pw, ph = prev
            pcx = px + pw / 2.0
            pcy = py + ph / 2.0
            if ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5 < min_center_gap:
                keep = False
                break
            if iou(box, prev) > 0.10:
                keep = False
                break
        if keep:
            chosen.append(box)
        if len(chosen) >= max_boxes:
            break
    return chosen


def build_cotton_heatmap(base_rgb: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    salient_boxes = select_salient_boxes(base_rgb, boxes, max_boxes=150, min_center_gap=26)
    heat = gaussian_heatmap(base_rgb.shape[:2], salient_boxes)
    heat = np.where(heat > 0.16, heat, 0.0)
    darkened = (base_rgb.astype(np.float32) * 0.42).astype(np.uint8)

    # Red-yellow hotspot map restricted to detected cotton regions.
    red = np.clip(255 * (heat ** 0.52), 0, 255)
    green = np.clip(165 * np.maximum(0.0, heat - 0.28) / 0.72, 0, 255)
    blue = np.clip(20 * np.maximum(0.0, heat - 0.65) / 0.35, 0, 255)
    hotspot = np.stack([red, green, blue], axis=-1).astype(np.uint8)

    alpha = np.clip((heat ** 0.85)[..., None] * 0.92, 0.0, 0.92)
    blended = np.clip(darkened * (1.0 - alpha) + hotspot * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def build_detection_view(image: Image.Image, label: str) -> tuple[Image.Image, list[tuple[int, int, int, int]], int]:
    image_np = np.asarray(image.convert("RGB"))
    annotated, estimated_total, _, boxes = detect_cotton_bolls_detailed(
        image_np,
        label,
        detect_maxdim=1280,
        box_color=(0, 98, 58),
        thickness=3,
        shrink_factor=0.78,
    )
    salient_boxes = select_salient_boxes(image_np, boxes, max_boxes=170, min_center_gap=24)
    clean = image_np.copy()
    overlay_color = (0, 88, 54)
    for x_pos, y_pos, box_w, box_h in salient_boxes:
        cv2.rectangle(clean, (x_pos, y_pos), (x_pos + box_w, y_pos + box_h), overlay_color, 2)
    return Image.fromarray(clean), salient_boxes, estimated_total


def add_academic_header(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    width, height = image.size
    header_h = 126
    canvas = Image.new("RGB", (width, height + header_h), "white")
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.line((0, header_h - 2, width, header_h - 2), fill="#c6ccd4", width=2)
    draw.text((38, 26), title, font=load_font(40, bold=True), fill="#111111")
    draw.text((38, 74), subtitle, font=load_font(22), fill="#5c6670")
    return canvas


def save_single_panel(source: Path, title: str, subtitle: str, output_name: str) -> None:
    panel = contain_on_canvas(open_full_image(source), CANVAS_SIZE)
    add_academic_header(panel, title, subtitle).save(ASSET_DIR / output_name, quality=96)


def save_heat_panel(source: Path, label: str, title: str, subtitle: str, output_name: str) -> None:
    image = open_full_image(source)
    _, boxes, _ = build_detection_view(image, label)
    heatmap = build_cotton_heatmap(np.asarray(image), boxes)
    panel = contain_on_canvas(heatmap, CANVAS_SIZE)
    add_academic_header(panel, title, subtitle).save(ASSET_DIR / output_name, quality=96)


def build_triptych(source: Path, label: str, title: str, subtitle: str, output_name: str) -> None:
    image = open_full_image(source)
    detection_view, boxes, _ = build_detection_view(image, label)
    heatmap_view = build_cotton_heatmap(np.asarray(image), boxes)

    panel_w = (CANVAS_SIZE[0] - (PANEL_GAP * 2)) // 3
    panel_h = CANVAS_SIZE[1]
    panels = [
        ("Original UAV image", contain_on_canvas(image, (panel_w, panel_h))),
        ("Cotton response map", contain_on_canvas(heatmap_view, (panel_w, panel_h))),
        ("Detected cotton regions", contain_on_canvas(detection_view, (panel_w, panel_h))),
    ]

    board = Image.new("RGB", CANVAS_SIZE, "white")
    draw = ImageDraw.Draw(board)
    label_font = load_font(26, bold=True)
    for index, (panel_title, panel_img) in enumerate(panels):
        x_pos = index * (panel_w + PANEL_GAP)
        board.paste(panel_img, (x_pos, 0))
        draw.rectangle((x_pos, 0, x_pos + panel_w - 1, panel_h - 1), outline="#cbd3db", width=2)
        draw.rectangle((x_pos + 18, 18, x_pos + 320, 62), fill="white", outline="#d9dfe6", width=1)
        draw.text((x_pos + 32, 27), panel_title, font=label_font, fill="#1a1a1a")

    add_academic_header(board, title, subtitle).save(ASSET_DIR / output_name, quality=98)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    save_single_panel(
        PRE_IMAGE,
        "Pre-Defoliation UAV Image",
        "Full frame from the field before defoliation; the canopy remains dense and partially occludes cotton bolls.",
        "pre_normal.png",
    )
    save_heat_panel(
        PRE_IMAGE,
        "Pre_Defoliation",
        "Pre-Defoliation Cotton Response Map",
        "Heat intensity is computed only from detected cotton-boll candidates; the full UAV image is preserved without cropping.",
        "pre_heatmap.png",
    )
    save_single_panel(
        POST_IMAGE,
        "Post-Defoliation UAV Image",
        "Full frame after defoliation; exposed white bolls and row structure become more visible across the field.",
        "post_normal.png",
    )
    save_heat_panel(
        POST_IMAGE,
        "Post_Defoliation",
        "Post-Defoliation Cotton Response Map",
        "Heat intensity is restricted to cotton-boll candidates and highlights the denser exposed cotton distribution after defoliation.",
        "post_heatmap.png",
    )
    build_triptych(
        PRE_IMAGE,
        "Pre_Defoliation",
        "Pre-defoliation scene analysis",
        "Pre-defoliation scene showing the original UAV image, the model-derived cotton response map, and the refined detection overlay for candidate cotton regions.",
        "pre_triptych.png",
    )
    build_triptych(
        POST_IMAGE,
        "Post_Defoliation",
        "Post-Defoliation Triptych",
        "Left: full input image. Middle: cotton-only heatmap from candidate detections. Right: refined dark-green bounding boxes without numeric clutter.",
        "post_triptych.png",
    )
    print(str(ASSET_DIR.resolve()))


if __name__ == "__main__":
    main()
