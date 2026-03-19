from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = ROOT / "cvpr demo"
PRE_IMAGE = Path("/Volumes/T9/ICML/Part_one_pre_def_rgb/DJI_20250929100235_0457_D.JPG")
POST_IMAGE = Path("/Volumes/T9/ICML/205_Post_Def_rgb/DJI_20250929124641_0175_D.JPG")

PANEL_GAP = 24
FIGURE_TITLE_BAND = 100
FIGURE_CAPTION_BAND = 120
MAX_SOURCE_DIM = 5200
DETECT_MAX_DIM = 2560


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Tahoma Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Tahoma.ttf",
        "/Library/Fonts/Tahoma.ttf",
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

def dynamic_resize(image: Image.Image, target_width: int) -> Image.Image:
    scale = target_width / float(image.width)
    target_height = int(image.height * scale)
    return image.resize((target_width, target_height), Image.LANCZOS)

def open_full_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        w, h = image.size
        m = max(w, h)
        if m > MAX_SOURCE_DIM:
            scale = MAX_SOURCE_DIM / m
            image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        return image

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

def score_boxes(base_rgb: np.ndarray, boxes: list[tuple[int, int, int, int]], label: str) -> list[tuple[float, tuple[int, int, int, int]]]:
    rgb = base_rgb.astype(np.float32) / 255.0
    gray = rgb.mean(axis=2)
    r, g_channel = rgb[:, :, 0], rgb[:, :, 1]
    exg = (2.0 * g_channel) - r - rgb[:, :, 2]
    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    post = label == "Post_Defoliation"
    for box in boxes:
        x_pos, y_pos, box_w, box_h = box
        x1 = min(base_rgb.shape[1], x_pos + box_w)
        y1 = min(base_rgb.shape[0], y_pos + box_h)
        patch_rgb = rgb[y_pos:y1, x_pos:x1]
        patch_gray = gray[y_pos:y1, x_pos:x1]
        patch_exg = exg[y_pos:y1, x_pos:x1]
        if patch_rgb.size == 0:
            continue
        whiteness = float(np.mean(np.minimum.reduce([patch_rgb[:, :, 0], patch_rgb[:, :, 1], patch_rgb[:, :, 2]])))
        local_contrast = float(patch_gray.std())
        green_leaf = float(np.mean(np.maximum(0.0, patch_exg)))
        if post:
            score = whiteness + 0.62 * local_contrast - 0.22 * green_leaf
        else:
            score = whiteness + 0.48 * local_contrast - 0.52 * green_leaf
        scored.append((score, box))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored

def select_salient_boxes(base_rgb: np.ndarray, boxes: list[tuple[int, int, int, int]], label: str, *, max_boxes: int, min_center_gap: int) -> list[tuple[int, int, int, int]]:
    chosen: list[tuple[int, int, int, int]] = []
    for _, box in score_boxes(base_rgb, boxes, label):
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
            if iou(box, prev) > 0.35:
                keep = False
                break
        if keep:
            chosen.append(box)
        if len(chosen) >= max_boxes:
            break
    return chosen

def cotton_candidate_boxes(img_rgb: np.ndarray, label: str) -> list[tuple[int, int, int, int]]:
    h, w = img_rgb.shape[:2]
    scale = DETECT_MAX_DIM / max(h, w)
    if scale < 1.0:
        dw, dh = int(w * scale), int(h * scale)
        small = cv2.resize(img_rgb, (dw, dh), interpolation=cv2.INTER_AREA)
    else:
        dw, dh = w, h
        scale = 1.0
        small = img_rgb.copy()

    small_f = small.astype(np.float32) / 255.0
    r, g_ch, b = small_f[:, :, 0], small_f[:, :, 1], small_f[:, :, 2]
    exg = np.clip((2.0 * g_ch) - r - b, -1.0, 1.0)
    exg_n = cv2.normalize(exg, None, 0.0, 1.0, cv2.NORM_MINMAX)

    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    eq = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    gray = cv2.cvtColor(eq, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(eq, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0

    d_small = max(3, int(max(dw, dh) * 0.0045))
    d_large = max(7, int(max(dw, dh) * 0.022))
    se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_small | 1, d_small | 1))
    se_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_large | 1, d_large | 1))
    th_small = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se_small)
    th_large = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se_large)
    th = cv2.max(th_small, th_large).astype(np.float32)

    if label == "Post_Defoliation":
        soil_suppress = np.clip((exg_n - 0.08) / 0.55, 0.0, 1.0)
        leaf_penalty = np.clip((exg_n * saturation - 0.22) / 0.55, 0.0, 1.0)
        weighted = th * (0.35 + 0.65 * soil_suppress) * (1.0 - 0.55 * leaf_penalty)
    else:
        canopy = np.clip((exg_n - 0.12) / 0.78, 0.0, 1.0)
        leaf_glare = np.clip(exg_n * (saturation ** 0.85) - 0.42, 0.0, 1.0)
        weighted = th * (0.2 + 0.8 * np.sqrt(canopy + 0.05)) * (1.0 - 0.72 * leaf_glare)

    weighted = np.clip(weighted, 0, 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(weighted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_val = max(8, min(40, int(otsu_val * 0.22))) 
    _, boll_mask = cv2.threshold(weighted, thresh_val, 255, cv2.THRESH_BINARY)
    boll_mask = cv2.morphologyEx(boll_mask, cv2.MORPH_OPEN, se_small, iterations=1)
    contours, _ = cv2.findContours(boll_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        x_pos, y_pos, width, height = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        aspect = max(width, height) / (min(width, height) + 1e-6)
        if aspect > 2.8:  continue
        bbox_area = width * height
        if bbox_area <= 0: continue
        fill_ratio = contour_area / float(bbox_area)
        if fill_ratio < 0.12: continue
        if bbox_area > 0.0024 * (dw * dh): continue
        if width > 0.09 * dw or height > 0.09 * dh: continue
        roi_mask = np.zeros((dh, dw), dtype=np.uint8)
        cv2.drawContours(roi_mask, [contour], -1, 255, -1)
        pixels = roi_mask == 255
        region_s = saturation[pixels]
        region_exg = exg_n[pixels]
        region_gray = gray.astype(np.float32)[pixels] / 255.0
        if len(region_s) == 0: continue
        if float(np.mean(region_s)) > 0.82 and label == "Pre_Defoliation": continue
        if float(np.mean(region_gray)) < 0.05: continue
        if label == "Pre_Defoliation" and float(np.mean(region_exg)) > 0.88: continue
        candidate_boxes.append((x_pos, y_pos, width, height, contour_area))

    inv_scale = 1.0 / scale
    full_res_boxes: list[tuple[int, int, int, int]] = []
    for x_pos, y_pos, width, height, _ in candidate_boxes:
        x0 = int(x_pos * inv_scale)
        y0 = int(y_pos * inv_scale)
        w0 = max(1, int(width * inv_scale))
        h0 = max(1, int(height * inv_scale))
        full_res_boxes.append((x0, y0, w0, h0))

    return full_res_boxes

def gaussian_heatmap(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    height, width = shape
    max_dim = 900
    sc = min(1.0, max_dim / max(height, width))
    small_h = max(1, int(height * sc))
    small_w = max(1, int(width * sc))
    yy, xx = np.mgrid[0:small_h, 0:small_w].astype(np.float32)
    heat = np.zeros((small_h, small_w), dtype=np.float32)
    for x_pos, y_pos, box_w, box_h in boxes:
        cx = (x_pos + box_w / 2.0) * sc
        cy = (y_pos + box_h / 2.0) * sc
        sigma_x = max(1.5, box_w * sc * 0.45)
        sigma_y = max(1.5, box_h * sc * 0.45)
        heat += np.exp(-(((xx - cx) ** 2) / (2.0 * sigma_x**2) + ((yy - cy) ** 2) / (2.0 * sigma_y**2)))
    max_expected = 2.0
    heat = np.clip(heat, 0.0, max_expected) / max_expected
    if sc < 1.0:
        heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(heat, 0.0, 1.0)

def build_cotton_heatmap(base_rgb: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    heat = gaussian_heatmap(base_rgb.shape[:2], boxes)
    heat = np.where(heat > 0.04, heat, 0.0)
    if heat.max() > 0:
        heat = heat / heat.max()
    heat_norm = np.clip(heat * 255, 0, 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    base = base_rgb.astype(np.float32)
    alpha = np.clip((heat**0.6)[..., None] * 0.65, 0.0, 0.65)
    blended = np.clip(base * (1.0 - alpha) + heatmap_colored.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")

def draw_detection_overlay(base_rgb: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    out = base_rgb.copy()
    color = (0, 255, 0)
    thickness = max(2, int(base_rgb.shape[1] * 0.0016))
    for x_pos, y_pos, box_w, box_h in boxes:
        cv2.rectangle(out, (int(x_pos), int(y_pos)), (int(x_pos + box_w), int(y_pos + box_h)), color, thickness)
    return Image.fromarray(out, mode="RGB")

def run_cotton_visual_pipeline(image: Image.Image, label: str) -> tuple[Image.Image, Image.Image, Image.Image, list[tuple[int, int, int, int]]]:
    arr = np.asarray(image.convert("RGB"))
    raw_boxes = cotton_candidate_boxes(arr, label)
    max_boxes = 28000 
    gap = 2 
    boxes = select_salient_boxes(arr, raw_boxes, label, max_boxes=max_boxes, min_center_gap=gap)
    heatmap_view = build_cotton_heatmap(arr, boxes)
    detection_view = draw_detection_overlay(arr, boxes)
    return image, heatmap_view, detection_view, boxes

def compose_three_panel_figure(
    original: Image.Image,
    response_map: Image.Image,
    detection: Image.Image,
    boxes: list,
    *,
    figure_title: str,
    caption: str,
    panel_labels: tuple[str, str, str],
    output_path: Path,
) -> None:
    total_w = 3840
    panel_w = (total_w - (PANEL_GAP * 2)) // 3
    sc = panel_w / float(original.width)
    panel_h = max(1, int(original.height * sc))

    panels = [
        dynamic_resize(original, panel_w),
        dynamic_resize(response_map, panel_w),
        dynamic_resize(detection, panel_w),
    ]

    board = Image.new("RGB", (total_w, panel_h), "white")
    draw = ImageDraw.Draw(board)
    label_font = load_font(42, bold=True) # visual Tahoma 14 equivalence for large image

    labels = list(panel_labels)
    labels[2] += f" (Total bolls: {len(boxes)})"

    for index, panel_img in enumerate(panels):
        x0 = index * (panel_w + PANEL_GAP)
        board.paste(panel_img, (x0, 0))
        draw.rectangle((x0, 0, x0 + panel_w - 1, panel_h - 1), outline="#b8c0ca", width=2)
        title = labels[index]
        tb = draw.textbbox((0, 0), title, font=label_font)
        tw = tb[2] - tb[0]
        pad_x = 24
        pad_y = 12
        draw.rectangle(
            (x0 + 20, panel_h - (tb[3]-tb[1]) - pad_y*2 - 20, x0 + 20 + tw + pad_x * 2, panel_h - 20),
            fill="white",
            outline="#c5ccd4",
            width=2,
        )
        draw.text((x0 + 20 + pad_x, panel_h - (tb[3]-tb[1]) - pad_y*2 - 16), title, font=label_font, fill="#1a1a1a")

    total_h = FIGURE_TITLE_BAND + panel_h + FIGURE_CAPTION_BAND
    canvas = Image.new("RGB", (total_w, total_h), "white")
    canvas.paste(board, (0, FIGURE_TITLE_BAND))
    draw = ImageDraw.Draw(canvas)
    draw.line((0, FIGURE_TITLE_BAND - 1, total_w, FIGURE_TITLE_BAND - 1), fill="#9aa3ad", width=2)
    draw.line((0, FIGURE_TITLE_BAND + panel_h, total_w, FIGURE_TITLE_BAND + panel_h), fill="#9aa3ad", width=2)

    title_font = load_font(60, bold=True)
    draw.text((48, 20), figure_title, font=title_font, fill="#111111")

    cap_font = load_font(38)
    cap_y = FIGURE_TITLE_BAND + panel_h + 28
    draw.text((48, cap_y), caption, font=cap_font, fill="#3d4852")
    canvas.save(output_path, format="PNG", optimize=True)

def add_academic_header(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    width, height = image.size
    header_h = 160
    canvas = Image.new("RGB", (width, height + header_h), "white")
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.line((0, header_h - 2, width, header_h - 2), fill="#c6ccd4", width=2)
    draw.text((48, 30), title, font=load_font(52, bold=True), fill="#111111")
    draw.text((48, 96), subtitle, font=load_font(38), fill="#5c6670")
    return canvas

def save_single_panel(source: Path, title: str, subtitle: str, output_name: str) -> None:
    img = open_full_image(source)
    panel = dynamic_resize(img, 3840)
    add_academic_header(panel, title, subtitle).save(ASSET_DIR / output_name, format="PNG", optimize=True)

def save_heat_panel(source: Path, label: str, title: str, subtitle: str, output_name: str) -> None:
    image = open_full_image(source)
    _, heatmap, detection, boxes = run_cotton_visual_pipeline(image, label)
    panel = dynamic_resize(heatmap, 3840)
    add_academic_header(panel, title, f"{subtitle} (Detected {len(boxes)} items)").save(ASSET_DIR / output_name, format="PNG", optimize=True)

def save_detect_panel(source: Path, label: str, title: str, subtitle: str, output_name: str) -> None:
    image = open_full_image(source)
    _, _, detection, boxes = run_cotton_visual_pipeline(image, label)
    panel = dynamic_resize(detection, 3840)
    add_academic_header(panel, title, f"{subtitle} (Detected {len(boxes)} items)").save(ASSET_DIR / output_name, format="PNG", optimize=True)

def build_scene_analysis_figure(source: Path, label: str, figure_title: str, caption: str, output_name: str) -> None:
    image = open_full_image(source)
    original, response, detection, boxes = run_cotton_visual_pipeline(image, label)
    compose_three_panel_figure(
        original, response, detection, boxes,
        figure_title=figure_title,
        caption=caption,
        panel_labels=("Original UAV image", "Cotton response map", "Detected cotton regions"),
        output_path=ASSET_DIR / output_name,
    )

def main() -> None:
    import shutil
    print("Generating pre-defoliation images...")
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    save_single_panel(PRE_IMAGE, "Pre-Defoliation UAV Image", "Full frame from the field before defoliation", "01_pre_original.png")
    save_heat_panel(PRE_IMAGE, "Pre_Defoliation", "Pre-Defoliation Cotton Response Map", "Response map", "02_pre_heatmap.png")
    save_detect_panel(PRE_IMAGE, "Pre_Defoliation", "Pre-Defoliation Cotton Detects", "Bounding boxes overlaid", "03_pre_bounding_box.png")
    build_scene_analysis_figure(PRE_IMAGE, "Pre_Defoliation", "Pre-defoliation scene analysis", "Scene showing original UAV image, cotton response map, and detections", "04_pre_composite.png")

    print("Generating post-defoliation images...")
    save_single_panel(POST_IMAGE, "Post-Defoliation UAV Image", "Full frame after defoliation", "05_post_original.png")
    save_heat_panel(POST_IMAGE, "Post_Defoliation", "Post-Defoliation Cotton Response Map", "Response map", "06_post_heatmap.png")
    save_detect_panel(POST_IMAGE, "Post_Defoliation", "Post-Defoliation Cotton Detects", "Bounding boxes overlaid", "07_post_bounding_box.png")
    build_scene_analysis_figure(POST_IMAGE, "Post_Defoliation", "Post-defoliation scene analysis", "Scene showing original UAV image, cotton response map, and detections", "08_post_composite.png")

    print("Copying k_comparison analysis plots...")
    plots_dir = ROOT / "results" / "plots" / "k_comparison"
    if plots_dir.exists():
        for f in sorted(plots_dir.glob("*.png")):
            print(f"Copying {f.name}...")
            shutil.copy(f, ASSET_DIR / f.name)
    else:
        print("Warning: Directory results/plots/k_comparison not found")
        
    print("All image generation operations completed.")

if __name__ == "__main__":
    main()
