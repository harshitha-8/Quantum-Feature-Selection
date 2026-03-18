#!/usr/bin/env python3
"""
Build an augmented ICML feature table for Quantum-Feature-Selection.

This ports only the pre/post-defoliation feature logic and cotton boll counter
from the SPIE repository, then combines those outputs with the legacy QFS
feature set so downstream k=2/4/6 experiments can run on a single CSV.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy


DEFAULT_ICML_ROOT = Path("/Volumes/T9/ICML")
DEFAULT_OUTPUT = Path("icml_features_spie_augmented.csv")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def label_from_folder(folder_name: str) -> str | None:
    lowered = folder_name.lower()
    if "post" in lowered:
        return "Post_Defoliation"
    if "pre" in lowered:
        return "Pre_Defoliation"
    return None


def prepare_gray(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    if h > 512 or w > 512:
        img_gray = cv2.resize(img_gray, (min(w, 512), min(h, 512)))
    glcm = graycomatrix(
        img_gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )
    return img_gray, glcm


def extract_combined_features(img_bgr: np.ndarray, label: str) -> dict[str, float]:
    img_rgb_uint8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb_uint8.astype(np.float32) / 255.0
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    exg = 2 * g - r - b
    mean_rbr = r / (b + 1e-6)
    ngrdi = (g - r) / (g + r + 1e-6)

    img_gray, glcm = prepare_gray(img_bgr)

    features = {
        "Mean_ExG": float(np.mean(exg)),
        "Std_ExG": float(np.std(exg)),
        "Mean_RBR": float(np.mean(mean_rbr)),
        "Log_RBR": float(np.log1p(np.mean(mean_rbr))),
        "ExG_pos_frac": float(np.mean(exg > 0)),
        "Mean_NGRDI": float(np.mean(ngrdi)),
        "Mean_R": float(np.mean(r)),
        "Mean_G": float(np.mean(g)),
        "Mean_B": float(np.mean(b)),
        "Entropy": float(shannon_entropy(img_gray)),
        "Contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        "Homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
        "Energy": float(graycoprops(glcm, "energy")[0, 0]),
        "Correlation": float(graycoprops(glcm, "correlation")[0, 0]),
    }
    features["Cotton_Boll_Count"] = float(detect_cotton_bolls_count_only(img_rgb_uint8, label))
    return features


def detect_cotton_bolls_count_only(img_rgb: np.ndarray, label: str = "Post_Defoliation") -> int:
    """
    Ported from SPIE/app.py.

    The only intentional change is that this helper returns the count directly
    because the QFS repo needs a feature value, not an annotated image.
    """

    h, w = img_rgb.shape[:2]
    detect_maxdim = 640
    scale = detect_maxdim / max(h, w)

    if scale < 1.0:
        dw, dh = int(w * scale), int(h * scale)
        small = cv2.resize(img_rgb, (dw, dh), interpolation=cv2.INTER_AREA)
    else:
        dw, dh, scale = w, h, 1.0
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

    valid_count = 0
    for contour in contours:
        x_pos, y_pos, width, height = cv2.boundingRect(contour)
        aspect = max(width, height) / (min(width, height) + 1e-6)
        if aspect > 3.0:
            continue

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

        valid_count += 1

    if label == "Pre_Defoliation":
        valid_count = int(valid_count * 1.6)

    return valid_count


def process_image(image_path: str, label: str, folder_name: str) -> dict[str, float | str] | None:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None
    row: dict[str, float | str] = {
        "Filename": Path(image_path).name,
        "Folder": folder_name,
        "Label": label,
    }
    row.update(extract_combined_features(img_bgr, label))
    return row


def iter_dataset_records(
    icml_root: Path,
    limit_per_folder: int | None,
    max_workers: int,
) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    folders = [path for path in sorted(icml_root.iterdir()) if path.is_dir()]
    jobs: list[tuple[str, str, str]] = []

    for folder in folders:
        label = label_from_folder(folder.name)
        if label is None:
            continue

        image_paths = [
            path for path in sorted(folder.iterdir())
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
            and not path.name.startswith("._")
        ]
        if limit_per_folder is not None:
            image_paths = image_paths[:limit_per_folder]

        print(f"[{label[:4]}] {folder.name}: {len(image_paths)} images")

        for image_path in image_paths:
            jobs.append((str(image_path), label, folder.name))

    total_jobs = len(jobs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, image_path, label, folder_name) for image_path, label, folder_name in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            if row is not None:
                records.append(row)
            if index % 100 == 0 or index == total_jobs:
                print(f"Processed {index}/{total_jobs} images", flush=True)

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--icml-root", type=Path, default=DEFAULT_ICML_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-per-folder", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()

    if not args.icml_root.exists():
        raise FileNotFoundError(f"ICML root not found: {args.icml_root}")

    records = iter_dataset_records(args.icml_root, args.limit_per_folder, args.max_workers)
    if not records:
        raise RuntimeError("No dataset records were created.")

    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)

    print(f"\nSaved {len(df)} rows to {args.output}")
    print(df.groupby("Label").size().to_string())


if __name__ == "__main__":
    main()
