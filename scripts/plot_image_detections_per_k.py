#!/usr/bin/env python3
"""
Create image-based detection boards for one pre- and one post-defoliation sample.

For each k in {2,4,6}, this script:
1. trains an SVM on the augmented SPIE-transfer feature table,
2. predicts the selected sample image,
3. runs cotton-boll detection using the predicted pre/post label,
4. saves annotated images and a combined board.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATASET = ROOT / "icml_features_spie_augmented.csv"
DEFAULT_SUBSETS = ROOT / "results" / "spie_transfer" / "spie_transfer_selected_subsets.json"
DEFAULT_OUTPUT = ROOT / "results" / "spie_transfer_eval" / "image_detections"
DEFAULT_PRE = Path("/Volumes/T9/ICML/Part_one_pre_def_rgb/DJI_20250929095743_0311_D.JPG")
DEFAULT_POST = Path("/Volumes/T9/ICML/205_Post_Def_rgb/DJI_20250929124505_0127_D.JPG")


def make_clf() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
        ]
    )


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


def extract_features(img_bgr: np.ndarray, label_for_count: str) -> dict[str, float]:
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
    _, count = detect_cotton_bolls(img_rgb_uint8, label_for_count)
    features["Cotton_Boll_Count"] = float(count)
    return features


def detect_cotton_bolls(img_rgb: np.ndarray, label: str) -> tuple[np.ndarray, int]:
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

    valid = []
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
        valid.append(contour)

    count = len(valid)
    if label == "Pre_Defoliation":
        count = int(count * 1.6)

    annotated = img_rgb.copy()
    box_color = (0, 180, 80)
    thickness = max(2, int(min(h, w) * 0.001))
    cv2.drawContours(annotated, valid, -1, box_color, max(1, thickness - 1))
    badge_h = max(28, int(min(h, w) * 0.045))
    badge_w = max(200, int(min(h, w) * 0.28))
    cv2.rectangle(annotated, (0, 0), (badge_w, badge_h), (6, 6, 16), -1)
    cv2.putText(
        annotated,
        f"BOLLS: {count}",
        (8, int(badge_h * 0.78)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.4, min(h, w) * 0.00055),
        (0, 220, 140),
        max(1, thickness),
        cv2.LINE_AA,
    )
    return annotated, count


def predict_image(
    df: pd.DataFrame,
    subsets: dict[str, list[str]],
    image_path: Path,
) -> dict[str, object]:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    base_features = extract_features(img_bgr, "Post_Defoliation")
    results: dict[str, object] = {
        "original_rgb": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        "filename": image_path.name,
    }
    y = (df["Label"] == "Post_Defoliation").astype(int).values

    for key, feature_names in subsets.items():
        clf = make_clf()
        clf.fit(df[feature_names].values, y)
        x = np.array([[base_features[name] for name in feature_names]])
        probs = clf.predict_proba(x)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = "Post_Defoliation" if pred_idx == 1 else "Pre_Defoliation"

        # Recompute count feature using the predicted class because SPIE counting
        # uses class-aware thresholds.
        image_features = extract_features(img_bgr, pred_label)
        x = np.array([[image_features[name] for name in feature_names]])
        probs = clf.predict_proba(x)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = "Post_Defoliation" if pred_idx == 1 else "Pre_Defoliation"
        annotated, count = detect_cotton_bolls(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), pred_label)

        results[key] = {
            "features": feature_names,
            "pred_label": pred_label,
            "post_prob": float(probs[1]),
            "pre_prob": float(probs[0]),
            "count": int(count),
            "annotated": annotated,
        }
    return results


def save_annotated_images(sample_name: str, sample_result: dict[str, object], outdir: Path) -> None:
    for key in ["k=2", "k=4", "k=6"]:
        panel = sample_result[key]
        out_path = outdir / f"{sample_name}_{key.replace('=', '')}_annotated.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(panel["annotated"], cv2.COLOR_RGB2BGR))


def build_board(pre_result: dict[str, object], post_result: dict[str, object], outdir: Path) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor="white")
    samples = [("Pre Sample", pre_result), ("Post Sample", post_result)]
    columns = ["Original", "k=2", "k=4", "k=6"]

    for row_idx, (row_title, sample) in enumerate(samples):
        for col_idx, col_name in enumerate(columns):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_name == "Original":
                ax.imshow(sample["original_rgb"])
                ax.set_title(f"{row_title}\n{sample['filename']}", fontsize=10, fontweight="bold")
                continue

            panel = sample[col_name]
            ax.imshow(panel["annotated"])
            ax.set_title(
                f"{row_title} | {col_name}\n"
                f"Pred: {panel['pred_label'].replace('_', ' ')}\n"
                f"Post={panel['post_prob']:.3f} | Pre={panel['pre_prob']:.3f} | Bolls={panel['count']}",
                fontsize=9.5,
                fontweight="bold",
            )
            ax.text(
                0.02,
                -0.08,
                ", ".join(panel["features"]),
                transform=ax.transAxes,
                fontsize=8.2,
                color="#455A64",
            )

    fig.suptitle(
        "Cotton Pre/Post Image Detections with Boll Counting\nQFS-based predictions at k=2, 4, 6",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path = outdir / "pre_post_image_detection_board.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_summary_csv(pre_result: dict[str, object], post_result: dict[str, object], outdir: Path) -> Path:
    rows = []
    for sample_name, sample in [("Pre", pre_result), ("Post", post_result)]:
        for key in ["k=2", "k=4", "k=6"]:
            panel = sample[key]
            rows.append(
                {
                    "sample_type": sample_name,
                    "filename": sample["filename"],
                    "k": key,
                    "predicted_label": panel["pred_label"],
                    "post_probability": panel["post_prob"],
                    "pre_probability": panel["pre_prob"],
                    "cotton_boll_count": panel["count"],
                    "features": ", ".join(panel["features"]),
                }
            )
    df = pd.DataFrame(rows)
    out_path = outdir / "pre_post_image_detection_summary.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--subsets-json", type=Path, default=DEFAULT_SUBSETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pre-image", type=Path, default=DEFAULT_PRE)
    parser.add_argument("--post-image", type=Path, default=DEFAULT_POST)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.dataset).dropna().reset_index(drop=True)
    subsets = json.loads(args.subsets_json.read_text())

    pre_result = predict_image(df, subsets, args.pre_image)
    post_result = predict_image(df, subsets, args.post_image)

    save_annotated_images("pre", pre_result, args.output_dir)
    save_annotated_images("post", post_result, args.output_dir)
    board_path = build_board(pre_result, post_result, args.output_dir)
    summary_path = save_summary_csv(pre_result, post_result, args.output_dir)

    print(f"Saved board to {board_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
