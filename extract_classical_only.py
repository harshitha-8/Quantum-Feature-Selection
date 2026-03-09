#!/usr/bin/env python3
"""
Fast classical feature extraction (no LLM/CLIP).
Runs on all subfolders of the given root path.
"""
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import argparse

def extract_color_features(image_rgb):
    img = image_rgb.astype(float) / 255.0
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    ExG = 2 * G - R - B
    RB_Ratio = R / (B + 1e-6)
    NGRDI = (G - R) / (G + R + 1e-6)   # Normalized Green-Red Difference
    return {
        "Mean_ExG": float(np.mean(ExG)),
        "Std_ExG": float(np.std(ExG)),
        "Mean_RBR": float(np.mean(RB_Ratio)),
        "Mean_NGRDI": float(np.mean(NGRDI)),
        "Mean_R": float(np.mean(R)),
        "Mean_G": float(np.mean(G)),
        "Mean_B": float(np.mean(B)),
    }

def extract_texture_features(image_gray):
    # Downsample for speed on large images
    h, w = image_gray.shape
    if h > 1000 or w > 1000:
        image_gray = cv2.resize(image_gray, (min(w, 1000), min(h, 1000)))
    entropy_val = shannon_entropy(image_gray)
    glcm = graycomatrix(image_gray, distances=[1], angles=[0], levels=256,
                        symmetric=True, normed=True)
    return {
        "Entropy": float(entropy_val),
        "Contrast": float(graycoprops(glcm, 'contrast')[0, 0]),
        "Homogeneity": float(graycoprops(glcm, 'homogeneity')[0, 0]),
        "Energy": float(graycoprops(glcm, 'energy')[0, 0]),
        "Correlation": float(graycoprops(glcm, 'correlation')[0, 0]),
    }

def process_dataset(root_path, limit=None):
    data = []
    valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    total_processed = 0

    for dirpath, _, files in os.walk(root_path):
        folder_name = os.path.basename(dirpath).lower()
        if 'pre' in folder_name:
            label = "Pre_Defoliation"
        elif 'post' in folder_name:
            label = "Post_Defoliation"
        else:
            continue   # skip root or unknown folders

        img_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]
        if limit:
            img_files = img_files[:limit]

        print(f"[{label}] {folder_name}: {len(img_files)} images")

        for i, filename in enumerate(img_files):
            filepath = os.path.join(dirpath, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            try:
                row = {"Filename": filename, "Folder": folder_name, "Label": label}
                row.update(extract_color_features(img_rgb))
                row.update(extract_texture_features(img_gray))
                data.append(row)
                total_processed += 1
                if (i + 1) % 50 == 0:
                    print(f"  ... {i+1}/{len(img_files)} done")
            except Exception as e:
                print(f"  Error on {filename}: {e}")

    print(f"\nTotal images processed: {total_processed}")
    return pd.DataFrame(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="icml_features_full.csv")
    parser.add_argument("--limit", type=int, default=None, help="Max images per folder")
    args = parser.parse_args()

    df = process_dataset(args.input, args.limit)
    if not df.empty:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
        print(df['Label'].value_counts())
    else:
        print("No data extracted.")
