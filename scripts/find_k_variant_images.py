#!/usr/bin/env python3
"""
Find dataset images with the largest k-by-k prediction variation.

This script ranks images by confidence spread across the selected k=2/4/6
subsets and writes a small CSV summary for reproducibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATASET = ROOT / "icml_features_spie_augmented.csv"
DEFAULT_SUBSETS = ROOT / "results" / "spie_transfer" / "spie_transfer_selected_subsets.json"
DEFAULT_OUTPUT = ROOT / "results" / "spie_transfer_eval" / "k_variant_image_candidates.csv"


def make_clf() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--subsets-json", type=Path, default=DEFAULT_SUBSETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset).dropna().reset_index(drop=True)
    subsets: dict[str, list[str]] = json.loads(args.subsets_json.read_text())
    y = (df["Label"] == "Post_Defoliation").astype(int).values

    out = df[["Filename", "Folder", "Label"]].copy()
    for key, feats in subsets.items():
        clf = make_clf()
        clf.fit(df[feats].values, y)
        probs = clf.predict_proba(df[feats].values)[:, 1]
        out[f"{key}_post_prob"] = probs
        out[f"{key}_pred"] = (probs >= 0.5).astype(int)

    prob_cols = [f"{key}_post_prob" for key in subsets]
    out["prob_spread"] = out[prob_cols].max(axis=1) - out[prob_cols].min(axis=1)
    out["uncertainty_min_margin"] = out[prob_cols].apply(lambda row: min(abs(v - 0.5) for v in row), axis=1)
    out["label_disagreement"] = (
        (out["k=2_pred"] != out["k=4_pred"])
        | (out["k=2_pred"] != out["k=6_pred"])
        | (out["k=4_pred"] != out["k=6_pred"])
    )

    ranked = out.sort_values(["label_disagreement", "prob_spread", "uncertainty_min_margin"], ascending=[False, False, True])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output, index=False)

    print(f"Saved ranked candidates to {args.output}")
    print(ranked.groupby('Label').head(5).to_string())


if __name__ == "__main__":
    main()
