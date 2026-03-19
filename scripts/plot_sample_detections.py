#!/usr/bin/env python3
"""
Generate sample-level detection plots for pre/post-defoliation at k=2, 4, and 6.

The plots show held-out sample predictions for the transferred SPIE-augmented
feature table and the currently selected QFS subsets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_INPUT = ROOT / "icml_features_spie_augmented.csv"
DEFAULT_SUBSETS = ROOT / "results" / "spie_transfer" / "spie_transfer_selected_subsets.json"
DEFAULT_OUTDIR = ROOT / "results" / "spie_transfer_eval" / "sample_detection_plots"


def make_clf() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
        ]
    )


def load_valid_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = (df["Label"] == "Post_Defoliation").astype(int).values
    groups = df["Folder"].values
    splitter = GroupShuffleSplit(n_splits=12, test_size=0.30, random_state=42)
    for train_idx, test_idx in splitter.split(df, y, groups=groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        return train_idx, test_idx
    raise RuntimeError("Could not find a valid held-out split with both classes present.")


def build_prediction_frame(
    df: pd.DataFrame,
    subset_name: str,
    feature_names: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    y = (df["Label"] == "Post_Defoliation").astype(int).values
    clf = make_clf()
    clf.fit(df.iloc[train_idx][feature_names].values, y[train_idx])

    probs = clf.predict_proba(df.iloc[test_idx][feature_names].values)[:, 1]
    pred = (probs >= 0.5).astype(int)

    frame = df.iloc[test_idx][["Filename", "Folder", "Label"]].copy()
    frame["subset"] = subset_name
    frame["predicted_post_probability"] = probs
    frame["predicted_label"] = np.where(pred == 1, "Post_Defoliation", "Pre_Defoliation")
    frame["correct"] = (pred == y[test_idx]).astype(int)
    frame["sample_order"] = 0

    parts = []
    for label_name in ["Pre_Defoliation", "Post_Defoliation"]:
        part = frame[frame["Label"] == label_name].copy()
        part = part.sort_values("predicted_post_probability").reset_index(drop=True)
        part["sample_order"] = np.arange(len(part))
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def plot_subset_detection(ax: plt.Axes, pred_df: pd.DataFrame, subset_name: str, feature_names: list[str]) -> None:
    palette = {
        "Pre_Defoliation": "#2E7D32",
        "Post_Defoliation": "#8D6E63",
    }

    for label_name in ["Pre_Defoliation", "Post_Defoliation"]:
        part = pred_df[pred_df["Label"] == label_name].copy()
        x = part["sample_order"].values
        y = part["predicted_post_probability"].values
        correct = part["correct"].values.astype(bool)

        ax.scatter(
            x[correct],
            y[correct],
            s=28,
            alpha=0.80,
            color=palette[label_name],
            edgecolors="white",
            linewidth=0.4,
            label=f"True {label_name.replace('_', ' ')}",
        )
        ax.scatter(
            x[~correct],
            y[~correct],
            s=62,
            alpha=0.95,
            facecolors="none",
            edgecolors="#C62828",
            linewidth=1.4,
            label="Misclassified" if label_name == "Pre_Defoliation" else None,
        )

    acc = accuracy_score(
        (pred_df["Label"] == "Post_Defoliation").astype(int).values,
        (pred_df["predicted_label"] == "Post_Defoliation").astype(int).values,
    )
    features_text = ", ".join(feature_names)

    ax.axhline(0.5, color="#37474F", linestyle="--", linewidth=1.2)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Held-out samples within class")
    ax.set_ylabel("Predicted P(Post_Defoliation)")
    ax.set_title(f"{subset_name}  |  Accuracy={acc:.3f}", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.01,
        1.02,
        features_text,
        transform=ax.transAxes,
        fontsize=9,
        color="#455A64",
        va="bottom",
    )


def save_plots(all_predictions: dict[str, pd.DataFrame], subsets: dict[str, list[str]], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    ordered = [("k=2", "QFS k=2"), ("k=4", "QFS k=4"), ("k=6", "QFS k=6")]

    fig, axes = plt.subplots(3, 1, figsize=(13, 14), facecolor="white")
    legend_handles = None
    legend_labels = None

    for ax, (key, title) in zip(axes, ordered):
        plot_subset_detection(ax, all_predictions[key], title, subsets[key])
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        "Per-Sample Detection on Held-Out UAV Images\nPre- vs Post-Defoliation for QFS k=2, 4, 6",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    combined_path = outdir / "sample_detection_k2_k4_k6.png"
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    for key, title in ordered:
        fig, ax = plt.subplots(figsize=(12, 4.8), facecolor="white")
        plot_subset_detection(ax, all_predictions[key], title, subsets[key])
        handles, labels = ax.get_legend_handles_labels()
        dedup = {}
        for handle, label in zip(handles, labels):
            if label and label not in dedup:
                dedup[label] = handle
        ax.legend(dedup.values(), dedup.keys(), loc="best", frameon=False)
        plt.tight_layout()
        out_path = outdir / f"sample_detection_{key.replace('=', '')}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def save_prediction_csv(all_predictions: dict[str, pd.DataFrame], outdir: Path) -> None:
    rows = []
    for key, frame in all_predictions.items():
        rows.append(frame.copy())
    pd.concat(rows, ignore_index=True).to_csv(outdir / "sample_detection_predictions.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--subsets-json", type=Path, default=DEFAULT_SUBSETS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    df = pd.read_csv(args.input).dropna().reset_index(drop=True)
    subsets = json.loads(args.subsets_json.read_text())
    train_idx, test_idx = load_valid_split(df)

    all_predictions = {
        key: build_prediction_frame(df, key, feature_names, train_idx, test_idx)
        for key, feature_names in subsets.items()
    }

    save_plots(all_predictions, subsets, args.outdir)
    save_prediction_csv(all_predictions, args.outdir)

    print(f"Saved sample-level detection plots to {args.outdir}")
    print(f"Saved sample predictions to {args.outdir / 'sample_detection_predictions.csv'}")


if __name__ == "__main__":
    main()
