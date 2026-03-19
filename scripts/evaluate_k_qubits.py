#!/usr/bin/env python3
"""
evaluate_k_qubits.py
====================
DEFINITIVE evaluation of QFS vs MI subsets at k=2, 4, 6.

Why 1.0 is NOT wrong for clean data:
  Mean_ExG alone separates the classes perfectly (biological fact: green leaves
  vs brown stalks). Any subset containing Std_ExG/Mean_ExG will hit 1.0 on
  clean data. The interesting story is what happens under UAV sensor noise.

This script does THREE things:
  1) GROUP-FOLDER cross-validation (no leakage) on clean data  → shows 1.0 is real
  2) Noise-degraded accuracy curve (σ = 0 … 0.30)              → shows scaling advantage
  3) Augmentation stress test (fog / glare / shadow)            → realistic field conditions

All numbers are written to results/tables/ and all plots to results/plots/.
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, auc,
                             precision_recall_curve, average_precision_score)

# ─── Paths ───────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(HERE)           # /Volumes/T9/QuantumFeatureSelection
CSV_PATH    = os.path.join(ROOT, "icml_features_FULL.csv")
ACTIVE_INPUT_PATH = CSV_PATH
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")
LOGS_DIR    = os.path.join(RESULTS_DIR, "logs")

# ─── Subset definitions (same as in paper) ───────────────────────────────────
DEFAULT_SUBSETS = {
    "QFS_k2": ["Std_ExG", "Mean_RBR"],
    "MI_k2":  ["Std_ExG", "Mean_ExG"],
    "QFS_k4": ["Std_ExG", "Mean_RBR", "Mean_B", "Correlation"],
    "MI_k4":  ["Std_ExG", "Mean_ExG", "Mean_RBR", "Mean_B"],
    "QFS_k6": ["Std_ExG", "Mean_RBR", "Mean_B", "Correlation", "Mean_ExG", "Mean_NGRDI"],
    "MI_k6":  ["Std_ExG", "Mean_ExG", "Mean_RBR", "Mean_B", "Correlation", "Mean_NGRDI"],
}

SUBSETS = dict(DEFAULT_SUBSETS)

COLORS = {
    "QFS_k2": "#C62828",  # dark red
    "MI_k2":  "#EF9A9A",  # light red
    "QFS_k4": "#1565C0",  # dark blue
    "MI_k4":  "#90CAF9",  # light blue
    "QFS_k6": "#2E7D32",  # dark green
    "MI_k6":  "#A5D6A7",  # light green
}

MARKERS = {"QFS_k2":"o","MI_k2":"x","QFS_k4":"s","MI_k4":"^","QFS_k6":"D","MI_k6":"v"}

# ─── Augmentation helpers ─────────────────────────────────────────────────────
def fog(X):
    """Simulate fog: compress contrast toward mean, add small noise."""
    rng = np.random.RandomState(0)
    mu = X.mean(axis=0)
    return 0.65 * X + 0.35 * mu + rng.normal(0, 0.03, X.shape)

def glare(X):
    """Simulate glare: moderate brightness shift + mild noise."""
    rng = np.random.RandomState(1)
    out = X.copy()
    out += rng.normal(0, 0.08, X.shape)   # mild Gaussian distortion
    # Shift spectral features upward slightly (sensor saturation)
    out[:, :2] = out[:, :2] * 1.12
    return out

def shadow(X):
    """Simulate uneven shadow: darken some rows."""
    rng = np.random.RandomState(2)
    out = X.copy()
    mask = rng.rand(len(X)) < 0.4
    out[mask] *= 0.60
    return out

AUG_FNS = {"Fog": fog, "Glare": glare, "Shadow": shadow}

def noise(X, sigma):
    rng = np.random.RandomState(42)
    return X + rng.normal(0, sigma, X.shape)


# ─── Core evaluation ─────────────────────────────────────────────────────────
def make_clf():
    return Pipeline([
        ("sc",  StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                    probability=True, random_state=42)),
    ])


def evaluate_clean_cv(df, y, groups):
    """5-fold group cross-validation on clean data."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    records = []
    for name, feats in SUBSETS.items():
        X = df[feats].values
        accs, f1s, aucs = [], [], []
        for tr, te in sgkf.split(X, y, groups):
            clf = make_clf()
            clf.fit(X[tr], y[tr])
            p      = clf.predict(X[te])
            prob   = clf.predict_proba(X[te])[:, 1]
            accs.append(accuracy_score(y[te], p))
            # Guard: f1 and AUC need both classes present in test fold
            if len(np.unique(y[te])) > 1:
                f1s.append(f1_score(y[te], p, zero_division=0))
                try:
                    aucs.append(roc_auc_score(y[te], prob))
                except Exception:
                    pass
        records.append(dict(Method=name, k=int(name.split("k")[1]),
                            Accuracy=np.mean(accs)        if accs else float("nan"),
                            Accuracy_std=np.std(accs)     if accs else float("nan"),
                            F1=np.mean(f1s)               if f1s  else float("nan"),
                            F1_std=np.std(f1s)            if f1s  else float("nan"),
                            AUC=np.mean(aucs)             if aucs else float("nan"),
                            AUC_std=np.std(aucs)          if aucs else float("nan")))
    return pd.DataFrame(records)


def evaluate_noise_curve(df, y, groups):
    """Accuracy vs. Gaussian noise sigma on held-out test fold."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    # Use one fixed fold for the noise curve
    tr, te = next(iter(sgkf.split(df.values, y, groups)))

    sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    records = []
    for name, feats in SUBSETS.items():
        X = df[feats].values
        clf = make_clf()
        clf.fit(X[tr], y[tr])
        for sig in sigmas:
            Xte = noise(X[te], sig)
            acc = accuracy_score(y[te], clf.predict(Xte))
            records.append(dict(Method=name, k=int(name.split("k")[1]),
                                Sigma=sig, Accuracy=acc))
    return pd.DataFrame(records)


def evaluate_augmentation(df, y, groups):
    """Accuracy under field-realistic augmentations."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr, te = next(iter(sgkf.split(df.values, y, groups)))

    records = []
    for name, feats in SUBSETS.items():
        X = df[feats].values
        clf = make_clf()
        clf.fit(X[tr], y[tr])
        row = dict(Method=name, k=int(name.split("k")[1]),
                   Clean=accuracy_score(y[te], clf.predict(X[te])))
        for aug_name, aug_fn in AUG_FNS.items():
            Xte_aug = aug_fn(X[te].copy())
            row[aug_name] = accuracy_score(y[te], clf.predict(Xte_aug))
        records.append(row)
    return pd.DataFrame(records)


# ─── ROC / PR per k ──────────────────────────────────────────────────────────
def evaluate_roc_pr(df, y, groups):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr, te = next(iter(sgkf.split(df.values, y, groups)))
    curves = {}
    for name, feats in SUBSETS.items():
        X = df[feats].values
        clf = make_clf()
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        fpr, tpr, _ = roc_curve(y[te], prob)
        p, r, _     = precision_recall_curve(y[te], prob)
        curves[name] = dict(fpr=fpr, tpr=tpr, roc_auc=auc(fpr, tpr),
                            precision=p, recall=r,
                            ap=average_precision_score(y[te], prob))
    return curves, y[te]


# ─── Plotting ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

def plot_noise_curve(noise_df):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    for name in SUBSETS:
        sub = noise_df[noise_df.Method == name]
        ls  = "-" if "QFS" in name else "--"
        lw  = 2.5 if "QFS" in name else 1.8
        k   = name.split("k")[1]
        label = f"{'Quantum' if 'QFS' in name else 'Classical'} MI k={k}"
        ax.plot(sub.Sigma, sub.Accuracy * 100,
                color=COLORS[name], linestyle=ls, linewidth=lw,
                marker=MARKERS[name], markersize=7, label=label)

    ax.axvspan(0.18, 0.30, color="#FFF3E0", alpha=0.55, zorder=0)
    ax.text(0.24, 47, "Extreme Noise\nZone", ha="center", fontsize=10,
            color="#BF360C", fontweight="bold")
    ax.set_xlabel(r"Gaussian Noise $\sigma$ (applied to test features)")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_title("Noise Robustness: QFS vs MI Across k ∈ {2, 4, 6}\n"
                 "(Strict Group-Fold Evaluation, No Leakage)",
                 fontweight="bold")
    ax.set_ylim(35, 105)
    ax.legend(ncol=2, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "k_noise_robustness.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_augmentation_grouped(aug_df):
    conditions = ["Clean", "Fog", "Glare", "Shadow"]
    k_vals     = [2, 4, 6]
    fig, axes  = plt.subplots(1, 3, figsize=(15, 5), sharey=True, facecolor="white")

    for ax, k in zip(axes, k_vals):
        methods  = [f"QFS_k{k}", f"MI_k{k}"]
        labels   = [f"QFS k={k}", f"MI k={k}"]
        palettes = [COLORS[f"QFS_k{k}"], COLORS[f"MI_k{k}"]]
        x = np.arange(len(conditions))
        w = 0.35
        for j, (m, lbl, col) in enumerate(zip(methods, labels, palettes)):
            row   = aug_df[aug_df.Method == m].iloc[0]
            vals  = [row[c] * 100 for c in conditions]
            bars  = ax.bar(x + (j - 0.5) * w, vals, w, label=lbl,
                           color=col, edgecolor="black", linewidth=0.8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=8.5)

        ax.set_title(f"k = {k} Qubits", fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(conditions)
        ax.set_ylim(30, 110)
        ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines[["top","right"]].set_visible(False)

    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Augmentation Robustness by Qubit Subset Size (k=2, 4, 6)\n"
                 "Fog / Glare / Shadow — Simulating Real UAV Field Conditions",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "k_augmentation_stress.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_roc_pr_grid(curves):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="white")
    k_vals = [2, 4, 6]

    for col, k in enumerate(k_vals):
        ax_roc = axes[0, col]
        ax_pr  = axes[1, col]

        for method in ["QFS", "MI"]:
            name = f"{method}_k{k}"
            c    = curves[name]
            col_ = COLORS[name]
            ls   = "-" if method == "QFS" else "--"
            lbl  = f"{'Quantum' if method=='QFS' else 'MI'} (AUC={c['roc_auc']:.3f})"
            ax_roc.plot(c["fpr"], c["tpr"], color=col_, lw=2.2, ls=ls, label=lbl)
            lbl2 = f"{'Quantum' if method=='QFS' else 'MI'} (AP={c['ap']:.3f})"
            ax_pr.plot(c["recall"], c["precision"], color=col_, lw=2.2, ls=ls, label=lbl2)

        ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, ls=":")
        ax_roc.set(title=f"ROC Curve — k={k}", xlabel="FPR", ylabel="TPR")
        ax_roc.legend(loc="lower right"); ax_roc.grid(ls="--", alpha=0.4)
        ax_roc.spines[["top","right"]].set_visible(False)

        ax_pr.set(title=f"Precision-Recall — k={k}", xlabel="Recall", ylabel="Precision")
        ax_pr.legend(loc="lower left"); ax_pr.grid(ls="--", alpha=0.4)
        ax_pr.spines[["top","right"]].set_visible(False)

    fig.suptitle("ROC and Precision-Recall Profiles: Quantum vs Classical\n"
                 "Across Increasing Qubit Dimensionality (k ∈ {2, 4, 6})",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "k_roc_pr_grid.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_cv_boxplot(cv_df):
    """Clean-data 5-fold CV results as grouped bars with std error bars."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    k_grps = [2, 4, 6]
    x = np.arange(len(k_grps))
    w = 0.35
    for j, method in enumerate(["QFS", "MI"]):
        accs = [cv_df[cv_df.Method == f"{method}_k{k}"].Accuracy.values[0] * 100
                for k in k_grps]
        stds = [cv_df[cv_df.Method == f"{method}_k{k}"].Accuracy_std.values[0] * 100
                for k in k_grps]
        colors_bar = [COLORS[f"{method}_k{k}"] for k in k_grps]
        bars = ax.bar(x + (j - 0.5) * w, accs, w,
                      color=colors_bar,
                      edgecolor="black", linewidth=0.9,
                      label=f"{'Quantum VQC' if method=='QFS' else 'Classical MI'}",
                      yerr=stds, capsize=5, error_kw=dict(lw=1.5))
        for bar, v, s in zip(bars, accs, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.2,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in k_grps], fontsize=12)
    ax.set_ylim(85, 103)
    ax.set_ylabel("5-Fold CV Accuracy (%) ± std")
    ax.set_title("Clean-Data Cross-Validation: Quantum vs Classical Across k\n"
                 "5-Fold Stratified Group-CV, No Spatial Leakage",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.5)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "k_cv_accuracy_bars.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_radar(noise_df, sigma=0.15):
    """Spider / Radar chart at one noise level showing all 5 metrics."""
    from sklearn.metrics import precision_score as prc_sc, recall_score as rec_sc

    df_full = pd.read_csv(ACTIVE_INPUT_PATH).dropna()
    groups  = df_full["Folder"].values
    y       = (df_full["Label"] == "Post_Defoliation").astype(int).values
    sgkf    = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr, te  = next(iter(sgkf.split(df_full.values, y, groups)))

    metric_names = ["Accuracy", "F1", "Precision", "Recall", "AUC"]
    num_v = len(metric_names)
    angles = np.linspace(0, 2*np.pi, num_v, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor="white")

    for name, feats in SUBSETS.items():
        X = df_full[feats].values
        clf = make_clf()
        clf.fit(X[tr], y[tr])
        Xte = noise(X[te], sigma)
        p = clf.predict(Xte)
        prob = clf.predict_proba(Xte)[:, 1]
        vals = [
            accuracy_score(y[te], p),
            f1_score(y[te], p),
            prc_sc(y[te], p),
            rec_sc(y[te], p),
            roc_auc_score(y[te], prob),
        ]
        vals += vals[:1]
        ls = "-" if "QFS" in name else ":"
        lw = 2.5 if "QFS" in name else 1.6
        k  = name.split("k")[1]
        lbl = f"{'QFS' if 'QFS' in name else 'MI'} k={k}"
        ax.plot(angles, vals, color=COLORS[name], lw=lw, ls=ls, label=lbl)
        ax.fill(angles, vals, color=COLORS[name], alpha=0.08)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_yticklabels([]); ax.set_ylim(0, 1)
    ax.set_title(f"Multi-Metric Radar Chart at σ={sigma}\n"
                 "(All k=2/4/6 Subsets under Gaussian Sensor Noise)",
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15))
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, f"k_radar_sigma{int(sigma*100)}.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_feature_heatmap():
    """Binary heatmap: which features appear in each subset."""
    all_feats = sorted({f for feats in SUBSETS.values() for f in feats})
    data = []
    row_labels = []
    for name, feats in SUBSETS.items():
        row = [1 if f in feats else 0 for f in all_feats]
        data.append(row)
        k     = name.split("k")[1]
        label = f"{'QFS' if 'QFS' in name else 'MI '} k={k}"
        row_labels.append(label)

    arr = np.array(data, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
    im = ax.imshow(arr, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(all_feats)))
    ax.set_xticklabels(all_feats, rotation=35, ha="right", fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)

    for i in range(len(row_labels)):
        for j in range(len(all_feats)):
            sym = "✓" if arr[i, j] else ""
            ax.text(j, i, sym, ha="center", va="center",
                    fontsize=14, color="white" if arr[i, j] else "lightgray")

    # Separate QFS from MI visually
    ax.axhline(0.5, color="white", lw=3)
    ax.axhline(2.5, color="white", lw=3)
    ax.axhline(4.5, color="white", lw=3)

    ax.set_title("Feature Membership per Qubit Subset (QFS vs MI, k=2/4/6)",
                 fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "k_feature_membership.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def save_tables(cv_df, noise_df, aug_df):
    cv_df.to_csv(os.path.join(TABLES_DIR, "clean_cv_results.csv"), index=False)
    noise_df.to_csv(os.path.join(TABLES_DIR, "noise_curve_results.csv"), index=False)
    aug_df.to_csv(os.path.join(TABLES_DIR, "augmentation_results.csv"), index=False)

    # Pretty LaTeX-style text table
    lines = ["=" * 70,
             "  TABLE 1: Clean Data — 5-Fold Group-CV (No Spatial Leakage)",
             "=" * 70,
             f"{'Method':<12} {'k':>3} {'Accuracy':>10} {'F1':>10} {'AUC':>10}",
             "-" * 50]
    for _, r in cv_df.iterrows():
        lines.append(f"{r.Method:<12} {int(r.k):>3} "
                     f"{r.Accuracy:>9.4f} {r.F1:>9.4f} {r.AUC:>9.4f}")
    lines += ["", "=" * 70,
              "  TABLE 2: Augmentation Stress Test (field UAV conditions)",
              "=" * 70,
              f"{'Method':<12} {'k':>3} {'Clean':>8} {'Fog':>8} {'Glare':>8} {'Shadow':>8}",
              "-" * 55]
    for _, r in aug_df.iterrows():
        lines.append(f"{r.Method:<12} {int(r.k):>3} "
                     f"{r.Clean:>7.4f} {r.Fog:>7.4f} {r.Glare:>7.4f} {r.Shadow:>7.4f}")

    tbl = "\n".join(lines)
    print("\n" + tbl)
    with open(os.path.join(TABLES_DIR, "summary_tables.txt"), "w") as f:
        f.write(tbl)
    print(f"\n  Saved tables to {TABLES_DIR}/")


def configure_output_dirs(out_subdir: str | None):
    global PLOTS_DIR, TABLES_DIR, LOGS_DIR

    if out_subdir:
        PLOTS_DIR = os.path.join(RESULTS_DIR, out_subdir, "plots")
        TABLES_DIR = os.path.join(RESULTS_DIR, out_subdir, "tables")
        LOGS_DIR = os.path.join(RESULTS_DIR, out_subdir, "logs")

    for directory in [PLOTS_DIR, TABLES_DIR, LOGS_DIR]:
        os.makedirs(directory, exist_ok=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("Label", "Folder", "Filename")]


def top_mi_features(df: pd.DataFrame, y: np.ndarray, k: int) -> list[str]:
    cols = feature_columns(df)
    scores = mutual_info_classif(df[cols].values, y, random_state=42)
    ranked = [name for name, _ in sorted(zip(cols, scores), key=lambda item: item[1], reverse=True)]
    return ranked[:k]


def load_qfs_subsets(subsets_json_path: str) -> dict[str, list[str]]:
    with open(subsets_json_path) as handle:
        raw = json.load(handle)
    return {f"QFS_k{k}": raw[f"k={k}"] for k in [2, 4, 6]}


def configure_subsets(df: pd.DataFrame, y: np.ndarray, subsets_json_path: str | None):
    global SUBSETS
    if not subsets_json_path:
        SUBSETS = dict(DEFAULT_SUBSETS)
        return

    qfs_subsets = load_qfs_subsets(subsets_json_path)
    mi_subsets = {
        f"MI_k{k}": top_mi_features(df, y, k)
        for k in [2, 4, 6]
    }
    SUBSETS = {**qfs_subsets, **mi_subsets}


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    global ACTIVE_INPUT_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=CSV_PATH, help="Feature CSV to evaluate.")
    parser.add_argument("--qfs-subsets-json", default=None, help="Optional JSON with keys k=2/k=4/k=6 for transferred QFS subsets.")
    parser.add_argument("--out-subdir", default=None, help="Optional results subdirectory under results/.")
    args = parser.parse_args()

    configure_output_dirs(args.out_subdir)
    ACTIVE_INPUT_PATH = args.input

    print("Loading dataset …")
    df  = pd.read_csv(args.input).dropna()
    y   = (df["Label"] == "Post_Defoliation").astype(int).values
    grp = df["Folder"].values

    configure_subsets(df, y, args.qfs_subsets_json)

    feat_cols = feature_columns(df)
    X_df      = df[feat_cols]

    print(f"  {len(df)} samples | "
          f"Post={y.sum()} Pre={(1-y).sum()} | "
          f"{len(df.Folder.unique())} folders")

    print("\n[1/5] Clean 5-Fold Group-CV …")
    cv_df  = evaluate_clean_cv(X_df, y, grp)

    print("[2/5] Noise robustness curve …")
    noise_df = evaluate_noise_curve(X_df, y, grp)

    print("[3/5] Augmentation stress test …")
    aug_df = evaluate_augmentation(X_df, y, grp)

    print("[4/5] ROC / PR curves …")
    curves, y_te = evaluate_roc_pr(X_df, y, grp)

    print("[5/5] Generating all plots …")
    plot_noise_curve(noise_df)
    plot_augmentation_grouped(aug_df)
    plot_roc_pr_grid(curves)
    plot_cv_boxplot(cv_df)
    plot_radar(noise_df, sigma=0.15)
    save_tables(cv_df, noise_df, aug_df)

    print("[6/5] Feature membership heatmap …")
    plot_feature_heatmap()

    print("\n✅  All done!  Results saved to:")
    print(f"    Plots  → {PLOTS_DIR}")
    print(f"    Tables → {TABLES_DIR}")


if __name__ == "__main__":
    main()
