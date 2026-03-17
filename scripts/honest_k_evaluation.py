#!/usr/bin/env python3
"""
honest_k_evaluation.py
=======================
CORRECTED, HONEST evaluation of feature subset scaling at k=2, 4, 6.

ROOT CAUSE OF PREVIOUS 100% BUG
================================
1. Only 6 folders exist; 4 are pure single-class. GroupKFold creates folds
   where the test set contains ONLY one class → trivial prediction → 100%.

2. Both Std_ExG and Mean_ExG independently achieve AUC=1.0. Every subset
   previously used contained at least one of these → trivially 100%.

CORRECTED DESIGN
================
- Use a stratified image-level 70/30 train/test split (ensures both classes
  in every evaluation set).
- Define *honest* subsets that progressively add harder features (GLCM
  texture + non-trivial spectral indices ONLY).
- The ExG group is isolated and reported separately as "trivial baseline".
- Augmentation (Fog, Glare, Shadow) and Gaussian noise stress tests show
  the realistic performance gap between k=2 vs k=4 vs k=6.
"""

import warnings; warnings.filterwarnings("ignore")

import os, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedKFold, train_test_split)
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score, confusion_matrix)

HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(HERE)
CSV_PATH    = os.path.join(ROOT, "icml_features_FULL.csv")
PLOTS_DIR   = os.path.join(ROOT, "results", "plots")
TABLES_DIR  = os.path.join(ROOT, "results", "tables")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# ─── Honest subset definitions ────────────────────────────────────────────────
# These subsets EXCLUDE the trivially‑perfect ExG features.
# We build genuine complexity: k=2 (2 hardest non‑ExG), k=4 (adds GLCM),
# k=6 (adds more GLCM). Then QFS is shown as a separate "ExG‑aware" track.

HONEST_SUBSETS = {
    # ── Pure non-ExG scaling (the interesting story) ─────────────────────────
    "NonExG_k2":  ["Mean_RBR", "Mean_B"],
    "NonExG_k4":  ["Mean_RBR", "Mean_B", "Correlation", "Mean_NGRDI"],
    "NonExG_k6":  ["Mean_RBR", "Mean_B", "Correlation", "Mean_NGRDI",
                   "Contrast", "Homogeneity"],
    # ── QFS track: VQC-selected features (includes ExG as anchor) ────────────
    "QFS_k2":     ["Std_ExG", "Mean_RBR"],
    "QFS_k4":     ["Std_ExG", "Mean_RBR", "Mean_B", "Correlation"],
    "QFS_k6":     ["Std_ExG", "Mean_RBR", "Mean_B", "Correlation",
                   "Mean_ExG", "Mean_NGRDI"],
    # ── Classical MI baselines ────────────────────────────────────────────────
    "MI_k2":      ["Std_ExG", "Mean_ExG"],
    "MI_k4":      ["Std_ExG", "Mean_ExG", "Mean_RBR", "Mean_B"],
    "MI_k6":      ["Std_ExG", "Mean_ExG", "Mean_RBR", "Mean_B",
                   "Correlation", "Mean_NGRDI"],
    # ── Trivial baselines (ExG alone — documented as known-trivial) ───────────
    "ExG_trivial_k1": ["Mean_ExG"],
    "ExG_trivial_k2": ["Std_ExG", "Mean_ExG"],
}

COLORS = {
    "NonExG_k2":  "#BF360C",
    "NonExG_k4":  "#E64A19",
    "NonExG_k6":  "#FF8A65",
    "QFS_k2":     "#1A237E",
    "QFS_k4":     "#1565C0",
    "QFS_k6":     "#42A5F5",
    "MI_k2":      "#1B5E20",
    "MI_k4":      "#2E7D32",
    "MI_k6":      "#66BB6A",
    "ExG_trivial_k1": "#4A148C",
    "ExG_trivial_k2": "#7B1FA2",
}
MARKERS = {"NonExG_k2":"o","NonExG_k4":"s","NonExG_k6":"D",
           "QFS_k2":"^","QFS_k4":"*","QFS_k6":"P",
           "MI_k2":"X","MI_k4":"v","MI_k6":"<",
           "ExG_trivial_k1":"H","ExG_trivial_k2":"h"}

def make_clf():
    return Pipeline([
        ("sc",  StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                    probability=True, random_state=42)),
    ])

# ─── Data loading ─────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(CSV_PATH).dropna()
y  = (df["Label"] == "Post_Defoliation").astype(int).values

# Stratified 70/30 → both classes guaranteed in train AND test
np.random.seed(42)
tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.30,
                                   stratify=y, random_state=42)
print(f"  Total={len(df)} | Train={len(tr_idx)} Test={len(te_idx)}")
print(f"  Test: Post={y[te_idx].sum()} Pre={(1-y[te_idx]).sum()}")

# 5-fold CV on TRAINING set (for bar charts / stability)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─── Augmentation ─────────────────────────────────────────────────────────────
def aug_fog(X):
    rng = np.random.RandomState(0)
    return 0.65 * X + 0.35 * X.mean(axis=0) + rng.normal(0, 0.03, X.shape)

def aug_glare(X):
    rng = np.random.RandomState(1)
    out = X.copy() + rng.normal(0, 0.08, X.shape)
    out[:, :2] *= 1.12
    return out

def aug_shadow(X):
    rng = np.random.RandomState(2)
    out = X.copy()
    mask = rng.rand(len(X)) < 0.4
    out[mask] *= 0.60
    return out

def add_noise(X, sigma):
    return X + np.random.RandomState(42).normal(0, sigma, X.shape)

AUGS = {"Fog": aug_fog, "Glare": aug_glare, "Shadow": aug_shadow}

# ─── Core compute ─────────────────────────────────────────────────────────────

def compute_all():
    results = {}
    for name, feats in HONEST_SUBSETS.items():
        X = df[feats].values
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Train on full train fold
        clf_full = make_clf()
        clf_full.fit(X_tr, y_tr)

        # Clean test metrics
        pred_clean = clf_full.predict(X_te)
        prob_clean = clf_full.predict_proba(X_te)[:, 1]

        row = {
            "Method":   name,
            "k":        len(feats),
            "Features": feats,
            "Acc_clean":   accuracy_score(y_te, pred_clean),
            "F1_clean":    f1_score(y_te, pred_clean),
            "AUC_clean":   roc_auc_score(y_te, prob_clean),
        }

        # Augmentation
        for aug_name, aug_fn in AUGS.items():
            X_aug = aug_fn(X_te.copy())
            pred_aug = clf_full.predict(X_aug)
            row[f"Acc_{aug_name}"] = accuracy_score(y_te, pred_aug)

        # Noise curve
        sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        row["noise_accs"] = []
        for sig in sigmas:
            X_noise = add_noise(X_te.copy(), sig)
            row["noise_accs"].append(accuracy_score(y_te, clf_full.predict(X_noise)))
        row["sigmas"] = sigmas

        # 5-fold CV on train set
        cv_accs = []
        for cv_tr, cv_te in cv.split(X_tr, y_tr):
            clf_cv = make_clf()
            clf_cv.fit(X_tr[cv_tr], y_tr[cv_tr])
            cv_accs.append(accuracy_score(y_tr[cv_te], clf_cv.predict(X_tr[cv_te])))
        row["CV_mean"] = np.mean(cv_accs)
        row["CV_std"]  = np.std(cv_accs)

        # ROC / PR
        fpr, tpr, _ = roc_curve(y_te, prob_clean)
        p, r, _     = precision_recall_curve(y_te, prob_clean)
        row["fpr"] = fpr; row["tpr"] = tpr
        row["prec"] = p;  row["rec"] = r
        row["AP"] = average_precision_score(y_te, prob_clean)

        results[name] = row
        print(f"  {name:20s}: Acc={row['Acc_clean']:.4f}  "
              f"AUC={row['AUC_clean']:.4f}  CV={row['CV_mean']:.4f}±{row['CV_std']:.4f}")

    return results

# ─── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9.5, "figure.dpi": 150,
})
SPINE_OFF = lambda ax: [ax.spines[s].set_visible(False) for s in ["top","right"]]

# ─── PLOT 1: Honest k-scaling bar chart (Clean CV vs Noise) ────────────────────
def plot1_cv_bars(res):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True, facecolor="white")
    tracks = {
        "Non-Trivial Scaling\n(No ExG)":  ["NonExG_k2","NonExG_k4","NonExG_k6"],
        "QFS Quantum-Selected":          ["QFS_k2","QFS_k4","QFS_k6"],
        "Classical MI Baselines":        ["MI_k2","MI_k4","MI_k6"],
    }
    
    # We will plot Clean CV vs Noise (sigma=0.15, which is index 3 in sigmas)
    SIGMA_IDX = 3
    
    for ax, (title, names) in zip(axes, tracks.items()):
        ks     = np.array([res[n]["k"] for n in names])
        means  = np.array([res[n]["CV_mean"]*100 for n in names])
        stds   = np.array([res[n]["CV_std"]*100 for n in names])
        noises = np.array([res[n]["noise_accs"][SIGMA_IDX]*100 for n in names])
        colors = [COLORS[n] for n in names]
        
        w = 0.4
        # Clean CV Bar
        b1 = ax.bar(ks - w/2, means, color=colors, edgecolor="black",
                    linewidth=0.9, width=w, yerr=stds, capsize=4, error_kw=dict(lw=1.2),
                    label="Clean CV")
        # Noise-Stressed Bar
        b2 = ax.bar(ks + w/2, noises, color=colors, edgecolor="black",
                    linewidth=0.9, width=w, alpha=0.45, hatch="///",
                    label="Noise (σ=0.15)")
                    
        # Annotate Clean CV
        for bar, m in zip(b1, means):
            if m > 99.0: # If it's trivial 100%, put the text slightly inside
                ax.text(bar.get_x() + bar.get_width()/2, 101, "100%", 
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#B71C1C")
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1.5, f"{m:.1f}%", 
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")
                        
        # Annotate Noise accuracy
        for bar, n in zip(b2, noises):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, f"{n:.1f}%", 
                    ha="center", va="bottom", fontsize=8.5, color="#424242")
                    
        ax.set_xticks(ks); ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(60, 108)
        if title.startswith("Non-Trivial"):
            ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", ls="--", alpha=0.5); SPINE_OFF(ax)
        
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Scaling Robustness: Clean CV vs. Noise-Stressed Evaluating (σ=0.15)\n"
                 "Breaks the 'Trivial 100%' illusion by showing true degradation",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "k_comparison", "honest_01_cv_bars.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

# ─── PLOT 2: Noise robustness – non-trivial subsets only ─────────────────────
def plot2_noise(res):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    key_sets = ["NonExG_k2","NonExG_k4","NonExG_k6",
                "QFS_k2","QFS_k4","QFS_k6",
                "MI_k2","MI_k4","MI_k6"]
    sigmas = res["NonExG_k2"]["sigmas"]

    for name in key_sets:
        r  = res[name]
        ls = "-" if "NonExG" in name else ("--" if "QFS" in name else ":")
        lw = 2.5 if name.endswith("k6") else (2.0 if name.endswith("k4") else 1.5)
        k  = name.split("k")[1]
        track = ("Non-Trivial" if "NonExG" in name else
                 ("QFS" if "QFS" in name else "MI"))
        ax.plot(sigmas, [a*100 for a in r["noise_accs"]],
                color=COLORS[name], ls=ls, lw=lw,
                marker=MARKERS[name], ms=7,
                label=f"{track} k={k}")

    ax.axvspan(0.15, 0.30, color="#FFF3E0", alpha=0.4, zorder=0)
    ax.text(0.225, 52, "Severe noise\nzone", ha="center",
            color="#BF360C", fontsize=9.5, fontweight="bold")
    ax.set_xlabel(r"Gaussian noise $\sigma$ applied to test features")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Noise Robustness Across k=2, 4, 6\n"
                 "(Orange = Non-Trivial; Blue = QFS; Green = MI)",
                 fontweight="bold")
    ax.set_ylim(40, 103)
    
    # Legend moved outside the plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=True)
    
    ax.grid(axis="y", ls="--", alpha=0.4); SPINE_OFF(ax)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "honest_02_noise_curve.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

# ─── PLOT 3: Augmentation heat-table ─────────────────────────────────────────
def plot3_aug_heatmap(res):
    key_sets = ["NonExG_k2","NonExG_k4","NonExG_k6",
                "QFS_k2","QFS_k4","QFS_k6",
                "MI_k2","MI_k4","MI_k6"]
    augs = ["Acc_clean","Acc_Fog","Acc_Glare","Acc_Shadow"]
    col_labels = ["Clean","Fog","Glare","Shadow"]
    data = np.array([[res[n][a]*100 for a in augs] for n in key_sets])

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    im = ax.imshow(data, cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticks(range(len(key_sets))); ax.set_yticklabels(key_sets, fontsize=11)
    for i in range(len(key_sets)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{data[i,j]:.1f}",
                    ha="center", va="center", fontsize=10.5, fontweight="bold",
                    color="white" if data[i,j] < 75 else "black")
    for sep in [2.5, 5.5]:
        ax.axhline(sep, color="white", lw=3)
    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    ax.set_title("Augmentation Stress Test: Accuracy (%) per Condition\n"
                 "Red=Fails, Green=Robust — showing honest scaling benefit",
                 fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "honest_03_aug_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

# ─── PLOT 4: ROC grid (non-trivial + QFS + MI per k) ────────────────────────
def plot4_roc_grid(res):
    k_vals = [2, 4, 6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="white")
    for col, k in enumerate(k_vals):
        for ax, curve_type in zip([axes[0, col], axes[1, col]], ["ROC", "PR"]):
            for track, prefix in [("Non-Trivial","NonExG"), ("QFS","QFS"), ("MI","MI")]:
                name = f"{prefix}_k{k}"
                r = res[name]
                if curve_type == "ROC":
                    ax.plot(r["fpr"], r["tpr"], color=COLORS[name], lw=2.2,
                            label=f"{track} (AUC={r['AUC_clean']:.3f})")
                else:
                    ax.plot(r["rec"], r["prec"], color=COLORS[name], lw=2.2,
                            label=f"{track} (AP={r['AP']:.3f})")
            if curve_type == "ROC":
                ax.plot([0,1],[0,1], ":", color="gray", lw=1)
                ax.set(title=f"ROC — k={k}", xlabel="FPR", ylabel="TPR")
                ax.legend(loc="lower right")
            else:
                ax.set(title=f"P-R — k={k}", xlabel="Recall", ylabel="Precision")
                ax.legend(loc="lower left")
            ax.grid(ls="--", alpha=0.4); SPINE_OFF(ax)
    fig.suptitle("ROC and Precision-Recall Curves: Three Tracks Across k ∈ {2, 4, 6}\n"
                 "Stratified 70/30 test split (both classes guaranteed)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "honest_04_roc_pr.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

# ─── PLOT 5: Feature membership map ──────────────────────────────────────────
def plot5_membership(res):
    key_sets = ["NonExG_k2","NonExG_k4","NonExG_k6",
                "QFS_k2","QFS_k4","QFS_k6",
                "MI_k2","MI_k4","MI_k6"]
    all_feats = sorted({f for n in key_sets for f in HONEST_SUBSETS[n]})
    data = np.array([[1 if f in HONEST_SUBSETS[n] else 0
                      for f in all_feats] for n in key_sets], dtype=float)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_feats)))
    ax.set_xticklabels(all_feats, rotation=35, ha="right", fontsize=11)
    ax.set_yticks(range(len(key_sets)))
    ax.set_yticklabels(key_sets, fontsize=11)
    for i in range(len(key_sets)):
        for j in range(len(all_feats)):
            ax.text(j, i, "✓" if data[i,j] else "", ha="center", va="center",
                    fontsize=13, color="white" if data[i,j] else "lightgray")
    for sep in [2.5, 5.5]:
        ax.axhline(sep, color="white", lw=3)
    ax.set_title("Feature Membership Map: Which Features Are In Each Subset",
                 fontweight="bold", fontsize=14, pad=14)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "honest_05_feature_map.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

# ─── PLOT 6: Summary table graphic ───────────────────────────────────────────
def plot6_summary_table(res):
    key_sets = ["NonExG_k2","NonExG_k4","NonExG_k6",
                "QFS_k2","QFS_k4","QFS_k6",
                "MI_k2","MI_k4","MI_k6",
                "ExG_trivial_k1","ExG_trivial_k2"]
    cols = ["Clean Acc","CV Acc","AUC","Fog Acc","Shadow Acc"]
    def row_vals(n):
        r = res[n]
        return [r["Acc_clean"]*100, r["CV_mean"]*100, r["AUC_clean"]*100,
                r["Acc_Fog"]*100, r["Acc_Shadow"]*100]

    data = np.array([row_vals(n) for n in key_sets])
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    ax.axis("off")
    tbl = ax.table(
        cellText=[[f"{v:.1f}" for v in row] for row in data],
        rowLabels=key_sets,
        colLabels=cols,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(12)
    tbl.scale(1.2, 2.0)
    # Color cells by value
    for (r, c), cell in tbl.get_celld().items():
        if r == 0: cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white")
        elif c == -1: cell.set_facecolor(COLORS.get(key_sets[r-1], "#FAFAFA"))
        else:
            val = data[r-1, c]
            g = (val - 50) / 50.0
            g = max(0, min(1, g))
            cell.set_facecolor(plt.cm.RdYlGn(g))
    ax.set_title("Comprehensive Results Summary (All Subsets)",
                 fontweight="bold", fontsize=14, pad=20, y=0.98)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "honest_06_summary_table.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

# ─── Save CSV tables ──────────────────────────────────────────────────────────
def save_tables(res):
    rows = []
    for name, r in res.items():
        rows.append({
            "Method":      name,
            "k":           r["k"],
            "Features":    "|".join(r["Features"]),
            "Clean_Acc":   round(r["Acc_clean"], 4),
            "Clean_AUC":   round(r["AUC_clean"], 4),
            "Clean_F1":    round(r["F1_clean"],  4),
            "CV_Acc":      round(r["CV_mean"],   4),
            "CV_Std":      round(r["CV_std"],    4),
            "Acc_Fog":     round(r["Acc_Fog"],   4),
            "Acc_Glare":   round(r["Acc_Glare"], 4),
            "Acc_Shadow":  round(r["Acc_Shadow"],4),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES_DIR, "honest_results.csv"), index=False)
    print(f"\n  Saved: {TABLES_DIR}/honest_results.csv")
    print("\n" + out.to_string(index=False))

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nComputing all subset results …\n")
    results = compute_all()

    print("\n\nGenerating plots …")
    plot1_cv_bars(results)
    plot2_noise(results)
    plot3_aug_heatmap(results)
    plot4_roc_grid(results)
    plot5_membership(results)
    plot6_summary_table(results)
    save_tables(results)

    print("\n✅  All plots saved to: results/plots/honest_*.png")
