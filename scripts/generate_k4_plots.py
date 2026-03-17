#!/usr/bin/env python3
"""
generate_k4_plots.py
====================
Generates focused, publication-quality plots specifically for the
chosen k=4 qubit subset: [Std_ExG, Mean_RBR, Mean_B, Correlation].
All plots go to results/plots/k4_only/.
"""

import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, auc,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
CSV     = os.path.join(ROOT, "icml_features_FULL.csv")
OUT     = os.path.join(ROOT, "results", "plots", "k4_only")
os.makedirs(OUT, exist_ok=True)

# ─── Data ──────────────────────────────────────────────────────────────────────
df  = pd.read_csv(CSV).dropna()
y   = (df["Label"] == "Post_Defoliation").astype(int).values
K4_FEATS = ["Std_ExG", "Mean_RBR", "Mean_B", "Correlation"]  # VQC-chosen k=4 subset

tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.30,
                                   stratify=y, random_state=42)
X_tr = df[K4_FEATS].values[tr_idx]
X_te = df[K4_FEATS].values[te_idx]
y_tr, y_te = y[tr_idx], y[te_idx]

FEAT_LABELS = {
    "Std_ExG":     "Std ExG\n(Green variability)",
    "Mean_RBR":    "Mean RBR\n(Red-Blue ratio)",
    "Mean_B":      "Mean Blue\n(Background channel)",
    "Correlation": "GLCM Correlation\n(Texture smoothness)",
}

def make_clf():
    return Pipeline([("sc", StandardScaler()),
                     ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                                 probability=True, random_state=42))])

clf = make_clf()
clf.fit(X_tr, y_tr)
pred  = clf.predict(X_te)
prob  = clf.predict_proba(X_te)[:, 1]

CLASS_NAMES = ["Pre-Defoliation", "Post-Defoliation"]
COLORS      = {"Pre_Defoliation": "#5D4037", "Post_Defoliation": "#2E7D32"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "figure.dpi": 150,
})
def spine_off(ax):
    ax.spines[["top","right"]].set_visible(False)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT k4-01: Confusion Matrix (normalized + raw counts)
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion():
    cm_raw  = confusion_matrix(y_te, pred)
    cm_norm = confusion_matrix(y_te, pred, normalize="true")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")

    for ax, cm, title, fmt in zip(
        axes,
        [cm_raw, cm_norm],
        ["Raw Counts", "Normalized (Row %)"],
        ["d", ".2f"],
    ):
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=fmt)
        ax.set_title(f"Confusion Matrix — {title}", fontweight="bold")
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")

    fig.suptitle("k=4 QFS Subset: Confusion Matrix\n"
                 "Features: [Std_ExG, Mean_RBR, Mean_B, Correlation]",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, "k4_01_confusion_matrix.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT k4-02: ROC + Precision-Recall (k=4 only, with CI from 5-fold)
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc_pr():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")

    all_fpr  = np.linspace(0, 1, 200)
    all_tprs, all_aucs = [], []
    all_rec  = np.linspace(0, 1, 200)
    all_precs, all_aps = [], []

    for cv_tr, cv_te in cv.split(X_tr, y_tr):
        c = make_clf()
        c.fit(X_tr[cv_tr], y_tr[cv_tr])
        pr = c.predict_proba(X_tr[cv_te])[:, 1]
        fpr, tpr, _ = roc_curve(y_tr[cv_te], pr)
        all_tprs.append(np.interp(all_fpr, fpr, tpr))
        all_aucs.append(auc(fpr, tpr))
        p_, r_, _   = precision_recall_curve(y_tr[cv_te], pr)
        all_precs.append(np.interp(all_rec, r_[::-1], p_[::-1]))
        all_aps.append(average_precision_score(y_tr[cv_te], pr))

    # ROC
    ax = axes[0]
    mean_tpr = np.mean(all_tprs, axis=0); std_tpr = np.std(all_tprs, axis=0)
    ax.plot(all_fpr, mean_tpr, color="#1565C0", lw=2.5,
            label=f"Mean ROC (AUC={np.mean(all_aucs):.3f} ± {np.std(all_aucs):.3f})")
    ax.fill_between(all_fpr, mean_tpr-std_tpr, mean_tpr+std_tpr,
                    alpha=0.25, color="#1565C0", label="±1 std")
    # Held-out test curve
    fpr_te, tpr_te, _ = roc_curve(y_te, prob)
    ax.plot(fpr_te, tpr_te, color="#E65100", lw=2, ls="--",
            label=f"Hold-out test (AUC={roc_auc_score(y_te, prob):.3f})")
    ax.plot([0,1],[0,1], ":", color="gray", lw=1)
    ax.set(title="ROC Curve — k=4 Subset", xlabel="False Positive Rate",
           ylabel="True Positive Rate")
    ax.legend(loc="lower right"); ax.grid(ls="--", alpha=0.4); spine_off(ax)

    # PR
    ax = axes[1]
    mean_pr = np.mean(all_precs, axis=0); std_pr = np.std(all_precs, axis=0)
    ax.plot(all_rec, mean_pr, color="#2E7D32", lw=2.5,
            label=f"Mean P-R (AP={np.mean(all_aps):.3f} ± {np.std(all_aps):.3f})")
    ax.fill_between(all_rec, mean_pr-std_pr, mean_pr+std_pr,
                    alpha=0.25, color="#2E7D32", label="±1 std")
    p_te, r_te, _ = precision_recall_curve(y_te, prob)
    ax.plot(r_te, p_te, color="#E65100", lw=2, ls="--",
            label=f"Hold-out test (AP={average_precision_score(y_te, prob):.3f})")
    ax.set(title="Precision-Recall Curve — k=4 Subset",
           xlabel="Recall", ylabel="Precision")
    ax.legend(loc="lower left"); ax.grid(ls="--", alpha=0.4); spine_off(ax)

    fig.suptitle("k=4 QFS Subset: ROC & Precision-Recall\n"
                 "5-Fold CV (blue band) + Hold-out Test (orange dashed)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(OUT, "k4_02_roc_pr_curves.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT k4-03: Per-feature KDE distributions (the 4 selected features)
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_distributions():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
    axes = axes.flatten()

    for ax, feat in zip(axes, K4_FEATS):
        for label, col in COLORS.items():
            vals = df[df["Label"] == label][feat].values
            lbl  = ("Post-Defoliation (Brown)" if "Post" in label
                    else "Pre-Defoliation (Green)")
            sns.kdeplot(vals, ax=ax, fill=True, color=col, alpha=0.55,
                        linewidth=2, label=lbl)
        ax.set_title(FEAT_LABELS[feat], fontweight="bold")
        ax.set_xlabel(feat); ax.set_ylabel("Density")
        ax.legend(fontsize=9); ax.grid(ls="--", alpha=0.4); spine_off(ax)

    fig.suptitle("k=4 QFS Subset: Per-Feature Class Distributions\n"
                 "[Std_ExG | Mean_RBR | Mean_B | Correlation] — the VQC-selected 4 features",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, "k4_03_feature_distributions.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT k4-04: Noise degradation curve (k=4 only) with annotated key thresholds
# ══════════════════════════════════════════════════════════════════════════════
def plot_noise_k4():
    sigmas = np.arange(0.0, 0.35, 0.025)
    accs   = []
    for sig in sigmas:
        Xn = X_te + np.random.RandomState(42).normal(0, sig, X_te.shape)
        accs.append(accuracy_score(y_te, clf.predict(Xn)) * 100)

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    ax.plot(sigmas, accs, color="#1565C0", lw=2.8, marker="o", ms=7,
            markerfacecolor="white", markeredgecolor="#1565C0", markeredgewidth=2)
    ax.fill_between(sigmas, accs, min(accs)*0.98, alpha=0.12, color="#1565C0")

    # Annotate key thresholds
    for threshold, label in [(95, "95%"), (90, "90%"), (80, "80%"), (70, "70%")]:
        ax.axhline(threshold, color="gray", lw=0.9, ls="--", alpha=0.5)
        ax.text(sigmas[-1]*1.01, threshold, label, va="center",
                color="gray", fontsize=9)

    # Find where accuracy drops below 90%
    for i, (s, a) in enumerate(zip(sigmas, accs)):
        if a < 90:
            ax.annotate(f"Drops below 90%\nσ={s:.3f}",
                        xy=(s, a), xytext=(s+0.02, a+5),
                        arrowprops=dict(arrowstyle="->", color="#C62828"),
                        fontsize=9.5, color="#C62828", fontweight="bold")
            break

    ax.set_xlabel(r"Gaussian Noise $\sigma$ (applied to test features)", fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontweight="bold")
    ax.set_title("k=4 QFS Subset: Accuracy Under Gaussian Sensor Noise\n"
                 "Features: [Std_ExG, Mean_RBR, Mean_B, Correlation]",
                 fontweight="bold")
    ax.set_ylim(min(accs)*0.92, 103)
    ax.grid(axis="y", ls="--", alpha=0.4); spine_off(ax)
    plt.tight_layout()
    p = os.path.join(OUT, "k4_04_noise_degradation.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT k4-05: 2D PCA scatter of k=4 features (class manifold)
# ══════════════════════════════════════════════════════════════════════════════
def plot_pca_k4():
    X_all = df[K4_FEATS].values
    X_sc  = StandardScaler().fit_transform(X_all)
    pca   = PCA(n_components=2, random_state=42)
    X_2d  = pca.fit_transform(X_sc)
    ev    = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="white")
    for label, col in COLORS.items():
        mask = df["Label"].values == label
        lbl  = ("Post-Defoliation (Brown/Bare)" if "Post" in label
                else "Pre-Defoliation (Lush Green)")
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=col, label=lbl,
                   alpha=0.65, edgecolors="w", linewidth=0.4, s=30)
    try:
        sns.kdeplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=df["Label"].values,
                    fill=True,
                    palette={"Pre_Defoliation": "#A5D6A7",
                             "Post_Defoliation": "#D7CCC8"},
                    alpha=0.30, ax=ax, legend=False, levels=4)
    except Exception:
        pass

    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)", fontweight="bold")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)", fontweight="bold")
    ax.set_title("2D PCA Manifold — k=4 QFS Feature Space\n"
                 "[Std_ExG, Mean_RBR, Mean_B, Correlation]",
                 fontweight="bold")
    ax.legend(markerscale=1.5); ax.grid(ls=":", alpha=0.4); spine_off(ax)
    plt.tight_layout()
    p = os.path.join(OUT, "k4_05_pca_manifold.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT k4-06: Augmentation bar chart (k=4 only, detailed)
# ══════════════════════════════════════════════════════════════════════════════
def plot_augmentation_k4():
    AUG_CONFIGS = {
        "Clean": lambda X: X,
        "Fog (light)": lambda X: 0.75*X + 0.25*X.mean(0) + np.random.RandomState(0).normal(0, 0.02, X.shape),
        "Fog (heavy)": lambda X: 0.55*X + 0.45*X.mean(0) + np.random.RandomState(0).normal(0, 0.04, X.shape),
        "Glare":       lambda X: X + np.random.RandomState(1).normal(0, 0.08, X.shape),
        "Shadow (light)": lambda X: (lambda o: (o.__setitem__(slice(None), X.copy()) or o))(np.empty_like(X)),
        "Shadow (heavy)": lambda X: X * (1 - 0.5*(np.random.RandomState(2).rand(len(X),1) < 0.45)),
        "Noise σ=0.05":  lambda X: X + np.random.RandomState(42).normal(0, 0.05, X.shape),
        "Noise σ=0.10":  lambda X: X + np.random.RandomState(42).normal(0, 0.10, X.shape),
        "Noise σ=0.20":  lambda X: X + np.random.RandomState(42).normal(0, 0.20, X.shape),
    }
    # Fix shadow lambdas properly
    def shadow_light(X):
        out = X.copy()
        mask = np.random.RandomState(2).rand(len(X)) < 0.30
        out[mask] *= 0.75; return out
    def shadow_heavy(X):
        out = X.copy()
        mask = np.random.RandomState(2).rand(len(X)) < 0.55
        out[mask] *= 0.50; return out
    AUG_CONFIGS["Shadow (light)"] = shadow_light
    AUG_CONFIGS["Shadow (heavy)"] = shadow_heavy

    conditions = list(AUG_CONFIGS.keys())
    accs, f1s = [], []
    for name, aug_fn in AUG_CONFIGS.items():
        Xte = aug_fn(X_te.copy())
        p   = clf.predict(Xte)
        accs.append(accuracy_score(y_te, p) * 100)
        f1s.append(f1_score(y_te, p, zero_division=0) * 100)

    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor="white")
    x = np.arange(len(conditions))
    w = 0.38
    bars_acc = ax.bar(x - w/2, accs, w, label="Accuracy",
                      color="#1565C0", edgecolor="black", lw=0.8)
    bars_f1  = ax.bar(x + w/2, f1s,  w, label="F1-Score",
                      color="#2E7D32", edgecolor="black", lw=0.8)
    for bar, v in list(zip(bars_acc, accs)) + list(zip(bars_f1, f1s)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(90, color="#C62828", lw=1.5, ls="--", alpha=0.8, label="90% threshold")
    ax.set_xticks(x); ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Score (%)"); ax.set_ylim(0, 112)
    ax.set_title("k=4 QFS Subset: Accuracy & F1 Under All Augmentation Conditions\n"
                 "[Std_ExG, Mean_RBR, Mean_B, Correlation]", fontweight="bold")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.4); spine_off(ax)
    plt.tight_layout()
    p = os.path.join(OUT, "k4_06_augmentation_detailed.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT k4-07: 5-Fold CV — noise-stressed per-fold metrics (honest variation)
# WHY noise-stressed?
#   Clean CV gives 100% on every fold because Std_ExG alone (AUC=1.0) is in
#   the k=4 QFS subset. To show real fold-to-fold variation we evaluate each
#   fold's trained classifier under σ=0.12 Gaussian noise at test time.
# ══════════════════════════════════════════════════════════════════════════════
def plot_cv_metrics_k4():
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    SIGMA = 0.12          # noise level that produces meaningful variation
    fold_data_clean = []
    fold_data_noise = []

    for fold_i, (cv_tr, cv_te) in enumerate(cv.split(X_tr, y_tr)):
        c = make_clf()
        c.fit(X_tr[cv_tr], y_tr[cv_tr])
        Xte_clean = X_tr[cv_te]
        Xte_noisy = Xte_clean + np.random.RandomState(fold_i).normal(
                        0, SIGMA, Xte_clean.shape)

        label = f"Fold {fold_i+1}"
        for tag, Xte, store in [("clean", Xte_clean, fold_data_clean),
                                 ("noisy", Xte_noisy, fold_data_noise)]:
            pr = c.predict_proba(Xte)[:, 1]
            p  = c.predict(Xte)
            store.append({
                "Fold":      label,
                "Accuracy":  accuracy_score(y_tr[cv_te], p)           * 100,
                "F1":        f1_score(y_tr[cv_te], p, zero_division=0)* 100,
                "Precision": precision_score(y_tr[cv_te], p, zero_division=0)*100,
                "Recall":    recall_score(y_tr[cv_te], p, zero_division=0)   *100,
                "AUC":       roc_auc_score(y_tr[cv_te], pr)           * 100,
            })

    fdf_n = pd.DataFrame(fold_data_noise)
    fdf_c = pd.DataFrame(fold_data_clean)
    metrics = ["Accuracy", "F1", "Precision", "Recall", "AUC"]
    MCOLS   = ["#1565C0", "#2E7D32", "#C62828", "#F57F17", "#6A1B9A"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), facecolor="white",
                              sharey=False)

    # ── Left: noise-stressed (honest variation) ────────────────────────────
    ax = axes[0]
    x  = np.arange(len(fdf_n))
    w  = 0.15
    for i, (m, col) in enumerate(zip(metrics, MCOLS)):
        ax.bar(x + (i-2)*w, fdf_n[m], w, label=m,
               color=col, edgecolor="black", lw=0.7, alpha=0.88)
    for m, col in zip(metrics, MCOLS):
        ax.axhline(fdf_n[m].mean(), color=col, lw=1.5, ls=":", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(fdf_n["Fold"], fontsize=11)
    ax.set_ylabel("Score (%)"); ax.set_ylim(55, 105)
    ax.set_title(f"Noise-Stressed CV (σ={SIGMA})\n"
                 "Shows real fold-to-fold variation", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", ls="--", alpha=0.4); spine_off(ax)

    # ── Right: clean (documented as trivial baseline) ──────────────────────
    ax = axes[1]
    for i, (m, col) in enumerate(zip(metrics, MCOLS)):
        ax.bar(x + (i-2)*w, fdf_c[m], w, label=m,
               color=col, edgecolor="black", lw=0.7, alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(fdf_c["Fold"], fontsize=11)
    ax.set_ylim(85, 105)
    ax.set_title("Clean CV (100% — Std_ExG trivially separates)\n"
                 "Shown for reference only", fontweight="bold",
                 color="#B71C1C")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", ls="--", alpha=0.4); spine_off(ax)

    fig.suptitle("k=4 QFS Subset: Per-Fold CV Metrics\n"
                 "Left = honest (noise-stressed) | Right = clean (trivially 100%)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, "k4_07_cv_per_fold_metrics.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\nGenerating k=4 specific plots → {OUT}\n")
    plot_confusion()
    plot_roc_pr()
    plot_feature_distributions()
    plot_noise_k4()
    plot_pca_k4()
    plot_augmentation_k4()
    plot_cv_metrics_k4()
    print(f"\n✅  7 k=4-specific plots saved to results/plots/k4_only/")
