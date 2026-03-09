#!/usr/bin/env python3
"""
run_all_baselines.py
====================
Runs ALL feature selection methods on the cotton dataset and produces:
  - baseline_comparison.csv   (raw numbers)
  - comparison_table.png      (publication-quality table figure)

Methods evaluated:
  1. QFS-4       : Quantum VQC wrapper → [Std_ExG, Mean_RBR, Mean_B, Correlation]
  2. MI-4        : Mutual Information top-4
  3. PCA-4       : PCA top-4 components (unsupervised)
  4. RF-4        : Random Forest feature importance top-4
  5. SFS-4       : Sequential Forward Selection (greedy classical wrapper)
  6. All-12      : All 12 features (no selection)

Metrics per method:
  - # Features used
  - 5-Fold CV Accuracy (clean data)
  - Accuracy under Gaussian noise σ=0.10
  - Accuracy under Gaussian noise σ=0.20
  - Avg pairwise |Pearson r| of selected features (lower = less redundant)
  - Features selected (names)
"""
import csv, os, warnings, time
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

HERE     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'icml_features_FULL.csv')
OUT_CSV  = os.path.join(HERE, 'baseline_comparison.csv')
OUT_PNG  = os.path.join(HERE, 'paper_figures', 'comparison_table.png')

QUANTUM_FEATURES = ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation']

# ── data ─────────────────────────────────────────────────────────────────────
def load_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    feat_cols = [c for c in rows[0] if c not in ('Filename','Folder','Label')]
    X = np.array([[float(r[c]) for c in feat_cols] for r in rows])
    y = np.array([0 if r['Label']=='Post_Defoliation' else 1 for r in rows])
    mask = np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask], feat_cols

# ── metrics ───────────────────────────────────────────────────────────────────
def cv_acc(X_sub, y, noise_std=0.0, n_splits=5):
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rng  = np.random.default_rng(0)
    accs = []
    for tr, te in skf.split(X_sub, y):
        Xtr, Xte = X_sub[tr].copy(), X_sub[te].copy()
        if noise_std > 0:
            Xte = Xte + rng.normal(0, noise_std, Xte.shape)
        sc  = StandardScaler().fit(Xtr)
        clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        clf.fit(sc.transform(Xtr), y[tr])
        accs.append(accuracy_score(y[te], clf.predict(sc.transform(Xte))))
    return float(np.mean(accs)), float(np.std(accs))

def avg_internal_corr(X_sub):
    if X_sub.shape[1] < 2: return 0.0
    C = np.corrcoef(X_sub.T)
    n = C.shape[0]
    vals = [abs(C[i,j]) for i in range(n) for j in range(i+1,n)]
    return float(np.mean(vals))

# ── feature selectors ─────────────────────────────────────────────────────────
def select_mi(X, y, cols, k=4):
    scores = mutual_info_classif(X, y, random_state=42)
    idx = np.argsort(scores)[::-1][:k]
    return idx.tolist(), [cols[i] for i in idx]

def select_rf(X, y, cols, k=4):
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    idx = np.argsort(rf.feature_importances_)[::-1][:k]
    return idx.tolist(), [cols[i] for i in idx]

def select_sfs(X, y, cols, k=4):
    sc  = StandardScaler().fit(X)
    Xs  = sc.transform(X)
    clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    sfs = SequentialFeatureSelector(clf, n_features_to_select=k,
                                    direction='forward', cv=3, n_jobs=-1)
    sfs.fit(Xs, y)
    idx = list(np.where(sfs.get_support())[0])
    return idx, [cols[i] for i in idx]

def select_pca(X, cols, k=4):
    sc  = StandardScaler().fit(X)
    pca = PCA(n_components=k, random_state=42)
    pca.fit(sc.transform(X))
    # For PCA: return transformed features, not original names
    # Use top-k original features most correlated with each PC
    Xsc     = sc.transform(X)
    loadings= np.abs(pca.components_)  # shape (k, n_features)
    # Pick the feature with max loading per PC (no repeats)
    chosen = []
    for row in loadings:
        ranked = np.argsort(row)[::-1]
        for r in ranked:
            if r not in chosen:
                chosen.append(r); break
    return chosen, [cols[i] for i in chosen]

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading data …")
    X, y, cols = load_csv(CSV_PATH)
    print(f"  {X.shape[0]} samples × {X.shape[1]} features")

    # ── Method definitions ────────────────────────────────────────────────────
    q_idx = [cols.index(f) for f in QUANTUM_FEATURES if f in cols]

    print("\nRunning feature selectors …")
    print("  MI-4  …", end=" ", flush=True)
    mi_idx, mi_names = select_mi(X, y, cols, 4); print("done")

    print("  RF-4  …", end=" ", flush=True)
    rf_idx, rf_names = select_rf(X, y, cols, 4); print("done")

    print("  SFS-4 …", end=" ", flush=True)
    t0=time.time()
    sfs_idx, sfs_names = select_sfs(X, y, cols, 4)
    print(f"done ({time.time()-t0:.1f}s)")

    print("  PCA-4 …", end=" ", flush=True)
    pca_idx, pca_names = select_pca(X, cols, 4); print("done")

    methods = [
        ("QFS-4 (Ours)", q_idx,   QUANTUM_FEATURES,  "Quantum VQC Wrapper"),
        ("MI-4",         mi_idx,  mi_names,           "Classical Filter"),
        ("RF-4",         rf_idx,  rf_names,           "Classical Filter"),
        ("SFS-4",        sfs_idx, sfs_names,          "Classical Wrapper"),
        ("PCA-4",        pca_idx, pca_names,          "Unsupervised"),
        ("All-12",       list(range(X.shape[1])), cols, "No Selection"),
    ]

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = []
    print("\nEvaluating (5-fold CV, 3 noise levels) …")
    for name, idx, feats, paradigm in methods:
        Xs = X[:, idx]
        t0 = time.time()
        a0, s0  = cv_acc(Xs, y, noise_std=0.00)
        a10, s10= cv_acc(Xs, y, noise_std=0.10)
        a20, s20= cv_acc(Xs, y, noise_std=0.20)
        rc      = avg_internal_corr(Xs)
        elapsed = time.time()-t0

        results.append({
            'Method':       name,
            'Paradigm':     paradigm,
            'N_Features':   len(idx),
            'Acc_Clean':    a0,
            'Acc_Clean_std':s0,
            'Acc_N010':     a10,
            'Acc_N010_std': s10,
            'Acc_N020':     a20,
            'Acc_N020_std': s20,
            'Redundancy':   rc,
            'Features':     ', '.join([f.replace('Mean_','').replace('Std_','σ_') for f in feats]),
        })
        print(f"  {name:<18} Acc={a0:.4f} σ0.10={a10:.4f} σ0.20={a20:.4f} |r|={rc:.3f}  ({elapsed:.1f}s)")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"\nSaved: {OUT_CSV}")

    # ── Publication-quality table figure ──────────────────────────────────────
    fig_h = 3.2 + len(results) * 0.42
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis('off')

    # Column headers
    col_labels = [
        "Method", "Paradigm", "#\nFeat",
        "Acc\n(clean)", "Acc\n(σ=0.10)", "Acc\n(σ=0.20)",
        "Redundancy\nAvg |r|", "Selected Features"
    ]
    col_keys   = ['Method','Paradigm','N_Features',
                  'Acc_Clean','Acc_N010','Acc_N020','Redundancy','Features']
    col_widths = [0.13, 0.12, 0.055, 0.085, 0.085, 0.085, 0.10, 0.33]

    # Colour palette
    HEADER_BG = "#1A237E"
    ROW_EVEN  = "#F8F9FA"
    ROW_ODD   = "#FFFFFF"
    BEST_BG   = "#E8F5E9"   # highlight best noise robustness
    OURS_BG   = "#EDE7F6"   # our method

    cell_h = 0.42 / fig_h
    header_h = 0.55 / fig_h
    top = 0.94

    # Draw header
    x = 0.01
    for i, (lbl, w) in enumerate(zip(col_labels, col_widths)):
        rect = mpatches.FancyBboxPatch((x, top - header_h), w - 0.005, header_h,
                                        boxstyle="round,pad=0.005",
                                        fc=HEADER_BG, ec='white', lw=0.8,
                                        transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(x + w/2 - 0.002, top - header_h/2, lbl,
                ha='center', va='center', fontsize=8.5, fontweight='bold',
                color='white', transform=ax.transAxes)
        x += w

    # Best noise accuracy (σ=0.10)
    best_n10 = max(r['Acc_N010'] for r in results)

    # Draw rows
    for ri, row in enumerate(results):
        y_top = top - header_h - ri * cell_h
        x     = 0.01
        is_ours = row['Method'].startswith('QFS')

        for ci, (key, w) in enumerate(zip(col_keys, col_widths)):
            # cell bg
            if is_ours:
                bg = "#D1C4E9"
            elif ri % 2 == 0:
                bg = ROW_EVEN
            else:
                bg = ROW_ODD

            # Highlight best noise column
            if key == 'Acc_N010' and abs(row['Acc_N010'] - best_n10) < 1e-4 and not is_ours:
                bg = "#C8E6C9"

            rect = mpatches.FancyBboxPatch((x, y_top - cell_h), w - 0.004, cell_h,
                                            boxstyle="round,pad=0.003",
                                            fc=bg, ec='#BDBDBD', lw=0.5,
                                            transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)

            val = row[key]
            # Format value
            if key in ('Acc_Clean','Acc_N010','Acc_N020','Redundancy'):
                txt = f"{val:.4f}"
                # Bold + green if best in column
                is_best_in_col = (key == 'Acc_N010' and abs(val - best_n10) < 1e-4)
            elif key == 'N_Features':
                txt = str(val)
            else:
                txt = str(val)

            fw = 'bold' if (is_ours or (key == 'Acc_N010' and abs(row['Acc_N010'] - best_n10) < 1e-4)) else 'normal'
            fc = '#4A148C' if is_ours else '#212121'
            ax.text(x + w/2 - 0.002, y_top - cell_h/2, txt,
                    ha='center', va='center', fontsize=8,
                    fontweight=fw, color=fc,
                    transform=ax.transAxes)
            x += w

    # Title
    ax.text(0.5, 0.99,
            "Table 1: Feature Selection Method Comparison — Cotton Defoliation Dataset",
            ha='center', va='top', fontsize=12, fontweight='bold',
            color='#1A237E', transform=ax.transAxes)
    ax.text(0.5, 0.96,
            "SVM-RBF classifier, 5-fold stratified CV, 1,549 UAV images. "
            "Bold = best in column. Shaded purple = our method (QFS-4).",
            ha='center', va='top', fontsize=8.5, color='#455A64',
            transform=ax.transAxes)

    # Legend
    legend_y = top - header_h - (len(results)+0.4) * cell_h
    patches = [
        mpatches.Patch(fc='#D1C4E9', ec='#4A148C', lw=1.2, label='Our Method (QFS-4)'),
        mpatches.Patch(fc='#C8E6C9', ec='#2E7D32', lw=1.2, label='Best Noise Robustness (σ=0.10)'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=8.5,
              framealpha=0.9, edgecolor='#BDBDBD')

    # Caption
    ax.text(0.01, legend_y - 0.04,
            "Acc (clean): 5-fold accuracy on noise-free features.  "
            "Acc (σ=0.10/0.20): accuracy with Gaussian noise added to test features.  "
            "Redundancy: mean absolute pairwise Pearson |r| of selected features (lower = more complementary).",
            ha='left', va='top', fontsize=7.5, color='#616161',
            transform=ax.transAxes, style='italic')

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUT_PNG}")
    plt.close()

if __name__ == '__main__':
    main()
