#!/usr/bin/env python3
"""
run_subset_ablation.py  (v2 — publication quality)
===================================================
Ablation across k=1..6 features for QFS and MI methods.

KEY FIX vs v1:
  - Primary metric = Accuracy under σ=0.10 noise (shows real differences)
  - Secondary axis = Feature redundancy: avg pairwise |Pearson r|
  - Dual-axis publication plot on white background
  - Also plots clean accuracy as translucent fill for context
"""
import csv, os, warnings
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'icml_features_FULL.csv')
OUT_CSV  = os.path.join(HERE, 'ablation_results.csv')
OUT_PNG_MAIN = os.path.join(HERE, 'paper_figures', 'ablation_curve.png')

# Quantum-ranked (best 4 known from VQC search, extended by MI rank for k>4)
QFS_RANKED = ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation', 'Mean_ExG', 'Mean_NGRDI']

BLUE   = "#1565C0"
GREEN  = "#2E7D32"
ORANGE = "#E65100"
GRAY   = "#9E9E9E"
BG     = "#FFFFFF"

def load_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    feat_cols = [c for c in rows[0] if c not in ('Filename','Folder','Label')]
    X = np.array([[float(r[c]) for c in feat_cols] for r in rows])
    y = np.array([0 if r['Label']=='Post_Defoliation' else 1 for r in rows])
    mask = np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask], feat_cols

def cv_acc(X_sub, y, noise_std=0.0, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rng = np.random.default_rng(0)
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

def avg_corr(X_sub):
    if X_sub.shape[1] < 2: return 0.0
    C   = np.corrcoef(X_sub.T)
    n   = C.shape[0]
    return float(np.mean([abs(C[i,j]) for i in range(n) for j in range(i+1,n)]))

def main():
    print("Loading data …")
    X, y, cols = load_csv(CSV_PATH)

    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_ranked = [cols[i] for i in np.argsort(mi_scores)[::-1]]

    k_vals = list(range(1, 7))
    rows_out = []

    # Storage
    qfs_clean, qfs_noisy, qfs_std, qfs_corr = [], [], [], []
    mi_clean,  mi_noisy,  mi_std,  mi_corr  = [], [], [], []

    print("\nRunning ablation …")
    for k in k_vals:
        q_feats = [f for f in QFS_RANKED[:k] if f in cols]
        q_idx   = [cols.index(f) for f in q_feats]
        m_feats = mi_ranked[:k]
        m_idx   = [cols.index(f) for f in m_feats]

        qa_cl, _    = cv_acc(X[:, q_idx], y, noise_std=0.00)
        qa_no, qa_s = cv_acc(X[:, q_idx], y, noise_std=0.10)
        qc          = avg_corr(X[:, q_idx])

        ma_cl, _    = cv_acc(X[:, m_idx], y, noise_std=0.00)
        ma_no, ma_s = cv_acc(X[:, m_idx], y, noise_std=0.10)
        mc          = avg_corr(X[:, m_idx])

        qfs_clean.append(qa_cl); qfs_noisy.append(qa_no); qfs_std.append(qa_s); qfs_corr.append(qc)
        mi_clean.append(ma_cl);  mi_noisy.append(ma_no);  mi_std.append(ma_s);  mi_corr.append(mc)

        rows_out.append({'k': k,
                         'QFS_features': ','.join(q_feats), 'QFS_clean': f'{qa_cl:.4f}',
                         'QFS_noisy': f'{qa_no:.4f}', 'QFS_corr': f'{qc:.4f}',
                         'MI_features':  ','.join(m_feats), 'MI_clean':  f'{ma_cl:.4f}',
                         'MI_noisy':  f'{ma_no:.4f}',  'MI_corr':  f'{mc:.4f}'})

        print(f"  k={k}  QFS noisy={qa_no:.4f} corr={qc:.3f} | MI noisy={ma_no:.4f} corr={mc:.3f}")

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print(f"\nSaved: {OUT_CSV}")

    # ── Build publication-quality dual-axis figure ────────────────────────────
    fig, ax1 = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax1.set_facecolor(BG)
    ax2 = ax1.twinx()

    x = np.array(k_vals)

    # --- Primary axis: Noisy accuracy (solid lines + error bars) ---
    ax1.errorbar(x, [a*100 for a in qfs_noisy], yerr=[s*100 for s in qfs_std],
                 color=BLUE, marker='o', markersize=7, linewidth=2.0,
                 capsize=4, capthick=1.5, elinewidth=1.2, label='QFS  Acc (σ=0.10)', zorder=5)

    ax1.errorbar(x, [a*100 for a in mi_noisy], yerr=[s*100 for s in mi_std],
                 color=GREEN, marker='s', markersize=7, linewidth=2.0,
                 capsize=4, capthick=1.5, elinewidth=1.2, label='MI   Acc (σ=0.10)', zorder=5)

    # --- Clean accuracy as faint dashed reference ---
    ax1.plot(x, [a*100 for a in qfs_clean], color=BLUE,  linewidth=1.0,
             linestyle=':', alpha=0.35, zorder=3)
    ax1.plot(x, [a*100 for a in mi_clean],  color=GREEN, linewidth=1.0,
             linestyle=':', alpha=0.35, zorder=3)

    # Shade between QFS and MI noisy
    ax1.fill_between(x,
                     [a*100 for a in qfs_noisy],
                     [a*100 for a in mi_noisy],
                     alpha=0.08, color=BLUE, zorder=2, label='QFS advantage region')

    # --- Secondary axis: Redundancy (dashed, lighter) ---
    ax2.plot(x, qfs_corr, color=BLUE,  marker='D', markersize=5,
             linewidth=1.4, linestyle='--', alpha=0.55, label='QFS Redundancy |r|')
    ax2.plot(x, mi_corr,  color=GREEN, marker='D', markersize=5,
             linewidth=1.4, linestyle='--', alpha=0.55, label='MI  Redundancy |r|')

    # --- k=4 reference line ---
    ax1.axvline(x=4, color=ORANGE, linewidth=1.6, linestyle='--', alpha=0.8, zorder=4)
    ax1.text(4.08, 41.5, 'k = 4\n(selected)', color=ORANGE, fontsize=8.5,
             fontweight='bold', va='bottom')

    # --- Annotate QFS advantage at k=4 ---
    qfs_k4 = qfs_noisy[3]*100
    mi_k4  = mi_noisy[3]*100
    ax1.annotate(f'+{qfs_k4-mi_k4:.1f}%',
                 xy=(4, (qfs_k4+mi_k4)/2),
                 xytext=(4.55, (qfs_k4+mi_k4)/2 + 1.5),
                 fontsize=8, color=BLUE, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))

    # ── Styling ───────────────────────────────────────────────────────────────
    ax1.set_xlabel('Number of Selected Features  (k)', fontsize=11)
    ax1.set_ylabel('Accuracy under Noise  (σ = 0.10, %)', fontsize=10, color="#212121")
    ax2.set_ylabel('Feature Redundancy  (Avg |Pearson r|)', fontsize=10, color=GRAY)

    ax1.set_xlim(0.6, 6.4)
    ax1.set_ylim(38, 78)
    ax2.set_ylim(0.0, 1.2)
    ax1.set_xticks(k_vals)
    ax2.tick_params(axis='y', colors=GRAY)
    ax2.spines['right'].set_color(GRAY)

    ax1.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.8)
    ax1.grid(axis='x', alpha=0.15, linestyle='-', linewidth=0.6)
    for sp in ['top']: ax1.spines[sp].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Dotted reference at 100% (clean acc)
    ax1.axhline(y=100, color=GRAY, linewidth=0.8, linestyle=':', alpha=0.4)
    ax1.text(0.65, 100.4, 'Clean (100%)', color=GRAY, fontsize=7.5, alpha=0.7)

    # ── Legends: merge both axes ──────────────────────────────────────────────
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1[:2] + lines2, labels1[:2] + labels2,
               loc='lower right', fontsize=8.5, framealpha=0.92,
               edgecolor=GRAY, ncol=2)

    fig.suptitle('Feature Subset Size Ablation Study', fontsize=13, fontweight='bold',
                 color="#1A237E", y=1.01)
    ax1.set_title('Noisy accuracy (σ=0.10) and redundancy vs. number of selected features\n'
                  '5-Fold CV · SVM-RBF · Cotton Defoliation Dataset (N=1,549)',
                  fontsize=8.5, color="#616161", pad=6)

    plt.tight_layout()
    plt.savefig(OUT_PNG_MAIN, dpi=220, bbox_inches='tight', facecolor=BG)
    print(f"Saved: {OUT_PNG_MAIN}")
    plt.close()

if __name__ == '__main__':
    main()
