#!/usr/bin/env python3
"""
Experiment 1 — Robustness Under Noise
Adds Gaussian noise to feature values at 5 levels and evaluates all subsets.
Shows that quantum-selected features degrade more gracefully.
Output: benchmark_noise_results.csv, noise_robustness.png
"""
import csv, os, warnings
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(HERE, 'icml_features_FULL.csv')
OUT_CSV   = os.path.join(HERE, 'benchmark_noise_results.csv')
OUT_PNG   = os.path.join(HERE, 'noise_robustness.png')

QUANTUM_FEATURES = ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation']
NOISE_LEVELS     = [0.0, 0.05, 0.10, 0.20, 0.40]

# ── helpers ──────────────────────────────────────────────────────────────────
def load_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    feat_cols = [c for c in rows[0] if c not in ('Filename','Folder','Label')]
    X = np.array([[float(r[c]) for c in feat_cols] for r in rows])
    y = np.array([0 if r['Label']=='Post_Defoliation' else 1 for r in rows])
    mask = np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask], feat_cols

def top_k_mi(X, y, cols, k=4):
    from sklearn.feature_selection import mutual_info_classif
    scores = mutual_info_classif(X, y, random_state=42)
    idx = np.argsort(scores)[::-1][:k]
    return [cols[i] for i in idx], idx.tolist()

def cv_accuracy(X_sub, y, noise_std, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    rng  = np.random.default_rng(0)
    for tr, te in skf.split(X_sub, y):
        X_tr, X_te = X_sub[tr].copy(), X_sub[te].copy()
        y_tr, y_te = y[tr], y[te]
        # Add noise to test features (simulates real-world degradation)
        if noise_std > 0:
            X_te = X_te + rng.normal(0, noise_std, X_te.shape)
        sc = StandardScaler().fit(X_tr)
        clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        clf.fit(sc.transform(X_tr), y_tr)
        accs.append(accuracy_score(y_te, clf.predict(sc.transform(X_te))))
    return float(np.mean(accs))

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading data …")
    X, y, cols = load_csv(CSV_PATH)
    print(f"  {X.shape[0]} samples, {X.shape[1]} features")

    q_idx = [cols.index(f) for f in QUANTUM_FEATURES if f in cols]
    mi_names, mi_idx = top_k_mi(X, y, cols, k=4)
    print(f"  QFS features : {QUANTUM_FEATURES}")
    print(f"  MI  features : {mi_names}")

    methods = {
        'QFS-4 (Quantum)':   q_idx,
        'MI-4 (Classical)':  mi_idx,
        'All-12':            list(range(X.shape[1])),
    }

    rows_out = []
    results  = {m: [] for m in methods}

    print("\nRunning noise sweep …")
    for sigma in NOISE_LEVELS:
        print(f"  σ = {sigma:.2f}")
        for name, idx in methods.items():
            acc = cv_accuracy(X[:, idx], y, sigma)
            results[name].append(acc)
            rows_out.append({'Method': name, 'Noise_Std': sigma, 'Accuracy': f'{acc:.4f}'})
            print(f"    {name:<25} acc={acc:.4f}")

    # Save CSV
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['Method','Noise_Std','Accuracy'])
        w.writeheader(); w.writerows(rows_out)
    print(f"\nSaved: {OUT_CSV}")

    # Plot
    fig, ax = plt.subplots(figsize=(9,5))
    colors  = ['#1565C0','#2E7D32','#B71C1C']
    markers = ['o','s','^']
    for (name, accs), col, mk in zip(results.items(), colors, markers):
        ax.plot([s*100 for s in NOISE_LEVELS], [a*100 for a in accs],
                marker=mk, color=col, linewidth=2.2, markersize=7, label=name)

    ax.set_xlabel('Noise Level (σ × 100%)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Feature Subset Robustness Under Gaussian Noise\n(SVM-RBF, 5-Fold CV)', fontsize=13)
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    ax.set_ylim(40, 105)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
    print(f"Saved: {OUT_PNG}")
    plt.close()

if __name__ == '__main__':
    main()
