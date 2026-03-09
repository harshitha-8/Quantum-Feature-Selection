#!/usr/bin/env python3
"""
Experiment 2 — Cross-Folder Generalization
Leave-one-folder-out CV: train on 5 flight folders, test on the 6th.
Demonstrates generalization across different flight dates/conditions.
Output: cross_folder_results.csv, cross_folder_bar.png
"""
import csv, os, warnings
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'icml_features_FULL.csv')
OUT_CSV  = os.path.join(HERE, 'cross_folder_results.csv')
OUT_PNG  = os.path.join(HERE, 'cross_folder_bar.png')

QUANTUM_FEATURES = ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation']

# ── helpers ──────────────────────────────────────────────────────────────────
def load_csv_with_folders(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    feat_cols = [c for c in rows[0] if c not in ('Filename','Folder','Label')]
    X       = np.array([[float(r[c]) for c in feat_cols] for r in rows])
    y       = np.array([0 if r['Label']=='Post_Defoliation' else 1 for r in rows])
    folders = np.array([r['Folder'] for r in rows])
    mask    = np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask], folders[mask], feat_cols

def evaluate_subset(X_tr, y_tr, X_te, y_te, idx):
    sc  = StandardScaler().fit(X_tr[:, idx])
    clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    clf.fit(sc.transform(X_tr[:, idx]), y_tr)
    pred = clf.predict(sc.transform(X_te[:, idx]))
    return accuracy_score(y_te, pred), f1_score(y_te, pred, average='weighted', zero_division=0)

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading data …")
    X, y, folders, cols = load_csv_with_folders(CSV_PATH)
    unique_folders = sorted(set(folders))
    print(f"  {X.shape[0]} samples | Folders: {unique_folders}")

    # Build feature subsets
    q_idx   = [cols.index(f) for f in QUANTUM_FEATURES if f in cols]
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_idx  = np.argsort(mi_scores)[::-1][:4].tolist()
    mi_names = [cols[i] for i in mi_idx]

    # PCA baseline (4 components)
    pca = PCA(n_components=4, random_state=42).fit(StandardScaler().fit_transform(X))

    methods_idx = {
        'QFS-4 (Quantum)':  q_idx,
        'MI-4 (Classical)': mi_idx,
        'All-12':           list(range(X.shape[1])),
    }

    print(f"\n  QFS features : {QUANTUM_FEATURES}")
    print(f"  MI  features : {mi_names}")
    print(f"\nLeave-one-folder-out CV ({len(unique_folders)} folds) …")

    rows_out = []
    summary  = {m: {'accs':[], 'f1s':[]} for m in methods_idx}

    for test_folder in unique_folders:
        tr_mask = folders != test_folder
        te_mask = folders == test_folder
        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_te, y_te = X[te_mask], y[te_mask]

        print(f"\n  Test folder: {test_folder} ({te_mask.sum()} samples, "
              f"classes: {np.bincount(y_te)})")

        for name, idx in methods_idx.items():
            acc, f1 = evaluate_subset(X_tr, y_tr, X_te, y_te, idx)
            summary[name]['accs'].append(acc)
            summary[name]['f1s'].append(f1)
            rows_out.append({'TestFolder': test_folder, 'Method': name,
                             'Accuracy': f'{acc:.4f}', 'F1': f'{f1:.4f}'})
            print(f"    {name:<25} acc={acc:.4f}  f1={f1:.4f}")

    # Save CSV
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['TestFolder','Method','Accuracy','F1'])
        w.writeheader(); w.writerows(rows_out)
    print(f"\nSaved: {OUT_CSV}")

    # Summary
    print("\n── Summary ─────────────────────────────────────")
    for name, vals in summary.items():
        print(f"  {name:<25}  Acc={np.mean(vals['accs']):.4f}±{np.std(vals['accs']):.4f}"
              f"  F1={np.mean(vals['f1s']):.4f}±{np.std(vals['f1s']):.4f}")

    # Plot — grouped bar per folder
    n_methods = len(methods_idx)
    n_folders = len(unique_folders)
    x         = np.arange(n_folders)
    width     = 0.25
    colors    = ['#1565C0','#2E7D32','#B71C1C']

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, vals) in enumerate(summary.items()):
        bars = ax.bar(x + (i - 1)*width, [v*100 for v in vals['accs']],
                      width, label=name, color=colors[i], alpha=0.85)
        for b, v in zip(bars, vals['accs']):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                    f'{v*100:.0f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Held-Out Flight Folder', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Cross-Folder Generalization (Leave-One-Folder-Out)\nSVM-RBF Classifier', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_',' ')[:20] for f in unique_folders],
                       rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
    print(f"Saved: {OUT_PNG}")
    plt.close()

if __name__ == '__main__':
    main()
