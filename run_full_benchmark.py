#!/usr/bin/env python3
"""
Full Benchmark Script for Cotton Defoliation Classification
============================================================
Runs 5-fold cross-validation across 4 methods:
  1. Quantum-Selected Features  : [Std_ExG, Mean_RBR, Mean_B, Correlation]  (4 features)
  2. MI-Selected Features       : Top 4 by Mutual Information               (4 features)
  3. All Features               : All 12 extracted features                 (12 features)
  4. ResNet-18 (CNN Baseline)   : Fine-tuned on raw images (if available)
     NOTE: ResNet baseline requires raw images + torchvision. Skipped if not available.

Outputs:
  - benchmark_results.csv  : Per-fold accuracy/F1 for all methods
  - benchmark_report.txt   : Human-readable summary with means and std devs
  - benchmark_plots.png    : Bar chart comparing all methods
"""

import csv
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. Load CSV (no pandas dependency)
# ──────────────────────────────────────────────
def load_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    feature_cols = [c for c in rows[0].keys()
                    if c not in ('Filename', 'Folder', 'Label')]
    X = np.array([[float(r[c]) for c in feature_cols] for r in rows])
    y = np.array([0 if r['Label'] == 'Post_Defoliation' else 1 for r in rows])
    return X, y, feature_cols


# ──────────────────────────────────────────────
# 2. Minimal implementations (no sklearn needed
#    for the core logic — but we try sklearn first)
# ──────────────────────────────────────────────
try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_score, recall_score,
                                 confusion_matrix)
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not found. Install with: pip install scikit-learn")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found. Plots will be skipped.")


# ──────────────────────────────────────────────
# 3. Feature subset definitions
# ──────────────────────────────────────────────
QUANTUM_FEATURES = ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation']

def get_feature_indices(all_cols, wanted):
    return [all_cols.index(w) for w in wanted if w in all_cols]

def top_k_mi_features(X, y, cols, k=4):
    scores = mutual_info_classif(X, y, random_state=42)
    top_idx = np.argsort(scores)[::-1][:k]
    return [cols[i] for i in top_idx], top_idx


# ──────────────────────────────────────────────
# 4. Cross-validation runner
# ──────────────────────────────────────────────
def run_cv(X_subset, y, n_splits=5, label=""):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, precs, recs = [], [], [], []

    for fold, (tr, te) in enumerate(skf.split(X_subset, y)):
        X_tr, X_te = X_subset[tr], X_subset[te]
        y_tr, y_te = y[tr], y[te]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average='weighted'))
        precs.append(precision_score(y_te, y_pred, average='weighted', zero_division=0))
        recs.append(recall_score(y_te, y_pred, average='weighted', zero_division=0))

        print(f"  [{label}] Fold {fold+1}/{n_splits}: "
              f"Acc={accs[-1]:.4f}  F1={f1s[-1]:.4f}")

    return {
        'acc_mean': np.mean(accs),   'acc_std': np.std(accs),
        'f1_mean':  np.mean(f1s),    'f1_std':  np.std(f1s),
        'prec_mean':np.mean(precs),  'prec_std':np.std(precs),
        'rec_mean': np.mean(recs),   'rec_std': np.std(recs),
        'folds_acc': accs, 'folds_f1': f1s,
    }


# ──────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────
def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'icml_features_FULL.csv')
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run extract_classical_only.py first.")
        sys.exit(1)

    print("=" * 60)
    print("  COTTON DEFOLIATION — FULL BENCHMARK")
    print("=" * 60)
    print(f"Loading: {csv_path}")
    X, y, cols = load_csv(csv_path)

    # Remove rows with NaN/Inf
    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: Post_Def={np.sum(y==0)}, Pre_Def={np.sum(y==1)}")
    print()

    results = {}

    # ── Method 1: Quantum-Selected Features ──────────────────
    q_idx = get_feature_indices(cols, QUANTUM_FEATURES)
    if len(q_idx) < len(QUANTUM_FEATURES):
        missing = [f for f in QUANTUM_FEATURES if f not in cols]
        print(f"WARNING: Missing quantum features: {missing}")
    print(f"[1/3] Quantum-Selected ({len(q_idx)} features): {QUANTUM_FEATURES}")
    t0 = time.time()
    results['Quantum-Selected (4 feat)'] = run_cv(X[:, q_idx], y, label="QFS")
    results['Quantum-Selected (4 feat)']['time'] = time.time() - t0
    print()

    # ── Method 2: MI-Selected Features (same k=4) ────────────
    mi_names, mi_idx = top_k_mi_features(X, y, cols, k=4)
    print(f"[2/3] MI-Selected ({len(mi_idx)} features): {mi_names}")
    t0 = time.time()
    results['MI-Selected (4 feat)'] = run_cv(X[:, mi_idx], y, label="MI-4")
    results['MI-Selected (4 feat)']['time'] = time.time() - t0
    results['MI-Selected (4 feat)']['features'] = mi_names
    print()

    # ── Method 3: All 12 Features ────────────────────────────
    print(f"[3/3] All Features ({X.shape[1]} features)")
    t0 = time.time()
    results['All Features (12 feat)'] = run_cv(X, y, label="ALL")
    results['All Features (12 feat)']['time'] = time.time() - t0
    print()

    # ── Print Summary Table ───────────────────────────────────
    print("=" * 60)
    print("  RESULTS SUMMARY (5-Fold Cross-Validation, SVM-RBF)")
    print("=" * 60)
    header = f"{'Method':<30} {'Acc':>8} {'±':>5} {'F1':>8} {'±':>5} {'Time':>7}"
    print(header)
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<30} {r['acc_mean']:>7.4f} {r['acc_std']:>5.4f} "
              f"{r['f1_mean']:>7.4f} {r['f1_std']:>5.4f} "
              f"{r.get('time', 0):>6.1f}s")

    # ── Save benchmark_results.csv ────────────────────────────
    out_csv = os.path.join(os.path.dirname(__file__), 'benchmark_results.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Acc_Mean', 'Acc_Std', 'F1_Mean', 'F1_Std',
                         'Prec_Mean', 'Rec_Mean', 'Features', 'Time_s'])
        for name, r in results.items():
            feat_str = ';'.join(r.get('features', QUANTUM_FEATURES if 'Quantum' in name else []))
            writer.writerow([name,
                             f"{r['acc_mean']:.4f}", f"{r['acc_std']:.4f}",
                             f"{r['f1_mean']:.4f}",  f"{r['f1_std']:.4f}",
                             f"{r['prec_mean']:.4f}", f"{r['rec_mean']:.4f}",
                             feat_str, f"{r.get('time',0):.1f}"])
    print(f"\nSaved: {out_csv}")

    # ── Save benchmark_report.txt ─────────────────────────────
    out_txt = os.path.join(os.path.dirname(__file__), 'benchmark_report.txt')
    with open(out_txt, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  COTTON DEFOLIATION CLASSIFICATION BENCHMARK\n")
        f.write(f"  Dataset: {X.shape[0]} images, {X.shape[1]} features\n")
        f.write(f"  Classes: Post_Def={np.sum(y==0)}, Pre_Def={np.sum(y==1)}\n")
        f.write(f"  Classifier: SVM-RBF (C=10, gamma=scale)\n")
        f.write(f"  Evaluation: 5-fold Stratified Cross-Validation\n")
        f.write("=" * 60 + "\n\n")
        for name, r in results.items():
            f.write(f"Method: {name}\n")
            f.write(f"  Accuracy:  {r['acc_mean']:.4f} ± {r['acc_std']:.4f}\n")
            f.write(f"  F1 Score:  {r['f1_mean']:.4f} ± {r['f1_std']:.4f}\n")
            f.write(f"  Precision: {r['prec_mean']:.4f} ± {r['prec_std']:.4f}\n")
            f.write(f"  Recall:    {r['rec_mean']:.4f} ± {r['rec_std']:.4f}\n")
            if 'features' in r:
                f.write(f"  Features:  {r['features']}\n")
            f.write("\n")

        # Key finding paragraph
        q = results['Quantum-Selected (4 feat)']
        m = results['MI-Selected (4 feat)']
        a = results['All Features (12 feat)']
        delta_mi  = (q['acc_mean'] - m['acc_mean']) * 100
        delta_all = (q['acc_mean'] - a['acc_mean']) * 100
        f.write("KEY FINDING:\n")
        f.write(f"  Quantum-selected subset ({len(q_idx)} features) achieves "
                f"{q['acc_mean']*100:.1f}% accuracy.\n")
        f.write(f"  vs MI-selected (same k=4): {delta_mi:+.1f}% difference\n")
        f.write(f"  vs All-features (12):      {delta_all:+.1f}% difference\n")

    print(f"Saved: {out_txt}")

    # ── Plot ──────────────────────────────────────────────────
    if HAS_MPL:
        names  = list(results.keys())
        accs   = [results[n]['acc_mean']  for n in names]
        stds   = [results[n]['acc_std']   for n in names]
        f1s    = [results[n]['f1_mean']   for n in names]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, accs, width, yerr=stds, capsize=5,
                       label='Accuracy', color='#2196F3', alpha=0.85)
        bars2 = ax.bar(x + width/2, f1s,  width,
                       label='F1 Score',  color='#4CAF50', alpha=0.85)

        ax.set_xlabel('Method', fontsize=13)
        ax.set_ylabel('Score', fontsize=13)
        ax.set_title('Cotton Defoliation Classification\n5-Fold CV Benchmark (SVM-RBF)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars1, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        for bar, val in zip(bars2, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        out_png = os.path.join(os.path.dirname(__file__), 'benchmark_plots.png')
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_png}")
        plt.close()

    print("\nDone! Check benchmark_report.txt for the paper-ready summary.")


if __name__ == "__main__":
    main()
