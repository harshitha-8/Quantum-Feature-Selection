#!/usr/bin/env python3
"""
Complete Evaluation Pipeline for Quantum Feature Selection
===========================================================
Runs VQC pipeline and computes comprehensive metrics:
- Confusion matrices for each method
- FPR, F1, mIoU for each configuration
- Metrics under fog/glare/shadow augmentation

Methods evaluated:
1. All-14 features (no selection)
2. MI-4 (Mutual Information top 4)
3. VQC k=2, k=4, k=6 (Quantum Feature Selection)

Output:
- evaluation_results.csv: All metrics
- confusion_matrices.txt: Detailed confusion matrices
- augmentation_results.csv: Performance under augmentations
"""

import os
import csv
import warnings
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'icml_features_FULL.csv')

# Feature subsets from quantum selection (from ablation study)
QFS_SUBSETS = {
    2: ['Std_ExG', 'Mean_RBR'],
    4: ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation'],
    6: ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation', 'Mean_ExG', 'Mean_NGRDI'],
}


def load_data(path):
    """Load CSV and return X, y, feature names."""
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    
    feat_cols = [c for c in rows[0] if c not in ('Filename', 'Folder', 'Label')]
    X = np.array([[float(r[c]) for c in feat_cols] for r in rows])
    y = np.array([0 if r['Label'] == 'Post_Defoliation' else 1 for r in rows])
    
    mask = np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask], feat_cols


def get_feature_indices(all_cols, wanted):
    """Get indices of wanted features from column list."""
    return [all_cols.index(w) for w in wanted if w in all_cols]


def select_mi_features(X, y, cols, k):
    """Select top-k features using Mutual Information."""
    scores = mutual_info_classif(X, y, random_state=42)
    top_idx = np.argsort(scores)[::-1][:k]
    return top_idx.tolist(), [cols[i] for i in top_idx]


def compute_miou(y_true, y_pred, n_classes=2):
    """Compute mean Intersection over Union."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    iou_per_class = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        if tp + fp + fn > 0:
            iou = tp / (tp + fp + fn)
        else:
            iou = 0.0
        iou_per_class.append(iou)
    return np.mean(iou_per_class)


def compute_fpr(y_true, y_pred):
    """Compute False Positive Rate."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def apply_fog_augmentation(X, intensity=0.3):
    """Simulate fog by reducing contrast and adding haze."""
    X_aug = X.copy()
    X_aug = X_aug * (1 - intensity) + intensity * np.mean(X_aug, axis=0)
    noise = np.random.normal(0, intensity * 0.1, X_aug.shape)
    return X_aug + noise


def apply_glare_augmentation(X, intensity=0.3):
    """Simulate glare by adding bright spots and saturation."""
    X_aug = X.copy()
    n_samples = X_aug.shape[0]
    glare_mask = np.random.random(n_samples) < 0.3
    X_aug[glare_mask] = X_aug[glare_mask] * (1 + intensity)
    return np.clip(X_aug, X.min(axis=0), X.max(axis=0) * 1.5)


def apply_shadow_augmentation(X, intensity=0.3):
    """Simulate shadows by darkening random regions."""
    X_aug = X.copy()
    n_samples = X_aug.shape[0]
    shadow_mask = np.random.random(n_samples) < 0.4
    X_aug[shadow_mask] = X_aug[shadow_mask] * (1 - intensity)
    return X_aug


def evaluate_method(X_subset, y, method_name, n_splits=5):
    """
    Evaluate a feature subset with 5-fold CV.
    Returns dict with all metrics and aggregated confusion matrix.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    fold_metrics = []
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_subset, y)):
        X_tr, X_te = X_subset[tr_idx], X_subset[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        
        clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
        clf.fit(X_tr_scaled, y_tr)
        
        y_pred = clf.predict(X_te_scaled)
        y_prob = clf.predict_proba(X_te_scaled)[:, 1]
        
        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        
        fold_metrics.append({
            'accuracy': accuracy_score(y_te, y_pred),
            'f1': f1_score(y_te, y_pred, average='weighted'),
            'precision': precision_score(y_te, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_te, y_pred, average='weighted', zero_division=0),
        })
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)
    
    cm = confusion_matrix(all_y_true, all_y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    miou = compute_miou(all_y_true, all_y_pred)
    
    fpr_curve, tpr_curve, _ = roc_curve(all_y_true, all_y_prob)
    roc_auc = auc(fpr_curve, tpr_curve)
    
    return {
        'method': method_name,
        'accuracy_mean': np.mean([m['accuracy'] for m in fold_metrics]),
        'accuracy_std': np.std([m['accuracy'] for m in fold_metrics]),
        'f1_mean': np.mean([m['f1'] for m in fold_metrics]),
        'f1_std': np.std([m['f1'] for m in fold_metrics]),
        'precision_mean': np.mean([m['precision'] for m in fold_metrics]),
        'recall_mean': np.mean([m['recall'] for m in fold_metrics]),
        'fpr': fpr_val,
        'miou': miou,
        'auc': roc_auc,
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }


def evaluate_with_augmentation(X_subset, y, method_name, aug_func, aug_name, intensity=0.3):
    """Evaluate under augmentation (applied to test set only)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_y_true = []
    all_y_pred = []
    
    for tr_idx, te_idx in skf.split(X_subset, y):
        X_tr, X_te = X_subset[tr_idx], X_subset[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        
        X_te_aug = aug_func(X_te, intensity)
        X_te_scaled = scaler.transform(X_te_aug)
        
        clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        clf.fit(X_tr_scaled, y_tr)
        
        y_pred = clf.predict(X_te_scaled)
        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred)
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    cm = confusion_matrix(all_y_true, all_y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'method': method_name,
        'augmentation': aug_name,
        'intensity': intensity,
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'f1': f1_score(all_y_true, all_y_pred, average='weighted'),
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'miou': compute_miou(all_y_true, all_y_pred),
    }


def plot_confusion_matrices(results, output_path):
    """Plot confusion matrices for all methods."""
    n_methods = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, res in enumerate(results):
        if idx >= len(axes):
            break
        ax = axes[idx]
        cm = res['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Post-Def', 'Pre-Def'],
                    yticklabels=['Post-Def', 'Pre-Def'])
        ax.set_title(f"{res['method']}\nAcc={res['accuracy_mean']:.4f}, F1={res['f1_mean']:.4f}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices - 5-Fold Cross Validation (Aggregated)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_augmentation_comparison(aug_results, output_path):
    """Plot performance under different augmentations."""
    methods = list(set(r['method'] for r in aug_results))
    augmentations = ['Clean', 'Fog', 'Glare', 'Shadow']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(methods))
    width = 0.2
    
    for ax, metric, title in zip(axes, ['accuracy', 'f1'], ['Accuracy', 'F1 Score']):
        for i, aug in enumerate(augmentations):
            if aug == 'Clean':
                vals = [next((r[metric] for r in aug_results 
                             if r['method'] == m and r['augmentation'] == 'None'), 0) 
                       for m in methods]
            else:
                vals = [next((r[metric] for r in aug_results 
                             if r['method'] == m and r['augmentation'] == aug), 0) 
                       for m in methods]
            ax.bar(x + i * width, vals, width, label=aug)
        
        ax.set_xlabel('Method')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Under Different Conditions')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 70)
    print("  COMPLETE EVALUATION PIPELINE - QUANTUM FEATURE SELECTION")
    print("=" * 70)
    
    print("\n[1/5] Loading data...")
    X, y, cols = load_data(CSV_PATH)
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Classes: Post_Def={np.sum(y==0)}, Pre_Def={np.sum(y==1)}")
    
    print("\n[2/5] Preparing feature subsets...")
    
    mi_4_idx, mi_4_names = select_mi_features(X, y, cols, k=4)
    print(f"  MI-4 features: {mi_4_names}")
    
    methods = []
    
    methods.append(('All-14', list(range(X.shape[1])), cols))
    methods.append(('MI-4', mi_4_idx, mi_4_names))
    
    for k, feat_names in QFS_SUBSETS.items():
        idx = get_feature_indices(cols, feat_names)
        methods.append((f'QFS-{k}', idx, feat_names))
    
    print("\n[3/5] Running 5-fold CV evaluation...")
    results = []
    
    for name, idx, feat_names in methods:
        print(f"  Evaluating {name}...", end=" ", flush=True)
        X_sub = X[:, idx]
        res = evaluate_method(X_sub, y, name)
        res['n_features'] = len(idx)
        res['features'] = feat_names
        results.append(res)
        print(f"Acc={res['accuracy_mean']:.4f}, F1={res['f1_mean']:.4f}, "
              f"FPR={res['fpr']:.4f}, mIoU={res['miou']:.4f}")
    
    print("\n[4/5] Running augmentation tests...")
    aug_results = []
    
    augmentations = [
        ('None', None, 0.0),
        ('Fog', apply_fog_augmentation, 0.3),
        ('Glare', apply_glare_augmentation, 0.3),
        ('Shadow', apply_shadow_augmentation, 0.3),
    ]
    
    for name, idx, feat_names in methods:
        X_sub = X[:, idx]
        for aug_name, aug_func, intensity in augmentations:
            if aug_func is None:
                res = evaluate_method(X_sub, y, name)
                aug_results.append({
                    'method': name,
                    'augmentation': 'None',
                    'intensity': 0.0,
                    'accuracy': res['accuracy_mean'],
                    'f1': res['f1_mean'],
                    'fpr': res['fpr'],
                    'miou': res['miou'],
                })
            else:
                res = evaluate_with_augmentation(X_sub, y, name, aug_func, aug_name, intensity)
                aug_results.append(res)
        print(f"  {name}: Clean={aug_results[-4]['accuracy']:.4f}, "
              f"Fog={aug_results[-3]['accuracy']:.4f}, "
              f"Glare={aug_results[-2]['accuracy']:.4f}, "
              f"Shadow={aug_results[-1]['accuracy']:.4f}")
    
    print("\n[5/5] Saving results...")
    
    out_csv = os.path.join(HERE, 'evaluation_results.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'N_Features', 'Accuracy', 'Acc_Std', 'F1', 'F1_Std',
                        'Precision', 'Recall', 'FPR', 'mIoU', 'AUC', 'TP', 'TN', 'FP', 'FN', 'Features'])
        for r in results:
            feat_str = '; '.join(r['features']) if isinstance(r['features'], list) else str(r['features'])
            writer.writerow([
                r['method'], r['n_features'],
                f"{r['accuracy_mean']:.4f}", f"{r['accuracy_std']:.4f}",
                f"{r['f1_mean']:.4f}", f"{r['f1_std']:.4f}",
                f"{r['precision_mean']:.4f}", f"{r['recall_mean']:.4f}",
                f"{r['fpr']:.4f}", f"{r['miou']:.4f}", f"{r['auc']:.4f}",
                r['tp'], r['tn'], r['fp'], r['fn'],
                feat_str
            ])
    print(f"  Saved: {out_csv}")
    
    aug_csv = os.path.join(HERE, 'augmentation_results.csv')
    with open(aug_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Augmentation', 'Intensity', 'Accuracy', 'F1', 'FPR', 'mIoU'])
        for r in aug_results:
            writer.writerow([
                r['method'], r['augmentation'], r['intensity'],
                f"{r['accuracy']:.4f}", f"{r['f1']:.4f}",
                f"{r['fpr']:.4f}", f"{r['miou']:.4f}"
            ])
    print(f"  Saved: {aug_csv}")
    
    cm_txt = os.path.join(HERE, 'confusion_matrices.txt')
    with open(cm_txt, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  CONFUSION MATRICES - 5-FOLD CROSS VALIDATION\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(f"Method: {r['method']} ({r['n_features']} features)\n")
            f.write("-" * 40 + "\n")
            f.write(f"                 Predicted\n")
            f.write(f"              Post-Def  Pre-Def\n")
            f.write(f"Actual Post-Def  {r['tn']:5d}    {r['fp']:5d}\n")
            f.write(f"       Pre-Def   {r['fn']:5d}    {r['tp']:5d}\n")
            f.write(f"\nMetrics:\n")
            f.write(f"  Accuracy:  {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}\n")
            f.write(f"  F1 Score:  {r['f1_mean']:.4f} ± {r['f1_std']:.4f}\n")
            f.write(f"  Precision: {r['precision_mean']:.4f}\n")
            f.write(f"  Recall:    {r['recall_mean']:.4f}\n")
            f.write(f"  FPR:       {r['fpr']:.4f}\n")
            f.write(f"  mIoU:      {r['miou']:.4f}\n")
            f.write(f"  AUC:       {r['auc']:.4f}\n")
            f.write(f"  Features:  {r['features']}\n")
            f.write("\n")
    print(f"  Saved: {cm_txt}")
    
    cm_png = os.path.join(HERE, 'confusion_matrices.png')
    plot_confusion_matrices(results, cm_png)
    
    aug_png = os.path.join(HERE, 'augmentation_comparison.png')
    plot_augmentation_comparison(aug_results, aug_png)
    
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<12} {'#Feat':>6} {'Acc':>8} {'F1':>8} {'FPR':>8} {'mIoU':>8} {'AUC':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['method']:<12} {r['n_features']:>6} {r['accuracy_mean']:>8.4f} "
              f"{r['f1_mean']:>8.4f} {r['fpr']:>8.4f} {r['miou']:>8.4f} {r['auc']:>8.4f}")
    
    print("\n" + "=" * 70)
    print("  AUGMENTATION ROBUSTNESS SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<12} {'Clean':>10} {'Fog':>10} {'Glare':>10} {'Shadow':>10}")
    print("-" * 70)
    for name, _, _ in methods:
        clean = next((r['accuracy'] for r in aug_results if r['method'] == name and r['augmentation'] == 'None'), 0)
        fog = next((r['accuracy'] for r in aug_results if r['method'] == name and r['augmentation'] == 'Fog'), 0)
        glare = next((r['accuracy'] for r in aug_results if r['method'] == name and r['augmentation'] == 'Glare'), 0)
        shadow = next((r['accuracy'] for r in aug_results if r['method'] == name and r['augmentation'] == 'Shadow'), 0)
        print(f"{name:<12} {clean:>10.4f} {fog:>10.4f} {glare:>10.4f} {shadow:>10.4f}")
    
    print("\nDone! Check evaluation_results.csv and augmentation_results.csv for full details.")


if __name__ == '__main__':
    main()
