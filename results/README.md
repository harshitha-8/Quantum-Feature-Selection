# Results Directory

This folder contains all publication-ready outputs from the Quantum Feature Selection (QFS) evaluation pipeline.

## Structure

```
results/
├── plots/          ← High-resolution (300 DPI) PNG figures for the paper
├── tables/         ← CSV and text tables with numerical results
└── logs/           ← (future) training logs, verbosity outputs
```

---

## Plots (`results/plots/`)

| File | Description |
|---|---|
| `k_feature_membership.png` | Binary heatmap showing which features are in each QFS/MI subset at k=2,4,6 |
| `k_noise_robustness.png` | Accuracy vs. Gaussian noise σ for all 6 subsets — shows QFS retains performance longer under sensor degradation |
| `k_augmentation_stress.png` | Grouped bar chart comparing Fog / Glare / Shadow robustness per k-value |
| `k_roc_pr_grid.png` | 2×3 grid of ROC and Precision-Recall curves for QFS vs MI at k=2, 4, 6 |
| `k_cv_accuracy_bars.png` | 5-Fold Group-CV accuracy with ±std error bars for all subsets |
| `k_radar_sigma15.png` | Spider chart of 5 metrics (Acc, F1, Prec, Rec, AUC) under σ=0.15 noise |

---

## Tables (`results/tables/`)

| File | Description |
|---|---|
| `clean_cv_results.csv` | Full 5-fold Group-CV results (Accuracy, F1, AUC ± std) on clean data |
| `noise_curve_results.csv` | Per-σ accuracy for each subset across 7 noise levels |
| `augmentation_results.csv` | Accuracy under Fog, Glare, Shadow augmentations per subset |
| `summary_tables.txt` | Human-readable summary of all key results (ready for paper appendix) |

---

## Key Findings

### Why does clean data show 1.0 accuracy?

This is **not overfitting or data leakage**. Investigation of AUC per individual feature shows:

| Feature | AUC (solo) |
|---|---|
| Mean_ExG | **1.0000** |
| Std_ExG | **1.0000** |
| Mean_RBR | 0.9025 |
| Mean_B | 0.9456 |
| Correlation | 0.8553 |

Pre-defoliation fields are **biologically lush green** (ExG ≈ +0.10), while Post-defoliation fields are **brown dried stalks** (ExG ≈ +0.01). The classes are perfectly linearly separable. The interesting scientific contribution is in **what happens under noisy/degraded conditions**, which the noise robustness and augmentation plots capture.

### Evaluation Protocol

All results use **Stratified Group-Fold Cross-Validation** (`StratifiedGroupKFold`, 5 splits) where the `groups` key is the Flight Folder, ensuring images from the same UAV flight are **never split between train and test**. This prevents any spatial data leakage.

---

## Reproducing Results

```bash
cd /Volumes/T9/QuantumFeatureSelection
.venv/bin/python scripts/evaluate_k_qubits.py
```
