#!/usr/bin/env python3
"""
Paper Figures Generator
Produces all 7 publication-quality figures for the IEEE TGRS paper.
Run AFTER all other experiments have completed.
Output: paper_figures/ directory with all PNGs
"""
import csv, os, warnings
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'paper_figures')
os.makedirs(OUT_DIR, exist_ok=True)

QUANTUM_FEATURES = ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation']
ALL_FEATURES     = ['Mean_ExG','Std_ExG','Mean_RBR','Mean_NGRDI',
                    'Mean_R','Mean_G','Mean_B','Entropy',
                    'Contrast','Homogeneity','Energy','Correlation']
COLORS = {'Post_Defoliation':'#D32F2F', 'Pre_Defoliation':'#388E3C'}

# ── data loader ───────────────────────────────────────────────────────────────
def load_full(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    feat_cols = [c for c in rows[0] if c not in ('Filename','Folder','Label')]
    X      = np.array([[float(r[c]) for c in feat_cols] for r in rows])
    y      = np.array([0 if r['Label']=='Post_Defoliation' else 1 for r in rows])
    labels = [r['Label'] for r in rows]
    mask   = np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask], feat_cols, [labels[i] for i in range(len(labels)) if mask[i]]

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Pipeline Diagram (text-based via matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis('off')

    stages = [
        (1.2,  2.0, '📷  UAV Images\n(1,549 JPGs)',          '#1A237E', 'white'),
        (4.0,  2.0, '⚙️  Classical\nFeature Extraction\n12 features/image', '#1B5E20', 'white'),
        (7.0,  2.0, '🔢  MI Pre-Filter\nTop 6 candidates',   '#4A148C', 'white'),
        (10.0, 2.0, '⚛  VQC Wrapper\n4-qubit quantum\nselection',  '#B71C1C', 'white'),
        (12.8, 2.0, '✅  4 Features\n(classical deploy)',     '#E65100', 'white'),
    ]
    for x, y, txt, bg, fg in stages:
        ax.add_patch(mpatches.FancyBboxPatch((x-1.05, y-0.85), 2.1, 1.7,
                     boxstyle="round,pad=0.1", facecolor=bg, edgecolor='white', linewidth=2))
        ax.text(x, y, txt, ha='center', va='center', color=fg,
                fontsize=9, fontweight='bold', multialignment='center')

    for x in [2.35, 5.2, 8.15, 11.1]:
        ax.annotate('', xy=(x+0.4, 2.0), xytext=(x, 2.0),
                    arrowprops=dict(arrowstyle='->', color='#455A64', lw=2.5))

    ax.text(7.0, 0.3, '🔵 CLASSICAL  ←→  QUANTUM (training only)  ←→  CLASSICAL  🔵',
            ha='center', va='center', fontsize=10, color='#37474F', style='italic')
    ax.set_title('Hybrid Quantum-Classical Pipeline for Cotton Defoliation Feature Discovery',
                 fontsize=13, pad=10)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_pipeline.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close(); print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Feature Distribution Violin Plots
# ─────────────────────────────────────────────────────────────────────────────
def fig2_distributions(X, y, cols):
    feat_show = [f for f in QUANTUM_FEATURES if f in cols]
    fig, axes = plt.subplots(1, len(feat_show), figsize=(14, 4), sharey=False)
    for ax, feat in zip(axes, feat_show):
        idx = cols.index(feat)
        pre_vals  = X[y==1, idx]
        post_vals = X[y==0, idx]
        vp = ax.violinplot([pre_vals, post_vals], positions=[1,2], showmedians=True, showextrema=True)
        for pc, col in zip(vp['bodies'], ['#388E3C','#D32F2F']):
            pc.set_facecolor(col); pc.set_alpha(0.7)
        vp['cmedians'].set_color('white'); vp['cmedians'].set_linewidth(2)
        ax.set_xticks([1,2]); ax.set_xticklabels(['Pre-Def','Post-Def'], fontsize=10)
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Distribution of Quantum-Selected Features\n(Pre vs Post Defoliation)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_distributions.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(); print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Correlation Heatmap (load from existing if available)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_correlation(X, cols):
    corr = np.corrcoef(X.T)
    q_idx = [cols.index(f) for f in QUANTUM_FEATURES if f in cols]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.abs(corr), cmap='RdYlGn_r', vmin=0, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=9)
    plt.colorbar(im, ax=ax, label='|Pearson r|')
    for i in q_idx:
        for j in q_idx:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                        fill=False, edgecolor='#1565C0', linewidth=2.5))
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f'{abs(corr[i,j]):.2f}', ha='center', va='center',
                    fontsize=7.5, color='black' if abs(corr[i,j])<0.7 else 'white')
    ax.set_title('Feature Correlation Matrix (Blue Border = QFS Subset)', fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_correlation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(); print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Load noise results if available, else regenerate
# ─────────────────────────────────────────────────────────────────────────────
def fig4_noise():
    src = os.path.join(HERE, 'noise_robustness.png')
    if os.path.exists(src):
        import shutil
        shutil.copy(src, os.path.join(OUT_DIR, 'fig4_noise_robustness.png'))
        print(f"  Copied: fig4_noise_robustness.png")
    else:
        print("  fig4: run run_noise_robustness.py first")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Load cross-folder results if available
# ─────────────────────────────────────────────────────────────────────────────
def fig5_crossfolder():
    src = os.path.join(HERE, 'cross_folder_bar.png')
    if os.path.exists(src):
        import shutil
        shutil.copy(src, os.path.join(OUT_DIR, 'fig5_cross_folder.png'))
        print(f"  Copied: fig5_cross_folder.png")
    else:
        print("  fig5: run run_cross_folder_cv.py first")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Load ablation curve if available
# ─────────────────────────────────────────────────────────────────────────────
def fig6_ablation():
    src = os.path.join(HERE, 'ablation_curve.png')
    if os.path.exists(src):
        import shutil
        shutil.copy(src, os.path.join(OUT_DIR, 'fig6_ablation.png'))
        print(f"  Copied: fig6_ablation.png")
    else:
        print("  fig6: run run_subset_ablation.py first")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Quantum Circuit Diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig7_circuit():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14); ax.set_ylim(-0.5, 4.5); ax.axis('off')
    ax.set_facecolor('#0D1117')
    fig.patch.set_facecolor('#0D1117')

    qubit_labels = ['q[0] — Std_ExG', 'q[1] — Mean_RBR', 'q[2] — Mean_B', 'q[3] — Correlation']
    y_positions  = [4, 3, 2, 1]

    for yi, lbl in zip(y_positions, qubit_labels):
        ax.plot([0.5, 13.5], [yi, yi], color='#58A6FF', linewidth=1.5, zorder=1)
        ax.text(0.3, yi, lbl, ha='right', va='center', color='#E6EDF3', fontsize=9)

    def gate(ax, x, ys, label, color='#388BFD', width=0.5):
        for yi in ys:
            rect = plt.Rectangle((x-width/2, yi-0.3), width, 0.6,
                                  facecolor=color, edgecolor='white', linewidth=1.2, zorder=3)
            ax.add_patch(rect)
            ax.text(x, yi, label, ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold', zorder=4)

    def ctrl_gate(ax, x, ctrl_y, tgt_y, color='#F78166'):
        ax.plot([x, x], [ctrl_y, tgt_y], color=color, linewidth=1.5, zorder=2)
        ax.plot(x, ctrl_y, 'o', color=color, markersize=6, zorder=3)
        rect = plt.Rectangle((x-0.3, tgt_y-0.3), 0.6, 0.6,
                              facecolor=color, edgecolor='white', linewidth=1.2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, tgt_y, 'X', ha='center', va='center', color='white', fontsize=9, fontweight='bold', zorder=4)

    # ZZFeatureMap: H then Rz then CNOT pairs
    for i, yi in enumerate(y_positions):
        gate(ax, 1.2, [yi], 'H', '#1F6FEB')
        gate(ax, 2.0, [yi], f'Rz\n(x{i})', '#1F6FEB')

    ctrl_gate(ax, 2.8, 4, 3)
    ctrl_gate(ax, 3.4, 3, 2)
    ctrl_gate(ax, 4.0, 2, 1)

    ax.text(2.6, 4.4, 'ZZFeatureMap (reps=1)', ha='center', color='#79C0FF', fontsize=10, fontweight='bold')
    ax.axvline(x=5.0, color='#3D444D', linewidth=1.5, linestyle='--')

    # RealAmplitudes: Ry then CNOT
    for i, yi in enumerate(y_positions):
        gate(ax, 5.8, [yi], f'Ry\n(θ{i})', '#D2A8FF')

    ctrl_gate(ax, 6.6, 4, 3, '#F78166')
    ctrl_gate(ax, 7.2, 3, 2, '#F78166')
    ctrl_gate(ax, 7.8, 2, 1, '#F78166')

    for i, yi in enumerate(y_positions):
        gate(ax, 8.6, [yi], f'Ry\n(θ{i+4})', '#D2A8FF')

    ax.text(7.2, 4.4, 'RealAmplitudes Ansatz (reps=1)', ha='center', color='#D2A8FF', fontsize=10, fontweight='bold')
    ax.axvline(x=9.6, color='#3D444D', linewidth=1.5, linestyle='--')

    # Measurement
    for yi in y_positions:
        gate(ax, 10.4, [yi], 'M', '#3FB950', width=0.6)

    ax.text(10.4, 4.4, 'Measure', ha='center', color='#3FB950', fontsize=10, fontweight='bold')
    ax.text(7.0, -0.3,
            'Optimizer: COBYLA  |  Sampler: StatevectorSampler  |  Trained once → VQC accuracy used to rank feature subsets',
            ha='center', color='#8B949E', fontsize=9)
    ax.set_title('Variational Quantum Classifier Circuit (4 Qubits)', color='#E6EDF3', fontsize=13, pad=10)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig7_quantum_circuit.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0D1117')
    plt.close(); print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
def main():
    csv_path = os.path.join(HERE, 'icml_features_FULL.csv')
    print("Generating paper figures …")
    X, y, cols, label_strs = load_full(csv_path)

    print("  Fig 1: Pipeline diagram")
    fig1_pipeline()
    print("  Fig 2: Feature distributions")
    fig2_distributions(X, y, cols)
    print("  Fig 3: Correlation heatmap")
    fig3_correlation(X, cols)
    print("  Fig 4: Noise robustness")
    fig4_noise()
    print("  Fig 5: Cross-folder generalization")
    fig5_crossfolder()
    print("  Fig 6: Ablation curve")
    fig6_ablation()
    print("  Fig 7: Quantum circuit")
    fig7_circuit()

    print(f"\n✅ All figures saved to: {OUT_DIR}/")

if __name__ == '__main__':
    main()
