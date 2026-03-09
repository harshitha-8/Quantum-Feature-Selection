#!/usr/bin/env python3
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

OUT_PNG = "rigorous_academic_architecture.png"

def build():
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(22, 10), facecolor='#0d1117')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Fonts and Colors
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Inter']
    
    C_BG = '#0d1117'
    C_PANEL = '#161b22'
    C_BORDER = '#30363d'
    C_TEXT = '#c9d1d9'
    C_HL = '#58a6ff'       # Classical Blue
    C_QUANTUM = '#bc8cff'  # Quantum Purple
    C_INPUT = '#3fb950'    # Green Input
    C_ORANGE = '#d29922'   # Warning/Filter Orange

    def box(x, y, w, h, title, text, color, border, fontsize=12):
        b = patches.Rectangle((x, y), w, h, fill=True, color=color, 
                              ec=border, lw=2.5, alpha=0.95, transform=ax.transData)
        ax.add_patch(b)
        ax.text(x + 2, y + h - 3, title, color='white', fontsize=15, fontweight='bold', ha='left', va='top')
        ax.text(x + 2, y + h - 10, text, color=C_TEXT, fontsize=fontsize, ha='left', va='top', linespacing=1.8)

    def draw_arrow(x1, y1, x2, y2, color):
        path = Path([(x1, y1), ((x1+x2)/2, y1), ((x1+x2)/2, y2), (x2, y2)],
                    [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO])
        patch = patches.PathPatch(path, facecolor='none', lw=3, edgecolor=color)
        ax.add_patch(patch)
        ax.plot(x2, y2, marker='>', color=color, markersize=10)

    # 1. Inputs (Pre and Post Defoliation)
    box(3, 60, 18, 25, "Pre-Defoliation", "UAV RGB Modality\nHigh Canopy Density\nTarget: Baseline Leaf Volume", C_PANEL, C_INPUT)
    box(3, 20, 18, 25, "Post-Defoliation", "UAV RGB Modality\nExposed Cotton Bolls\nTarget: Defoliation Efficacy", C_PANEL, C_INPUT)

    # 2. Classical Feature Engineering
    f_text = """Mathematical Formulation:
$X \in \mathbb{R}^{700 \times M}$

Spatial Attributes:
• Gray Level Co-occurrence Matrix
  (GLCM Contrast, Correlation)
• Excess Green Index (ExG)
• Normalized Red-Blue Ratio"""
    box(28, 20, 24, 65, "Classical Feature Engineering Subsystem", f_text, C_PANEL, C_HL, fontsize=13)
    
    draw_arrow(21, 72.5, 28, 72.5, C_HL)
    draw_arrow(21, 32.5, 28, 32.5, C_HL)

    # 3. Dimensionality Reduction (Filter)
    box(58, 60, 15, 25, "Information Filter", "Mutual Information (MI)\n$I(X;Y) = H(Y) - H(Y|X)$\n\nSubset: Top $k=6$ Candidates", C_PANEL, C_ORANGE)
    draw_arrow(52, 72.5, 58, 72.5, C_ORANGE)

    # 4. Quantum VQC Evaluator
    q_text = """Variational Quantum Classifier (VQC)
Minimizing Surrogate Loss function over subsets.

State Preparation: $|0\\rangle^{\otimes 4}$
Data Encoding: ZZFeatureMap $U_{\Phi(\mathbf{x})}$
Ansatz: RealAmplitudes $W(\\theta)$
Measurement: $\\langle Z_i \\rangle$

Output Space: Optimal Combination $X_{opt} \in \mathbb{R}^4$"""
    box(58, 20, 39, 32, "Hybrid Quantum-Classical Feature Dimension Optimization", q_text, C_PANEL, C_QUANTUM, fontsize=12)
    
    draw_arrow(65.5, 60, 65.5, 52, C_QUANTUM)

    # 5. RBF Evaluator (Final Output)
    box(78, 60, 19, 25, "Downstream Inference", "Classical Base Learner\nSVM with RBF Kernel\n$K(x, x') = \\exp(-\gamma||x-x'||^2)$\n\nGenerates Ultimate Phenotypic Score", C_PANEL, C_INPUT)
    draw_arrow(77.5, 52, 77.5, 60, C_INPUT)

    # Section Headers (NeurIPS style)
    ax.text(12, 10, "(a) Data Acquisition Protocol", color=C_INPUT, fontsize=14, fontweight='bold', ha='center')
    ax.text(40, 10, "(b) High-Dimensional Classical Space", color=C_HL, fontsize=14, fontweight='bold', ha='center')
    ax.text(77, 10, "(c) Quantum Combinatorial Space & Inference", color=C_QUANTUM, fontsize=14, fontweight='bold', ha='center')

    # Main Title
    ax.text(50, 93, "Scientific Architecture for Quantum-Enhanced Phenotypic Defoliation Discovery", 
            color='white', fontsize=22, fontweight='bold', ha='center')

    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor=C_BG)
    print(f"Graph generated at: {OUT_PNG}")

if __name__ == "__main__":
    build()
