#!/usr/bin/env python3
"""
Publication-Quality Architecture Diagram for NeurIPS/CVPR/ICML
===============================================================
Hybrid Quantum-Classical Feature Selection for UAV-Based Cotton Defoliation

Design principles from top venues:
- Clean white background with minimal color palette
- Clear left-to-right information flow
- Real data samples as input visualization
- Mathematical notation where appropriate
- Section labels at bottom (a), (b), (c), (d), (e)
- No unnecessary decoration or gradients
- Vector-quality output at 300 DPI

Version: Final polished version
"""

import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import cv2

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
ICML_ROOT = "/Volumes/T9/ICML"

PRE_FOLDERS = ["Part_one_pre_def_rgb", "part 2_pre_def_rgb"]
POST_FOLDERS = ["Post_def_rgb_part1", "205_Post_Def_rgb", "part3_post_def_rgb"]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "paper_figures", "architecture_final.png")

# ══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE - Professional, muted colors inspired by top venues
# ══════════════════════════════════════════════════════════════════════════════
WHITE = "#FFFFFF"
BG = "#FFFFFF"

# Section colors
BLUE = "#3B82F6"
BLUE_LIGHT = "#EFF6FF"
BLUE_DARK = "#1D4ED8"

PURPLE = "#8B5CF6"
PURPLE_LIGHT = "#F5F3FF"
PURPLE_DARK = "#6D28D9"

GREEN = "#10B981"
GREEN_LIGHT = "#ECFDF5"
GREEN_DARK = "#059669"

ORANGE = "#F59E0B"
ORANGE_LIGHT = "#FFFBEB"
ORANGE_DARK = "#D97706"

# Neutrals
GRAY_900 = "#111827"
GRAY_700 = "#374151"
GRAY_600 = "#4B5563"
GRAY_500 = "#6B7280"
GRAY_400 = "#9CA3AF"
GRAY_300 = "#D1D5DB"
GRAY_200 = "#E5E7EB"
GRAY_100 = "#F3F4F6"

# Class colors
PRE_COLOR = "#22C55E"
POST_COLOR = "#EF4444"

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_image(folder, idx=0, size=(160, 120)):
    """Load and crop an image from the dataset."""
    folder_path = os.path.join(ICML_ROOT, folder)
    if not os.path.exists(folder_path):
        return np.ones((*size[::-1], 3), dtype=np.uint8) * 220
    
    # Filter out macOS metadata files (._*)
    files = sorted([f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                   and not f.startswith('._')])
    if not files or idx >= len(files):
        return np.ones((*size[::-1], 3), dtype=np.uint8) * 220
    
    img_path = os.path.join(folder_path, files[min(idx, len(files)-1)])
    img = cv2.imread(img_path)
    if img is None:
        return np.ones((*size[::-1], 3), dtype=np.uint8) * 220
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    target_ratio = size[0] / size[1]
    current_ratio = w / h
    
    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        start = (w - new_w) // 2
        img = img[:, start:start+new_w]
    else:
        new_h = int(w / target_ratio)
        start = (h - new_h) // 2
        img = img[start:start+new_h, :]
    
    img = cv2.resize(img, size)
    return img


def box(ax, x, y, w, h, fc=GRAY_100, ec=GRAY_400, lw=1.0, radius=0.015, zorder=2):
    """Draw a rounded rectangle."""
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=zorder, transform=ax.transData
    )
    ax.add_patch(b)
    return b


def txt(ax, x, y, s, color=GRAY_900, fs=9, fw="normal", ha="center", va="center", 
        zorder=5, style="normal"):
    """Add text."""
    return ax.text(x, y, s, ha=ha, va=va, color=color, fontsize=fs,
                   fontweight=fw, fontstyle=style, zorder=zorder,
                   multialignment="center")


def arrow(ax, x1, y1, x2, y2, color=GRAY_500, lw=1.5):
    """Draw an arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                               mutation_scale=14),
                zorder=6)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def build():
    """Build the architecture diagram."""
    
    fig = plt.figure(figsize=(16, 7.5), facecolor=BG, dpi=100)
    
    # Main axes
    ax = fig.add_axes([0.0, 0.10, 1.0, 0.82])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.5)
    ax.set_facecolor(BG)
    ax.axis("off")
    
    # ══════════════════════════════════════════════════════════════════════════
    # (a) INPUT IMAGES
    # ══════════════════════════════════════════════════════════════════════════
    
    # Load images
    pre_img = load_image(PRE_FOLDERS[0], idx=2, size=(130, 95))
    post_img = load_image(POST_FOLDERS[0], idx=5, size=(130, 95))
    
    # Pre-defoliation image
    ax_pre = fig.add_axes([0.015, 0.52, 0.075, 0.28])
    ax_pre.imshow(pre_img)
    ax_pre.axis("off")
    for sp in ax_pre.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(PRE_COLOR)
        sp.set_linewidth(2.5)
    
    # Post-defoliation image  
    ax_post = fig.add_axes([0.015, 0.18, 0.075, 0.28])
    ax_post.imshow(post_img)
    ax_post.axis("off")
    for sp in ax_post.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(POST_COLOR)
        sp.set_linewidth(2.5)
    
    # Labels
    txt(ax, 0.95, 5.55, "Pre-Defoliation", color=PRE_COLOR, fs=8, fw="bold")
    txt(ax, 0.95, 2.45, "Post-Defoliation", color=POST_COLOR, fs=8, fw="bold")
    
    # Dataset info
    txt(ax, 0.95, 1.35, "1,549 images\n6 flights", color=GRAY_500, fs=7.5)
    
    # Connecting bracket
    ax.annotate("", xy=(1.65, 4.0), xytext=(1.65, 5.2),
                arrowprops=dict(arrowstyle="-", color=GRAY_400, lw=1.2))
    ax.annotate("", xy=(1.65, 4.0), xytext=(1.65, 2.8),
                arrowprops=dict(arrowstyle="-", color=GRAY_400, lw=1.2))
    ax.plot([1.65, 1.85], [4.0, 4.0], color=GRAY_400, lw=1.2)
    
    # ══════════════════════════════════════════════════════════════════════════
    # ARROW: Input → Feature Extraction
    # ══════════════════════════════════════════════════════════════════════════
    arrow(ax, 1.85, 4.0, 2.35, 4.0, color=GRAY_500)
    
    # ══════════════════════════════════════════════════════════════════════════
    # (b) FEATURE EXTRACTION
    # ══════════════════════════════════════════════════════════════════════════
    
    box(ax, 2.35, 0.9, 2.9, 5.2, fc=BLUE_LIGHT, ec=BLUE, lw=1.5, radius=0.02)
    txt(ax, 3.8, 5.9, "Feature Extraction", color=BLUE, fs=10, fw="bold")
    
    # Color features
    box(ax, 2.5, 3.4, 2.6, 2.5, fc=WHITE, ec=BLUE, lw=1.0, radius=0.015)
    txt(ax, 3.8, 5.7, "Color Indices (7)", color=BLUE, fs=9, fw="bold")
    
    color_feats = [
        ("ExG", "Excess Green"),
        ("σ(ExG)", "ExG Std Dev ★"),
        ("RBR", "Red-Blue Ratio ★"),
        ("NGRDI", "Green-Red Diff"),
        ("R,G,B", "Channel Means ★"),
    ]
    for i, (abbr, desc) in enumerate(color_feats):
        yy = 5.35 - i * 0.40
        box(ax, 2.6, yy-0.14, 0.65, 0.30, fc=BLUE_LIGHT, ec=BLUE, lw=0.7, radius=0.01)
        txt(ax, 2.92, yy, abbr, color=BLUE_DARK, fs=7.5, fw="bold")
        star = "★" in desc
        txt(ax, 4.1, yy, desc.replace(" ★", ""), color=BLUE_DARK if star else GRAY_600, 
            fs=7, ha="left", fw="bold" if star else "normal")
    
    # Texture features
    box(ax, 2.5, 1.1, 2.6, 2.1, fc=WHITE, ec=BLUE, lw=1.0, radius=0.015)
    txt(ax, 3.8, 3.0, "GLCM Texture (5)", color=BLUE, fs=9, fw="bold")
    
    tex_feats = [
        ("H", "Entropy"),
        ("Con", "Contrast"),
        ("Hom", "Homogeneity"),
        ("Cor", "Correlation ★"),
        ("E", "Energy"),
    ]
    for i, (abbr, desc) in enumerate(tex_feats):
        yy = 2.65 - i * 0.33
        box(ax, 2.6, yy-0.12, 0.5, 0.26, fc=BLUE_LIGHT, ec=BLUE, lw=0.7, radius=0.01)
        txt(ax, 2.85, yy, abbr, color=BLUE_DARK, fs=7, fw="bold")
        star = "★" in desc
        txt(ax, 3.6, yy, desc.replace(" ★", ""), color=BLUE_DARK if star else GRAY_600,
            fs=7, ha="left", fw="bold" if star else "normal")
    
    # Output
    box(ax, 2.7, 0.95, 2.2, 0.32, fc=BLUE, ec=BLUE, lw=0, radius=0.012)
    txt(ax, 3.8, 1.11, "12-dim vector", color=WHITE, fs=8, fw="bold")
    
    # ══════════════════════════════════════════════════════════════════════════
    # ARROW: Feature Extraction → MI Filter
    # ══════════════════════════════════════════════════════════════════════════
    arrow(ax, 5.25, 4.0, 5.75, 4.0, color=GRAY_500)
    txt(ax, 5.5, 4.22, "×1549", color=GRAY_500, fs=7)
    
    # ══════════════════════════════════════════════════════════════════════════
    # (c) MI PRE-FILTER
    # ══════════════════════════════════════════════════════════════════════════
    
    box(ax, 5.75, 0.9, 2.3, 5.2, fc=ORANGE_LIGHT, ec=ORANGE, lw=1.5, radius=0.02)
    txt(ax, 6.9, 5.9, "MI Pre-Filter", color=ORANGE_DARK, fs=10, fw="bold")
    
    # MI formula
    box(ax, 5.9, 4.55, 2.0, 1.25, fc=WHITE, ec=ORANGE, lw=1.0, radius=0.015)
    txt(ax, 6.9, 5.6, "Mutual Information", color=ORANGE_DARK, fs=9, fw="bold")
    txt(ax, 6.9, 5.28, "SelectKBest (k=6)", color=GRAY_600, fs=8)
    box(ax, 6.0, 4.68, 1.8, 0.38, fc=ORANGE_LIGHT, ec=ORANGE, lw=0.7, radius=0.01)
    txt(ax, 6.9, 4.87, "I(X;Y) = H(Y)−H(Y|X)", color=GRAY_900, fs=7.5, style="italic")
    
    # Top-6 list
    box(ax, 5.9, 2.3, 2.0, 2.05, fc=WHITE, ec=ORANGE, lw=1.0, radius=0.015)
    txt(ax, 6.9, 4.15, "Top-6 Features", color=ORANGE_DARK, fs=9, fw="bold")
    
    top6 = [
        ("1.", "Std_ExG", True),
        ("2.", "Mean_ExG", False),
        ("3.", "Mean_RBR", True),
        ("4.", "Mean_NGRDI", False),
        ("5.", "Mean_B", True),
        ("6.", "Correlation", True),
    ]
    for i, (num, feat, sel) in enumerate(top6):
        yy = 3.85 - i * 0.27
        clr = ORANGE_DARK if sel else GRAY_500
        fw = "bold" if sel else "normal"
        txt(ax, 6.15, yy, num, color=GRAY_500, fs=7, ha="right")
        txt(ax, 6.25, yy, feat, color=clr, fs=7.5, fw=fw, ha="left")
        if sel:
            txt(ax, 7.7, yy, "★", color=ORANGE, fs=7)
    
    # Rationale
    box(ax, 5.9, 1.0, 2.0, 1.1, fc=WHITE, ec=ORANGE, lw=1.0, radius=0.015)
    txt(ax, 6.9, 1.9, "Complexity", color=ORANGE_DARK, fs=8, fw="bold")
    txt(ax, 6.9, 1.55, "VQC: O(2ⁿ) qubits\n12 → 6 features", color=GRAY_600, fs=7.5)
    
    # Output
    box(ax, 6.05, 0.95, 1.7, 0.32, fc=ORANGE, ec=ORANGE, lw=0, radius=0.012)
    txt(ax, 6.9, 1.11, "6-dim", color=WHITE, fs=8, fw="bold")
    
    # ══════════════════════════════════════════════════════════════════════════
    # ARROW: MI Filter → VQC
    # ══════════════════════════════════════════════════════════════════════════
    arrow(ax, 8.05, 4.0, 8.55, 4.0, color=GRAY_500)
    
    # ══════════════════════════════════════════════════════════════════════════
    # (d) VARIATIONAL QUANTUM CLASSIFIER
    # ══════════════════════════════════════════════════════════════════════════
    
    box(ax, 8.55, 0.9, 4.3, 5.2, fc=PURPLE_LIGHT, ec=PURPLE, lw=2.0, radius=0.02)
    txt(ax, 10.7, 5.9, "Variational Quantum Classifier", color=PURPLE_DARK, fs=10, fw="bold")
    
    # Wrapper search header
    box(ax, 8.7, 5.35, 4.0, 0.45, fc=WHITE, ec=PURPLE, lw=1.0, radius=0.012)
    txt(ax, 10.7, 5.58, "Wrapper: C(6,4) = 15 subsets → select best", color=PURPLE_DARK, fs=8.5, fw="bold")
    
    # Quantum circuit
    box(ax, 8.7, 2.55, 4.0, 2.6, fc=WHITE, ec=PURPLE, lw=1.0, radius=0.015)
    txt(ax, 10.7, 4.95, "4-Qubit Quantum Circuit", color=PURPLE_DARK, fs=9, fw="bold")
    
    # Draw circuit
    qubits = ["q₀ Std_ExG", "q₁ Mean_RBR", "q₂ Mean_B", "q₃ Correlation"]
    wire_y = [4.55, 4.05, 3.55, 3.05]
    
    for i, (ql, wy) in enumerate(zip(qubits, wire_y)):
        # Wire
        ax.plot([9.1, 12.4], [wy, wy], color=GRAY_300, lw=1.0, zorder=2)
        # Label
        txt(ax, 8.95, wy, ql, color=PURPLE_DARK, fs=6.5, fw="bold", ha="right")
        
        # H gate
        box(ax, 9.2, wy-0.13, 0.32, 0.26, fc=BLUE_LIGHT, ec=BLUE, lw=0.8, radius=0.008, zorder=4)
        txt(ax, 9.36, wy, "H", color=BLUE, fs=7, fw="bold")
        
        # Rz gate
        box(ax, 9.65, wy-0.13, 0.35, 0.26, fc=BLUE_LIGHT, ec=BLUE, lw=0.8, radius=0.008, zorder=4)
        txt(ax, 9.82, wy, "Rz", color=BLUE, fs=7, fw="bold")
        
        # Ry gate
        box(ax, 11.0, wy-0.13, 0.35, 0.26, fc=PURPLE_LIGHT, ec=PURPLE, lw=0.8, radius=0.008, zorder=4)
        txt(ax, 11.17, wy, "Ry", color=PURPLE, fs=7, fw="bold")
        
        # Measure
        box(ax, 11.9, wy-0.13, 0.35, 0.26, fc=GREEN_LIGHT, ec=GREEN, lw=0.8, radius=0.008, zorder=4)
        txt(ax, 12.07, wy, "M", color=GREEN, fs=7, fw="bold")
    
    # CNOT gates
    for i in range(3):
        cy, ty = wire_y[i], wire_y[i+1]
        ax.plot([10.35, 10.35], [ty+0.13, cy-0.13], color=BLUE, lw=0.8, zorder=3)
        ax.plot(10.35, cy, 'o', color=BLUE, markersize=4, zorder=4)
        box(ax, 10.2, ty-0.1, 0.30, 0.20, fc=BLUE_LIGHT, ec=BLUE, lw=0.8, radius=0.006, zorder=4)
        txt(ax, 10.35, ty, "⊕", color=BLUE, fs=8, fw="bold")
    
    # Gate labels
    for gx, gl in [(9.36, "H"), (9.82, "ZZ"), (10.35, "CX"), (11.17, "Ry"), (12.07, "M")]:
        txt(ax, gx, 2.72, gl, color=GRAY_500, fs=6, style="italic")
    
    # Optimizer
    box(ax, 8.7, 2.45, 4.0, 0.30, fc=PURPLE_LIGHT, ec=PURPLE, lw=0.6, radius=0.01)
    txt(ax, 10.7, 2.60, "COBYLA · StatevectorSampler · 40 iter", color=PURPLE, fs=7)
    
    # Result
    box(ax, 8.7, 1.35, 4.0, 0.9, fc=WHITE, ec=PURPLE, lw=1.0, radius=0.015)
    txt(ax, 10.7, 2.0, "Best Subset (VQC Acc: 72%)", color=PURPLE_DARK, fs=9, fw="bold")
    txt(ax, 10.7, 1.62, "Std_ExG · Mean_RBR · Mean_B · Correlation", color=GRAY_900, fs=8.5, fw="bold")
    
    # Output
    box(ax, 8.9, 0.95, 3.6, 0.32, fc=PURPLE, ec=PURPLE, lw=0, radius=0.012)
    txt(ax, 10.7, 1.11, "4 quantum-selected · Low redundancy", color=WHITE, fs=8, fw="bold")
    
    # ══════════════════════════════════════════════════════════════════════════
    # ARROW: VQC → Inference
    # ══════════════════════════════════════════════════════════════════════════
    arrow(ax, 12.85, 4.0, 13.35, 4.0, color=GRAY_500)
    txt(ax, 13.1, 4.22, "4-dim", color=GRAY_500, fs=7)
    
    # ══════════════════════════════════════════════════════════════════════════
    # (e) CLASSICAL INFERENCE
    # ══════════════════════════════════════════════════════════════════════════
    
    box(ax, 13.35, 0.9, 2.4, 5.2, fc=GREEN_LIGHT, ec=GREEN, lw=1.5, radius=0.02)
    txt(ax, 14.55, 5.9, "Classical Inference", color=GREEN_DARK, fs=10, fw="bold")
    
    # SVM
    box(ax, 13.5, 3.9, 2.1, 1.9, fc=WHITE, ec=GREEN, lw=1.0, radius=0.015)
    txt(ax, 14.55, 5.6, "SVM-RBF", color=GREEN_DARK, fs=9, fw="bold")
    txt(ax, 14.55, 5.28, "C=10, γ=scale", color=GRAY_600, fs=7.5, style="italic")
    
    # Accuracy
    box(ax, 13.65, 4.4, 1.8, 0.65, fc=GREEN_LIGHT, ec=GREEN, lw=0.8, radius=0.012)
    txt(ax, 14.55, 4.88, "5-Fold CV", color=GRAY_600, fs=7)
    txt(ax, 14.55, 4.55, "100.0%", color=GREEN_DARK, fs=14, fw="bold")
    
    txt(ax, 14.55, 4.1, "± 0.0 std", color=GRAY_500, fs=7.5)
    
    # Output classes
    box(ax, 13.5, 2.25, 2.1, 1.45, fc=WHITE, ec=GREEN, lw=1.0, radius=0.015)
    txt(ax, 14.55, 3.5, "Output", color=GREEN_DARK, fs=9, fw="bold")
    
    box(ax, 13.65, 2.9, 1.8, 0.38, fc=GREEN_LIGHT, ec=PRE_COLOR, lw=0.8, radius=0.01)
    txt(ax, 14.55, 3.09, "Pre-Defoliation", color=PRE_COLOR, fs=8, fw="bold")
    
    box(ax, 13.65, 2.4, 1.8, 0.38, fc="#FEF2F2", ec=POST_COLOR, lw=0.8, radius=0.01)
    txt(ax, 14.55, 2.59, "Post-Defoliation", color=POST_COLOR, fs=8, fw="bold")
    
    # Key insight
    box(ax, 13.5, 1.0, 2.1, 1.05, fc=WHITE, ec=GREEN, lw=1.0, radius=0.015)
    txt(ax, 14.55, 1.85, "Key Insight", color=GREEN_DARK, fs=8, fw="bold")
    txt(ax, 14.55, 1.45, "Quantum: training\nClassical: inference", color=GRAY_600, fs=7.5)
    
    # Output
    box(ax, 13.65, 0.95, 1.8, 0.32, fc=GREEN, ec=GREEN, lw=0, radius=0.012)
    txt(ax, 14.55, 1.11, "Edge-deployable", color=WHITE, fs=8, fw="bold")
    
    # ══════════════════════════════════════════════════════════════════════════
    # SECTION LABELS
    # ══════════════════════════════════════════════════════════════════════════
    
    ax_bot = fig.add_axes([0.0, 0.0, 1.0, 0.09])
    ax_bot.set_xlim(0, 16)
    ax_bot.set_ylim(0, 1)
    ax_bot.set_facecolor(BG)
    ax_bot.axis("off")
    
    sections = [
        (0.95, "(a) UAV Input", GRAY_700),
        (3.8, "(b) Feature\nExtraction", BLUE),
        (6.9, "(c) MI\nPre-Filter", ORANGE_DARK),
        (10.7, "(d) Quantum Feature\nSelection", PURPLE_DARK),
        (14.55, "(e) Classical\nInference", GREEN_DARK),
    ]
    
    dividers = [1.9, 5.5, 8.3, 13.1]
    for dv in dividers:
        ax_bot.axvline(dv, color=GRAY_300, lw=1.0, ls=':', ymin=0.1, ymax=0.9)
    
    for sx, slbl, sc in sections:
        ax_bot.text(sx/16, 0.5, slbl, ha="center", va="center", color=sc,
                   fontsize=9, fontweight="bold", multialignment="center")
    
    # ══════════════════════════════════════════════════════════════════════════
    # TITLE
    # ══════════════════════════════════════════════════════════════════════════
    
    ax_top = fig.add_axes([0.0, 0.92, 1.0, 0.08])
    ax_top.set_facecolor(BG)
    ax_top.axis("off")
    
    ax_top.text(0.5, 0.65,
                "Hybrid Quantum-Classical Feature Selection for UAV-Based Cotton Defoliation Monitoring",
                ha="center", va="center", fontsize=12, fontweight="bold", color=GRAY_900)
    
    ax_top.text(0.5, 0.15,
                "VQC identifies minimal sufficient features  •  No annotation required  •  Classical-only deployment",
                ha="center", va="center", fontsize=9, color=GRAY_500, style="italic")
    
    # ══════════════════════════════════════════════════════════════════════════
    # LEGEND
    # ══════════════════════════════════════════════════════════════════════════
    
    legend = [
        (BLUE, "Classical"),
        (ORANGE, "MI filter"),
        (PURPLE, "Quantum"),
        (GREEN, "Inference"),
    ]
    
    lx, ly = 14.8, 6.3
    for i, (lc, ll) in enumerate(legend):
        box(ax, lx, ly - i*0.25, 0.18, 0.16, fc=lc, ec=lc, lw=0, radius=0.006, zorder=8)
        txt(ax, lx + 0.25, ly - i*0.25 + 0.08, ll, color=GRAY_600, fs=7, ha="left")
    
    txt(ax, lx + 0.09, ly - 4*0.25 + 0.08, "★", color=ORANGE, fs=8)
    txt(ax, lx + 0.25, ly - 4*0.25 + 0.08, "Selected", color=GRAY_600, fs=7, ha="left")
    
    # ══════════════════════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════════════════════
    
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=BG, pad_inches=0.05)
    print(f"✅ Saved: {OUT_PNG}")
    
    pdf_path = OUT_PNG.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches="tight", facecolor=BG, pad_inches=0.05, format='pdf')
    print(f"✅ Saved: {pdf_path}")
    
    plt.close()


if __name__ == "__main__":
    build()
