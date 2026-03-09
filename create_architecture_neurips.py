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
"""

import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from matplotlib.lines import Line2D
import cv2

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
ICML_ROOT = "/Volumes/T9/ICML"

PRE_FOLDERS = [
    "Part_one_pre_def_rgb",
    "part 2_pre_def_rgb",
]
POST_FOLDERS = [
    "Post_def_rgb_part1",
    "205_Post_Def_rgb",
    "part3_post_def_rgb",
    "part4_post_def_rgb",
]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "paper_figures", "architecture_neurips.png")

# ══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE (NeurIPS/CVPR style - muted, professional)
# ══════════════════════════════════════════════════════════════════════════════
WHITE = "#FFFFFF"
BG = "#FFFFFF"

# Primary colors (muted, professional)
BLUE = "#2563EB"        # Classical processing
BLUE_LIGHT = "#DBEAFE"
BLUE_DARK = "#1E40AF"

PURPLE = "#7C3AED"      # Quantum
PURPLE_LIGHT = "#EDE9FE"
PURPLE_DARK = "#5B21B6"

GREEN = "#059669"       # Output/Results
GREEN_LIGHT = "#D1FAE5"
GREEN_DARK = "#047857"

ORANGE = "#EA580C"      # Pre-filter
ORANGE_LIGHT = "#FED7AA"
ORANGE_DARK = "#C2410C"

# Neutral colors
GRAY_900 = "#111827"    # Primary text
GRAY_700 = "#374151"    # Secondary text
GRAY_500 = "#6B7280"    # Tertiary text
GRAY_400 = "#9CA3AF"    # Borders
GRAY_200 = "#E5E7EB"    # Light borders
GRAY_100 = "#F3F4F6"    # Background fills
GRAY_50 = "#F9FAFB"
GRAY_300 = "#D1D5DB"

# Accent for pre/post
PRE_GREEN = "#16A34A"
POST_BROWN = "#B45309"

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_image(folder, idx=0, size=(180, 135)):
    """Load and crop an image from the dataset."""
    folder_path = os.path.join(ICML_ROOT, folder)
    if not os.path.exists(folder_path):
        return np.ones((*size[::-1], 3), dtype=np.uint8) * 200
    
    files = sorted([f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not files or idx >= len(files):
        return np.ones((*size[::-1], 3), dtype=np.uint8) * 200
    
    img_path = os.path.join(folder_path, files[idx])
    img = cv2.imread(img_path)
    if img is None:
        return np.ones((*size[::-1], 3), dtype=np.uint8) * 200
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Center crop to aspect ratio
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


def rounded_box(ax, x, y, w, h, fc=GRAY_100, ec=GRAY_400, lw=1.0, 
                radius=0.02, alpha=1.0, zorder=2):
    """Draw a rounded rectangle."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        alpha=alpha, zorder=zorder, transform=ax.transData
    )
    ax.add_patch(box)
    return box


def text(ax, x, y, s, color=GRAY_900, fontsize=9, fontweight="normal",
         ha="center", va="center", zorder=5, style="normal", family="sans-serif"):
    """Add text to axes."""
    return ax.text(x, y, s, ha=ha, va=va, color=color, fontsize=fontsize,
                   fontweight=fontweight, fontstyle=style, zorder=zorder,
                   fontfamily=family, multialignment="center")


def arrow(ax, x1, y1, x2, y2, color=GRAY_500, lw=1.5, style="-|>", 
          mutation_scale=12, connectionstyle="arc3,rad=0"):
    """Draw an arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               mutation_scale=mutation_scale,
                               connectionstyle=connectionstyle),
                zorder=6)


def section_header(ax, x, y, label, color=GRAY_700, fontsize=10):
    """Add a section header."""
    text(ax, x, y, label, color=color, fontsize=fontsize, fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def build_architecture():
    """Build the complete architecture diagram."""
    
    # Figure setup - wide aspect ratio for pipeline
    fig = plt.figure(figsize=(18, 8.5), facecolor=BG, dpi=100)
    
    # Main axes for the diagram
    ax = fig.add_axes([0.02, 0.12, 0.96, 0.78])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7)
    ax.set_facecolor(BG)
    ax.axis("off")
    
    # ══════════════════════════════════════════════════════════════════════════
    # SECTION (a): INPUT - UAV Images
    # ══════════════════════════════════════════════════════════════════════════
    
    # Load sample images
    pre_img1 = load_image(PRE_FOLDERS[0], idx=0, size=(140, 100))
    pre_img2 = load_image(PRE_FOLDERS[0], idx=5, size=(140, 100))
    post_img1 = load_image(POST_FOLDERS[0], idx=0, size=(140, 100))
    post_img2 = load_image(POST_FOLDERS[1], idx=3, size=(140, 100))
    
    # Pre-defoliation images (top)
    ax_pre1 = fig.add_axes([0.025, 0.62, 0.072, 0.16])
    ax_pre2 = fig.add_axes([0.025, 0.44, 0.072, 0.16])
    ax_pre1.imshow(pre_img1)
    ax_pre2.imshow(pre_img2)
    
    for ax_img in [ax_pre1, ax_pre2]:
        ax_img.axis("off")
        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(PRE_GREEN)
            spine.set_linewidth(2.5)
    
    # Post-defoliation images (bottom)
    ax_post1 = fig.add_axes([0.025, 0.22, 0.072, 0.16])
    ax_post2 = fig.add_axes([0.025, 0.04, 0.072, 0.16])
    ax_post1.imshow(post_img1)
    ax_post2.imshow(post_img2)
    
    for ax_img in [ax_post1, ax_post2]:
        ax_img.axis("off")
        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(POST_BROWN)
            spine.set_linewidth(2.5)
    
    # Labels for images
    text(ax, 1.05, 5.85, "Pre-Defoliation", color=PRE_GREEN, fontsize=8, fontweight="bold")
    text(ax, 1.05, 2.35, "Post-Defoliation", color=POST_BROWN, fontsize=8, fontweight="bold")
    
    # Dataset info
    text(ax, 1.05, 1.05, "1,549 UAV images\n6 flight sessions", 
         color=GRAY_500, fontsize=7.5)
    
    # Bracket - use matplotlib directly for rotation
    ax.text(0.15, 4.0, "}", color=GRAY_400, fontsize=55, fontweight="normal", 
            rotation=180, va="center", ha="center", zorder=5)
    
    # ══════════════════════════════════════════════════════════════════════════
    # ARROW: Input → Feature Extraction
    # ══════════════════════════════════════════════════════════════════════════
    arrow(ax, 2.1, 4.0, 2.7, 4.0, color=GRAY_500)
    
    # ══════════════════════════════════════════════════════════════════════════
    # SECTION (b): FEATURE EXTRACTION
    # ══════════════════════════════════════════════════════════════════════════
    
    # Main container
    rounded_box(ax, 2.7, 0.8, 3.3, 5.8, fc=BLUE_LIGHT, ec=BLUE, lw=1.5, radius=0.08)
    section_header(ax, 4.35, 6.35, "Feature Extraction", color=BLUE)
    
    # Color Features sub-box
    rounded_box(ax, 2.85, 3.6, 3.0, 2.75, fc=WHITE, ec=BLUE, lw=1.0, radius=0.05)
    text(ax, 4.35, 6.1, "Color Indices", color=BLUE, fontsize=9, fontweight="bold")
    
    color_features = [
        ("ExG", "Excess Green Index"),
        ("σ(ExG)", "ExG Standard Deviation"),
        ("RBR", "Red-Blue Ratio"),
        ("NGRDI", "Norm. Green-Red Diff."),
        ("R, G, B", "Channel Means"),
    ]
    
    for i, (abbr, desc) in enumerate(color_features):
        yy = 5.75 - i * 0.42
        rounded_box(ax, 2.95, yy-0.15, 0.7, 0.32, fc=BLUE_LIGHT, ec=BLUE, lw=0.7, radius=0.03)
        text(ax, 3.30, yy, abbr, color=BLUE_DARK, fontsize=7, fontweight="bold")
        text(ax, 4.55, yy, desc, color=GRAY_700, fontsize=7, ha="left")
    
    # Texture Features sub-box
    rounded_box(ax, 2.85, 1.0, 3.0, 2.35, fc=WHITE, ec=BLUE, lw=1.0, radius=0.05)
    text(ax, 4.35, 3.1, "GLCM Texture", color=BLUE, fontsize=9, fontweight="bold")
    
    texture_features = [
        ("H", "Entropy"),
        ("Con", "Contrast"),
        ("Hom", "Homogeneity"),
        ("Cor", "Correlation"),
        ("E", "Energy"),
    ]
    
    for i, (abbr, desc) in enumerate(texture_features):
        yy = 2.75 - i * 0.36
        rounded_box(ax, 2.95, yy-0.13, 0.55, 0.28, fc=BLUE_LIGHT, ec=BLUE, lw=0.7, radius=0.03)
        text(ax, 3.22, yy, abbr, color=BLUE_DARK, fontsize=7, fontweight="bold")
        text(ax, 4.0, yy, desc, color=GRAY_700, fontsize=7, ha="left")
    
    # Output badge
    rounded_box(ax, 3.1, 0.88, 2.5, 0.35, fc=BLUE, ec=BLUE, lw=0, radius=0.04)
    text(ax, 4.35, 1.06, "12-dim feature vector", color=WHITE, fontsize=8, fontweight="bold")
    
    # ══════════════════════════════════════════════════════════════════════════
    # ARROW: Feature Extraction → MI Pre-Filter
    # ══════════════════════════════════════════════════════════════════════════
    arrow(ax, 6.0, 4.0, 6.6, 4.0, color=GRAY_500)
    text(ax, 6.3, 4.25, "×1,549", color=GRAY_500, fontsize=7)
    
    # ══════════════════════════════════════════════════════════════════════════
    # SECTION (c): MI PRE-FILTER
    # ══════════════════════════════════════════════════════════════════════════
    
    rounded_box(ax, 6.6, 0.8, 2.6, 5.8, fc=ORANGE_LIGHT, ec=ORANGE, lw=1.5, radius=0.08)
    section_header(ax, 7.9, 6.35, "MI Pre-Filter", color=ORANGE)
    
    # MI formula box
    rounded_box(ax, 6.75, 4.8, 2.3, 1.65, fc=WHITE, ec=ORANGE, lw=1.0, radius=0.05)
    text(ax, 7.9, 6.2, "Mutual Information", color=ORANGE, fontsize=9, fontweight="bold")
    text(ax, 7.9, 5.85, "SelectKBest (k=6)", color=GRAY_700, fontsize=8)
    
    # MI formula
    rounded_box(ax, 6.9, 5.05, 2.0, 0.45, fc=ORANGE_LIGHT, ec=ORANGE, lw=0.7, radius=0.03)
    text(ax, 7.9, 5.28, "I(X;Y) = H(Y) − H(Y|X)", color=GRAY_900, fontsize=8, style="italic")
    
    # Top-6 features
    rounded_box(ax, 6.75, 2.4, 2.3, 2.2, fc=WHITE, ec=ORANGE, lw=1.0, radius=0.05)
    text(ax, 7.9, 4.35, "Top-6 Candidates", color=ORANGE, fontsize=9, fontweight="bold")
    
    top6_features = [
        ("1.", "Std_ExG", True),
        ("2.", "Mean_ExG", False),
        ("3.", "Mean_RBR", True),
        ("4.", "Mean_NGRDI", False),
        ("5.", "Mean_B", True),
        ("6.", "Correlation", True),
    ]
    
    for i, (num, feat, selected) in enumerate(top6_features):
        yy = 4.0 - i * 0.28
        color = ORANGE if selected else GRAY_500
        weight = "bold" if selected else "normal"
        marker = "★" if selected else ""
        text(ax, 7.2, yy, num, color=GRAY_500, fontsize=7, ha="right")
        text(ax, 7.35, yy, feat, color=color, fontsize=7.5, fontweight=weight, ha="left")
        if marker:
            text(ax, 8.85, yy, marker, color=ORANGE, fontsize=7, ha="right")
    
    # Rationale box
    rounded_box(ax, 6.75, 0.95, 2.3, 1.25, fc=WHITE, ec=ORANGE, lw=1.0, radius=0.05)
    text(ax, 7.9, 1.95, "Complexity Reduction", color=ORANGE, fontsize=8, fontweight="bold")
    text(ax, 7.9, 1.55, "VQC scales O(2ⁿ)\n12 → 6 features", color=GRAY_700, fontsize=7.5)
    
    # Output badge
    rounded_box(ax, 6.9, 0.88, 2.0, 0.35, fc=ORANGE, ec=ORANGE, lw=0, radius=0.04)
    text(ax, 7.9, 1.06, "6-dim candidates", color=WHITE, fontsize=8, fontweight="bold")
    
    # ══════════════════════════════════════════════════════════════════════════
    # ARROW: MI Pre-Filter → VQC
    # ══════════════════════════════════════════════════════════════════════════
    arrow(ax, 9.2, 4.0, 9.8, 4.0, color=GRAY_500)
    
    # ══════════════════════════════════════════════════════════════════════════
    # SECTION (d): VARIATIONAL QUANTUM CLASSIFIER
    # ══════════════════════════════════════════════════════════════════════════
    
    rounded_box(ax, 9.8, 0.8, 4.8, 5.8, fc=PURPLE_LIGHT, ec=PURPLE, lw=2.0, radius=0.08)
    section_header(ax, 12.2, 6.35, "Variational Quantum Classifier", color=PURPLE)
    
    # Subset evaluation header
    rounded_box(ax, 9.95, 5.7, 4.5, 0.55, fc=WHITE, ec=PURPLE, lw=1.0, radius=0.04)
    text(ax, 12.2, 5.98, "Wrapper Search: C(6,4) = 15 subsets", 
         color=PURPLE, fontsize=9, fontweight="bold")
    
    # Quantum circuit box
    rounded_box(ax, 9.95, 2.65, 4.5, 2.85, fc=WHITE, ec=PURPLE, lw=1.0, radius=0.05)
    text(ax, 12.2, 5.25, "4-Qubit Quantum Circuit", color=PURPLE, fontsize=9, fontweight="bold")
    
    # Draw quantum circuit
    qubit_labels = ["q₀", "q₁", "q₂", "q₃"]
    qubit_features = ["Std_ExG", "Mean_RBR", "Mean_B", "Correlation"]
    wire_y = [4.85, 4.35, 3.85, 3.35]
    
    for i, (ql, qf, wy) in enumerate(zip(qubit_labels, qubit_features, wire_y)):
        # Wire
        ax.plot([10.3, 14.2], [wy, wy], color=GRAY_300, lw=1.2, zorder=2)
        
        # Qubit label
        text(ax, 10.15, wy, ql, color=PURPLE_DARK, fontsize=7, fontweight="bold", ha="right")
        text(ax, 10.5, wy + 0.18, qf, color=GRAY_500, fontsize=6, ha="left")
        
        # Hadamard gate
        rounded_box(ax, 10.7, wy-0.15, 0.35, 0.30, fc=BLUE_LIGHT, ec=BLUE, lw=0.8, radius=0.02, zorder=4)
        text(ax, 10.875, wy, "H", color=BLUE, fontsize=7, fontweight="bold")
        
        # Rz gate (ZZ Feature Map)
        rounded_box(ax, 11.2, wy-0.15, 0.4, 0.30, fc=BLUE_LIGHT, ec=BLUE, lw=0.8, radius=0.02, zorder=4)
        text(ax, 11.4, wy, "Rz", color=BLUE, fontsize=7, fontweight="bold")
        
        # Ry gate (RealAmplitudes)
        rounded_box(ax, 12.8, wy-0.15, 0.4, 0.30, fc=PURPLE_LIGHT, ec=PURPLE, lw=0.8, radius=0.02, zorder=4)
        text(ax, 13.0, wy, "Ry", color=PURPLE, fontsize=7, fontweight="bold")
        
        # Measurement
        rounded_box(ax, 13.7, wy-0.15, 0.4, 0.30, fc=GREEN_LIGHT, ec=GREEN, lw=0.8, radius=0.02, zorder=4)
        text(ax, 13.9, wy, "M", color=GREEN, fontsize=7, fontweight="bold")
    
    # CNOT gates (entanglement)
    for i in range(3):
        cy, ty = wire_y[i], wire_y[i+1]
        ax.plot([12.1, 12.1], [ty, cy], color=BLUE, lw=1.0, zorder=3)
        ax.plot(12.1, cy, 'o', color=BLUE, markersize=5, zorder=4)
        rounded_box(ax, 11.95, ty-0.12, 0.30, 0.24, fc=BLUE_LIGHT, ec=BLUE, lw=0.8, radius=0.02, zorder=4)
        text(ax, 12.1, ty, "⊕", color=BLUE, fontsize=8, fontweight="bold")
    
    # Gate labels
    gate_labels = [
        (10.875, "Hadamard"),
        (11.4, "ZZ Map"),
        (12.1, "CNOT"),
        (13.0, "Ansatz"),
        (13.9, "Measure"),
    ]
    for gx, gl in gate_labels:
        text(ax, gx, 2.95, gl, color=GRAY_500, fontsize=6, style="italic")
    
    # Optimizer info
    rounded_box(ax, 9.95, 2.55, 4.5, 0.35, fc=PURPLE_LIGHT, ec=PURPLE, lw=0.7, radius=0.03)
    text(ax, 12.2, 2.73, "COBYLA optimizer  •  StatevectorSampler  •  40 iterations",
         color=PURPLE, fontsize=7)
    
    # Result box
    rounded_box(ax, 9.95, 1.35, 4.5, 1.0, fc=WHITE, ec=PURPLE, lw=1.0, radius=0.05)
    text(ax, 12.2, 2.1, "Selected Subset (VQC Acc: 72%)", color=PURPLE, fontsize=9, fontweight="bold")
    text(ax, 12.2, 1.7, "Std_ExG  •  Mean_RBR  •  Mean_B  •  Correlation",
         color=GRAY_900, fontsize=9, fontweight="bold")
    
    # Output badge
    rounded_box(ax, 10.2, 0.88, 4.0, 0.35, fc=PURPLE, ec=PURPLE, lw=0, radius=0.04)
    text(ax, 12.2, 1.06, "4 quantum-selected features  •  Lower redundancy",
         color=WHITE, fontsize=8, fontweight="bold")
    
    # ══════════════════════════════════════════════════════════════════════════
    # ARROW: VQC → Classical Inference
    # ══════════════════════════════════════════════════════════════════════════
    arrow(ax, 14.6, 4.0, 15.2, 4.0, color=GRAY_500)
    text(ax, 14.9, 4.25, "4-dim", color=GRAY_500, fontsize=7)
    
    # ══════════════════════════════════════════════════════════════════════════
    # SECTION (e): CLASSICAL INFERENCE
    # ══════════════════════════════════════════════════════════════════════════
    
    rounded_box(ax, 15.2, 0.8, 2.6, 5.8, fc=GREEN_LIGHT, ec=GREEN, lw=1.5, radius=0.08)
    section_header(ax, 16.5, 6.35, "Classical Inference", color=GREEN)
    
    # SVM Classifier box
    rounded_box(ax, 15.35, 4.0, 2.3, 2.4, fc=WHITE, ec=GREEN, lw=1.0, radius=0.05)
    text(ax, 16.5, 6.15, "SVM-RBF Classifier", color=GREEN, fontsize=9, fontweight="bold")
    text(ax, 16.5, 5.8, "kernel=rbf, C=10", color=GRAY_700, fontsize=7.5, style="italic")
    
    # Accuracy result
    rounded_box(ax, 15.5, 4.85, 2.0, 0.7, fc=GREEN_LIGHT, ec=GREEN, lw=0.8, radius=0.04)
    text(ax, 16.5, 5.35, "5-Fold CV Accuracy", color=GRAY_700, fontsize=7)
    text(ax, 16.5, 5.02, "100.0% ± 0.0", color=GREEN_DARK, fontsize=12, fontweight="bold")
    
    text(ax, 16.5, 4.45, "Zero annotation\nrequired", color=GRAY_500, fontsize=7.5)
    
    # Output classes
    rounded_box(ax, 15.35, 2.2, 2.3, 1.55, fc=WHITE, ec=GREEN, lw=1.0, radius=0.05)
    text(ax, 16.5, 3.5, "Output Classes", color=GREEN, fontsize=9, fontweight="bold")
    
    rounded_box(ax, 15.5, 2.9, 2.0, 0.42, fc=GREEN_LIGHT, ec=PRE_GREEN, lw=0.8, radius=0.03)
    text(ax, 16.5, 3.11, "Pre-Defoliation", color=PRE_GREEN, fontsize=8, fontweight="bold")
    
    rounded_box(ax, 15.5, 2.38, 2.0, 0.42, fc=ORANGE_LIGHT, ec=POST_BROWN, lw=0.8, radius=0.03)
    text(ax, 16.5, 2.59, "Post-Defoliation", color=POST_BROWN, fontsize=8, fontweight="bold")
    
    # Key insight box
    rounded_box(ax, 15.35, 0.95, 2.3, 1.05, fc=WHITE, ec=GREEN, lw=1.0, radius=0.05)
    text(ax, 16.5, 1.75, "Key Insight", color=GREEN, fontsize=8, fontweight="bold")
    text(ax, 16.5, 1.35, "Quantum at training\nClassical at inference", 
         color=GRAY_700, fontsize=7.5)
    
    # Output badge
    rounded_box(ax, 15.5, 0.88, 2.0, 0.35, fc=GREEN, ec=GREEN, lw=0, radius=0.04)
    text(ax, 16.5, 1.06, "Edge-deployable", color=WHITE, fontsize=8, fontweight="bold")
    
    # ══════════════════════════════════════════════════════════════════════════
    # SECTION LABELS (bottom strip)
    # ══════════════════════════════════════════════════════════════════════════
    
    ax_labels = fig.add_axes([0.02, 0.02, 0.96, 0.08])
    ax_labels.set_xlim(0, 18)
    ax_labels.set_ylim(0, 1)
    ax_labels.set_facecolor(BG)
    ax_labels.axis("off")
    
    sections = [
        (1.05, "(a) UAV Input", GRAY_700),
        (4.35, "(b) Feature\nExtraction", BLUE),
        (7.9, "(c) MI\nPre-Filter", ORANGE),
        (12.2, "(d) Quantum Feature\nSelection (VQC)", PURPLE),
        (16.5, "(e) Classical\nInference", GREEN),
    ]
    
    # Dividers
    dividers = [2.1, 6.3, 9.5, 14.9]
    for dv in dividers:
        ax_labels.axvline(dv, color=GRAY_300, lw=1.0, ls=':', ymin=0.1, ymax=0.9)
    
    for sx, slbl, sc in sections:
        ax_labels.text(sx/18, 0.5, slbl, ha="center", va="center", color=sc,
                      fontsize=9, fontweight="bold", multialignment="center",
                      fontfamily="sans-serif")
    
    # ══════════════════════════════════════════════════════════════════════════
    # TITLE (top strip)
    # ══════════════════════════════════════════════════════════════════════════
    
    ax_title = fig.add_axes([0.02, 0.91, 0.96, 0.08])
    ax_title.set_facecolor(BG)
    ax_title.axis("off")
    
    ax_title.text(0.5, 0.7,
                  "Hybrid Quantum-Classical Feature Selection for UAV-Based Cotton Defoliation Monitoring",
                  ha="center", va="center", fontsize=13, fontweight="bold",
                  color=GRAY_900, fontfamily="sans-serif")
    
    ax_title.text(0.5, 0.2,
                  "VQC identifies minimal sufficient features  •  No bounding-box annotation  •  Classical-only deployment",
                  ha="center", va="center", fontsize=9, color=GRAY_500,
                  fontfamily="sans-serif", style="italic")
    
    # ══════════════════════════════════════════════════════════════════════════
    # LEGEND
    # ══════════════════════════════════════════════════════════════════════════
    
    legend_items = [
        (BLUE, "Classical processing"),
        (ORANGE, "MI pre-filter"),
        (PURPLE, "Quantum (training)"),
        (GREEN, "Inference"),
    ]
    
    lx, ly = 16.8, 6.7
    for i, (lc, ll) in enumerate(legend_items):
        rounded_box(ax, lx, ly - i*0.28, 0.2, 0.18, fc=lc, ec=lc, lw=0, radius=0.02, zorder=8)
        text(ax, lx + 0.28, ly - i*0.28 + 0.09, ll, color=GRAY_700, fontsize=7, ha="left")
    
    # Star legend
    text(ax, lx + 0.1, ly - 4*0.28 + 0.09, "★", color=ORANGE, fontsize=8)
    text(ax, lx + 0.28, ly - 4*0.28 + 0.09, "= quantum-selected", color=GRAY_700, fontsize=7, ha="left")
    
    # ══════════════════════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════════════════════
    
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=BG, 
                pad_inches=0.1, format='png')
    print(f"✅ Saved: {OUT_PNG}")
    
    # Also save PDF for vector quality
    pdf_path = OUT_PNG.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches="tight", facecolor=BG, 
                pad_inches=0.1, format='pdf')
    print(f"✅ Saved: {pdf_path}")
    
    plt.close()


if __name__ == "__main__":
    build_architecture()
