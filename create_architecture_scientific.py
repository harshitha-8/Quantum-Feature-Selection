#!/usr/bin/env python3
"""
Scientific Architecture Diagram - Publication Quality
======================================================
Hybrid Quantum-Classical Feature Selection for UAV-Based Cotton Defoliation

Design following Nature/Science/NeurIPS standards:
- Clean, minimal design with consistent visual language
- Proper visual hierarchy with clear information flow
- Real sample images with scientific annotations
- Consistent spacing and alignment throughout
- Mathematical rigor in notation
"""

import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as path_effects
import cv2

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

ICML_ROOT = "/Volumes/T9/ICML"
PRE_FOLDER = "Part_one_pre_def_rgb"
POST_FOLDER = "Post_def_rgb_part1"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "paper_figures")

# ══════════════════════════════════════════════════════════════════════════════
# COLOR SCHEME - Consistent, professional palette
# ══════════════════════════════════════════════════════════════════════════════

# Background
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F8F9FA"

# Primary colors for pipeline stages
C_INPUT = "#2D5A27"       # Dark green for input/agriculture
C_INPUT_L = "#E8F5E9"
C_EXTRACT = "#1565C0"     # Blue for classical processing
C_EXTRACT_L = "#E3F2FD"
C_FILTER = "#E65100"      # Orange for filtering
C_FILTER_L = "#FFF3E0"
C_QUANTUM = "#6A1B9A"     # Purple for quantum
C_QUANTUM_L = "#F3E5F5"
C_OUTPUT = "#00695C"      # Teal for output
C_OUTPUT_L = "#E0F2F1"

# Text colors
TEXT_PRIMARY = "#212121"
TEXT_SECONDARY = "#616161"
TEXT_LIGHT = "#9E9E9E"

# Accent colors
ACCENT_PRE = "#4CAF50"    # Green for pre-defoliation
ACCENT_POST = "#D84315"   # Red-orange for post-defoliation

# Border/line colors
BORDER = "#BDBDBD"
BORDER_LIGHT = "#E0E0E0"

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_image(folder, idx=0, size=(200, 150)):
    """Load and process UAV image."""
    folder_path = os.path.join(ICML_ROOT, folder)
    if not os.path.exists(folder_path):
        img = np.ones((*size[::-1], 3), dtype=np.uint8) * 200
        return img
    
    files = sorted([f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                   and not f.startswith('._')])
    
    if not files:
        return np.ones((*size[::-1], 3), dtype=np.uint8) * 200
    
    img_path = os.path.join(folder_path, files[min(idx, len(files)-1)])
    img = cv2.imread(img_path)
    
    if img is None:
        return np.ones((*size[::-1], 3), dtype=np.uint8) * 200
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Center crop
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
    
    img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    return img


def draw_box(ax, x, y, w, h, fc='white', ec=BORDER, lw=1, radius=0.02, zorder=2):
    """Draw rounded rectangle."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=zorder, transform=ax.transData
    )
    ax.add_patch(box)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color=TEXT_SECONDARY, lw=1.5, style='-|>', 
               mutation=12, zorder=5):
    """Draw arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               mutation_scale=mutation),
                zorder=zorder)


def add_text(ax, x, y, text, color=TEXT_PRIMARY, size=9, weight='normal',
             ha='center', va='center', style='normal', zorder=10):
    """Add text with consistent styling."""
    return ax.text(x, y, text, color=color, fontsize=size, fontweight=weight,
                   ha=ha, va=va, fontstyle=style, zorder=zorder,
                   multialignment='center')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def create_architecture():
    """Create the scientific architecture diagram."""
    
    # Figure setup
    fig = plt.figure(figsize=(16, 9), facecolor=WHITE, dpi=100)
    
    # Main canvas
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.84])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.set_facecolor(WHITE)
    ax.axis('off')
    
    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1: INPUT DATA
    # ══════════════════════════════════════════════════════════════════════════
    
    # Section background
    draw_box(ax, 0.15, 0.6, 2.4, 6.8, fc=C_INPUT_L, ec=C_INPUT, lw=1.5, radius=0.03)
    
    # Section title
    add_text(ax, 1.35, 7.15, 'Input Data', color=C_INPUT, size=11, weight='bold')
    
    # Load real images
    pre_img = load_image(PRE_FOLDER, idx=10, size=(180, 130))
    post_img = load_image(POST_FOLDER, idx=15, size=(180, 130))
    
    # Pre-defoliation image
    ax_pre = fig.add_axes([0.025, 0.52, 0.11, 0.22])
    ax_pre.imshow(pre_img)
    ax_pre.axis('off')
    for spine in ax_pre.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(ACCENT_PRE)
        spine.set_linewidth(3)
    
    # Post-defoliation image
    ax_post = fig.add_axes([0.025, 0.22, 0.11, 0.22])
    ax_post.imshow(post_img)
    ax_post.axis('off')
    for spine in ax_post.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(ACCENT_POST)
        spine.set_linewidth(3)
    
    # Image labels
    add_text(ax, 1.35, 5.95, 'Pre-Defoliation', color=ACCENT_PRE, size=9, weight='bold')
    add_text(ax, 1.35, 5.65, 'y = 0', color=TEXT_SECONDARY, size=8, style='italic')
    
    add_text(ax, 1.35, 2.85, 'Post-Defoliation', color=ACCENT_POST, size=9, weight='bold')
    add_text(ax, 1.35, 2.55, 'y = 1', color=TEXT_SECONDARY, size=8, style='italic')
    
    # Dataset info box
    draw_box(ax, 0.35, 0.85, 2.0, 1.1, fc=WHITE, ec=C_INPUT, lw=1, radius=0.02)
    add_text(ax, 1.35, 1.65, 'Dataset', color=C_INPUT, size=9, weight='bold')
    add_text(ax, 1.35, 1.30, 'N = 1,549 images\n6 UAV flights\nRGB (4000×3000)', 
             color=TEXT_SECONDARY, size=8)
    
    # ══════════════════════════════════════════════════════════════════════════
    # ARROW 1→2
    # ══════════════════════════════════════════════════════════════════════════
    draw_arrow(ax, 2.55, 4.0, 3.15, 4.0, color=TEXT_SECONDARY)
    
    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: FEATURE EXTRACTION
    # ══════════════════════════════════════════════════════════════════════════
    
    draw_box(ax, 3.15, 0.6, 3.0, 6.8, fc=C_EXTRACT_L, ec=C_EXTRACT, lw=1.5, radius=0.03)
    add_text(ax, 4.65, 7.15, 'Feature Extraction', color=C_EXTRACT, size=11, weight='bold')
    
    # Color features section
    draw_box(ax, 3.35, 3.85, 2.6, 3.35, fc=WHITE, ec=C_EXTRACT, lw=1, radius=0.02)
    add_text(ax, 4.65, 6.95, 'Color Indices', color=C_EXTRACT, size=10, weight='bold')
    
    color_features = [
        ('ExG', '2G - R - B', False),
        ('σ(ExG)', 'Std. deviation', True),
        ('RBR', 'R / B ratio', True),
        ('NGRDI', '(G-R)/(G+R)', False),
        ('μ(R)', 'Mean red', False),
        ('μ(G)', 'Mean green', False),
        ('μ(B)', 'Mean blue', True),
    ]
    
    for i, (name, formula, selected) in enumerate(color_features):
        yy = 6.55 - i * 0.40
        # Feature box
        fc = C_EXTRACT_L if selected else WHITE
        ec = C_EXTRACT if selected else BORDER_LIGHT
        lw = 1.2 if selected else 0.8
        draw_box(ax, 3.45, yy-0.15, 2.4, 0.35, fc=fc, ec=ec, lw=lw, radius=0.015)
        
        # Feature name
        weight = 'bold' if selected else 'normal'
        color = C_EXTRACT if selected else TEXT_SECONDARY
        add_text(ax, 3.95, yy, name, color=color, size=8, weight=weight)
        
        # Formula
        add_text(ax, 5.15, yy, formula, color=TEXT_LIGHT, size=7, ha='left', style='italic')
        
        # Selection marker
        if selected:
            add_text(ax, 5.7, yy, '★', color=C_QUANTUM, size=8)
    
    # Texture features section
    draw_box(ax, 3.35, 0.85, 2.6, 2.75, fc=WHITE, ec=C_EXTRACT, lw=1, radius=0.02)
    add_text(ax, 4.65, 3.35, 'GLCM Texture', color=C_EXTRACT, size=10, weight='bold')
    
    texture_features = [
        ('Entropy', 'H = -Σp·log(p)', False),
        ('Contrast', 'Σ|i-j|²·p(i,j)', False),
        ('Homogeneity', 'Σp/(1+|i-j|)', False),
        ('Correlation', 'Σ(i-μ)(j-μ)·p/σ²', True),
        ('Energy', 'Σp(i,j)²', False),
    ]
    
    for i, (name, formula, selected) in enumerate(texture_features):
        yy = 2.95 - i * 0.40
        fc = C_EXTRACT_L if selected else WHITE
        ec = C_EXTRACT if selected else BORDER_LIGHT
        lw = 1.2 if selected else 0.8
        draw_box(ax, 3.45, yy-0.15, 2.4, 0.35, fc=fc, ec=ec, lw=lw, radius=0.015)
        
        weight = 'bold' if selected else 'normal'
        color = C_EXTRACT if selected else TEXT_SECONDARY
        add_text(ax, 3.95, yy, name, color=color, size=8, weight=weight)
        add_text(ax, 5.0, yy, formula, color=TEXT_LIGHT, size=6.5, ha='left', style='italic')
        
        if selected:
            add_text(ax, 5.7, yy, '★', color=C_QUANTUM, size=8)
    
    # Output dimension
    draw_box(ax, 3.65, 0.68, 2.0, 0.35, fc=C_EXTRACT, ec=C_EXTRACT, lw=0, radius=0.015)
    add_text(ax, 4.65, 0.86, 'X ∈ ℝ¹²', color=WHITE, size=9, weight='bold')
    
    # ══════════════════════════════════════════════════════════════════════════
    # ARROW 2→3
    # ══════════════════════════════════════════════════════════════════════════
    draw_arrow(ax, 6.15, 4.0, 6.75, 4.0, color=TEXT_SECONDARY)
    add_text(ax, 6.45, 4.25, '×N', color=TEXT_LIGHT, size=7)
    
    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3: MI PRE-FILTER
    # ══════════════════════════════════════════════════════════════════════════
    
    draw_box(ax, 6.75, 0.6, 2.4, 6.8, fc=C_FILTER_L, ec=C_FILTER, lw=1.5, radius=0.03)
    add_text(ax, 7.95, 7.15, 'MI Pre-Filter', color=C_FILTER, size=11, weight='bold')
    
    # MI formula box
    draw_box(ax, 6.95, 5.2, 2.0, 1.9, fc=WHITE, ec=C_FILTER, lw=1, radius=0.02)
    add_text(ax, 7.95, 6.85, 'Mutual Information', color=C_FILTER, size=9, weight='bold')
    
    # MI formula
    add_text(ax, 7.95, 6.40, r'$I(X_i; Y) = H(Y) - H(Y|X_i)$', 
             color=TEXT_PRIMARY, size=9, style='italic')
    
    add_text(ax, 7.95, 5.95, 'SelectKBest', color=TEXT_SECONDARY, size=8)
    add_text(ax, 7.95, 5.60, 'k = 6', color=C_FILTER, size=10, weight='bold')
    
    # Ranked features
    draw_box(ax, 6.95, 2.35, 2.0, 2.6, fc=WHITE, ec=C_FILTER, lw=1, radius=0.02)
    add_text(ax, 7.95, 4.7, 'Ranked Features', color=C_FILTER, size=9, weight='bold')
    
    ranked = [
        ('1', 'Std_ExG', True),
        ('2', 'Mean_ExG', False),
        ('3', 'Mean_RBR', True),
        ('4', 'Mean_NGRDI', False),
        ('5', 'Mean_B', True),
        ('6', 'Correlation', True),
    ]
    
    for i, (rank, name, selected) in enumerate(ranked):
        yy = 4.35 - i * 0.33
        color = C_FILTER if selected else TEXT_LIGHT
        weight = 'bold' if selected else 'normal'
        add_text(ax, 7.25, yy, rank + '.', color=TEXT_LIGHT, size=8, ha='right')
        add_text(ax, 7.35, yy, name, color=color, size=8, weight=weight, ha='left')
        if selected:
            add_text(ax, 8.75, yy, '★', color=C_QUANTUM, size=8)
    
    # Rationale box
    draw_box(ax, 6.95, 0.85, 2.0, 1.25, fc=WHITE, ec=C_FILTER, lw=1, radius=0.02)
    add_text(ax, 7.95, 1.85, 'Complexity', color=C_FILTER, size=8, weight='bold')
    add_text(ax, 7.95, 1.45, r'VQC: $O(2^n)$ qubits', color=TEXT_SECONDARY, size=8)
    add_text(ax, 7.95, 1.10, '12 → 6 features', color=TEXT_SECONDARY, size=8)
    
    # Output
    draw_box(ax, 7.2, 0.68, 1.5, 0.35, fc=C_FILTER, ec=C_FILTER, lw=0, radius=0.015)
    add_text(ax, 7.95, 0.86, 'X ∈ ℝ⁶', color=WHITE, size=9, weight='bold')
    
    # ══════════════════════════════════════════════════════════════════════════
    # ARROW 3→4
    # ══════════════════════════════════════════════════════════════════════════
    draw_arrow(ax, 9.15, 4.0, 9.75, 4.0, color=TEXT_SECONDARY)
    
    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4: QUANTUM FEATURE SELECTION (VQC)
    # ══════════════════════════════════════════════════════════════════════════
    
    draw_box(ax, 9.75, 0.6, 3.6, 6.8, fc=C_QUANTUM_L, ec=C_QUANTUM, lw=2.0, radius=0.03)
    add_text(ax, 11.55, 7.15, 'Quantum Feature Selection', color=C_QUANTUM, size=11, weight='bold')
    
    # Wrapper search header
    draw_box(ax, 9.95, 6.35, 3.2, 0.65, fc=WHITE, ec=C_QUANTUM, lw=1, radius=0.02)
    add_text(ax, 11.55, 6.75, 'Wrapper Search', color=C_QUANTUM, size=9, weight='bold')
    add_text(ax, 11.55, 6.48, r'$\binom{6}{4} = 15$ subsets evaluated', 
             color=TEXT_SECONDARY, size=8)
    
    # Quantum circuit
    draw_box(ax, 9.95, 2.85, 3.2, 3.25, fc=WHITE, ec=C_QUANTUM, lw=1, radius=0.02)
    add_text(ax, 11.55, 5.85, '4-Qubit VQC', color=C_QUANTUM, size=10, weight='bold')
    
    # Circuit diagram
    qubit_y = [5.35, 4.85, 4.35, 3.85]
    qubit_labels = [r'$|0\rangle$', r'$|0\rangle$', r'$|0\rangle$', r'$|0\rangle$']
    feature_labels = ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Corr.']
    
    for i, (qy, ql, fl) in enumerate(zip(qubit_y, qubit_labels, feature_labels)):
        # Qubit wire
        ax.plot([10.2, 12.9], [qy, qy], color=BORDER, lw=1.0, zorder=2)
        
        # Initial state
        add_text(ax, 10.1, qy, ql, color=C_QUANTUM, size=7, ha='right')
        
        # Feature label (small, above wire)
        add_text(ax, 10.45, qy + 0.18, fl, color=TEXT_LIGHT, size=6)
        
        # Hadamard gate
        draw_box(ax, 10.35, qy-0.12, 0.28, 0.24, fc=C_EXTRACT_L, ec=C_EXTRACT, 
                lw=0.8, radius=0.008, zorder=4)
        add_text(ax, 10.49, qy, 'H', color=C_EXTRACT, size=7, weight='bold')
        
        # ZZ Feature Map (Rz)
        draw_box(ax, 10.75, qy-0.12, 0.32, 0.24, fc=C_EXTRACT_L, ec=C_EXTRACT,
                lw=0.8, radius=0.008, zorder=4)
        add_text(ax, 10.91, qy, r'$R_z$', color=C_EXTRACT, size=7, weight='bold')
        
        # RealAmplitudes (Ry)
        draw_box(ax, 11.85, qy-0.12, 0.32, 0.24, fc=C_QUANTUM_L, ec=C_QUANTUM,
                lw=0.8, radius=0.008, zorder=4)
        add_text(ax, 12.01, qy, r'$R_y$', color=C_QUANTUM, size=7, weight='bold')
        
        # Measurement
        draw_box(ax, 12.55, qy-0.12, 0.28, 0.24, fc=C_OUTPUT_L, ec=C_OUTPUT,
                lw=0.8, radius=0.008, zorder=4)
        add_text(ax, 12.69, qy, 'M', color=C_OUTPUT, size=7, weight='bold')
    
    # CNOT gates (entanglement)
    for i in range(3):
        cy, ty = qubit_y[i], qubit_y[i+1]
        # Vertical line
        ax.plot([11.35, 11.35], [ty + 0.12, cy - 0.12], color=C_EXTRACT, lw=0.8, zorder=3)
        # Control dot
        ax.plot(11.35, cy, 'o', color=C_EXTRACT, markersize=4, zorder=4)
        # Target (XOR)
        draw_box(ax, 11.22, ty-0.10, 0.26, 0.20, fc=C_EXTRACT_L, ec=C_EXTRACT,
                lw=0.8, radius=0.006, zorder=4)
        add_text(ax, 11.35, ty, '⊕', color=C_EXTRACT, size=8, weight='bold')
    
    # Gate labels
    gate_info = [
        (10.49, 'Hadamard'),
        (10.91, 'ZZ Map'),
        (11.35, 'CNOT'),
        (12.01, 'Ansatz'),
        (12.69, 'Measure'),
    ]
    for gx, gl in gate_info:
        add_text(ax, gx, 3.45, gl, color=TEXT_LIGHT, size=6, style='italic')
    
    # Optimizer info
    draw_box(ax, 9.95, 2.75, 3.2, 0.32, fc=C_QUANTUM_L, ec=C_QUANTUM, lw=0.6, radius=0.01)
    add_text(ax, 11.55, 2.91, 'COBYLA · StatevectorSampler · 40 iter',
             color=C_QUANTUM, size=7)
    
    # Result box
    draw_box(ax, 9.95, 1.35, 3.2, 1.15, fc=WHITE, ec=C_QUANTUM, lw=1, radius=0.02)
    add_text(ax, 11.55, 2.25, 'Optimal Subset', color=C_QUANTUM, size=9, weight='bold')
    add_text(ax, 11.55, 1.90, 'VQC Accuracy: 72%', color=TEXT_SECONDARY, size=8)
    add_text(ax, 11.55, 1.55, '{Std_ExG, Mean_RBR, Mean_B, Corr.}',
             color=TEXT_PRIMARY, size=8, weight='bold')
    
    # Output
    draw_box(ax, 10.3, 0.68, 2.5, 0.35, fc=C_QUANTUM, ec=C_QUANTUM, lw=0, radius=0.015)
    add_text(ax, 11.55, 0.86, 'X* ∈ ℝ⁴ (quantum-selected)', color=WHITE, size=8, weight='bold')
    
    # ══════════════════════════════════════════════════════════════════════════
    # ARROW 4→5
    # ══════════════════════════════════════════════════════════════════════════
    draw_arrow(ax, 13.35, 4.0, 13.95, 4.0, color=TEXT_SECONDARY)
    add_text(ax, 13.65, 4.25, '4-dim', color=TEXT_LIGHT, size=7)
    
    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 5: CLASSICAL INFERENCE
    # ══════════════════════════════════════════════════════════════════════════
    
    draw_box(ax, 13.95, 0.6, 1.9, 6.8, fc=C_OUTPUT_L, ec=C_OUTPUT, lw=1.5, radius=0.03)
    add_text(ax, 14.9, 7.15, 'Inference', color=C_OUTPUT, size=11, weight='bold')
    
    # Classifier box
    draw_box(ax, 14.1, 4.55, 1.6, 2.4, fc=WHITE, ec=C_OUTPUT, lw=1, radius=0.02)
    add_text(ax, 14.9, 6.7, 'SVM-RBF', color=C_OUTPUT, size=10, weight='bold')
    add_text(ax, 14.9, 6.35, 'C = 10', color=TEXT_SECONDARY, size=8)
    add_text(ax, 14.9, 6.05, 'γ = scale', color=TEXT_SECONDARY, size=8)
    
    # Accuracy
    draw_box(ax, 14.2, 4.75, 1.4, 1.0, fc=C_OUTPUT_L, ec=C_OUTPUT, lw=0.8, radius=0.015)
    add_text(ax, 14.9, 5.55, '5-Fold CV', color=TEXT_SECONDARY, size=7)
    add_text(ax, 14.9, 5.15, '100.0%', color=C_OUTPUT, size=16, weight='bold')
    add_text(ax, 14.9, 4.85, '± 0.0', color=TEXT_SECONDARY, size=8)
    
    # Output classes
    draw_box(ax, 14.1, 2.4, 1.6, 1.9, fc=WHITE, ec=C_OUTPUT, lw=1, radius=0.02)
    add_text(ax, 14.9, 4.05, 'Prediction', color=C_OUTPUT, size=9, weight='bold')
    
    draw_box(ax, 14.2, 3.35, 1.4, 0.5, fc='#E8F5E9', ec=ACCENT_PRE, lw=1, radius=0.012)
    add_text(ax, 14.9, 3.6, 'ŷ = 0', color=ACCENT_PRE, size=9, weight='bold')
    add_text(ax, 14.9, 3.42, 'Pre-Defoliation', color=TEXT_SECONDARY, size=7)
    
    draw_box(ax, 14.2, 2.6, 1.4, 0.5, fc='#FFEBEE', ec=ACCENT_POST, lw=1, radius=0.012)
    add_text(ax, 14.9, 2.85, 'ŷ = 1', color=ACCENT_POST, size=9, weight='bold')
    add_text(ax, 14.9, 2.67, 'Post-Defoliation', color=TEXT_SECONDARY, size=7)
    
    # Key insight
    draw_box(ax, 14.1, 0.85, 1.6, 1.3, fc=WHITE, ec=C_OUTPUT, lw=1, radius=0.02)
    add_text(ax, 14.9, 1.95, 'Deployment', color=C_OUTPUT, size=8, weight='bold')
    add_text(ax, 14.9, 1.55, 'Quantum:\ntraining only', color=TEXT_SECONDARY, size=7)
    add_text(ax, 14.9, 1.05, 'Classical:\ninference', color=C_OUTPUT, size=7, weight='bold')
    
    # Output badge
    draw_box(ax, 14.2, 0.68, 1.4, 0.35, fc=C_OUTPUT, ec=C_OUTPUT, lw=0, radius=0.015)
    add_text(ax, 14.9, 0.86, 'Edge-ready', color=WHITE, size=8, weight='bold')
    
    # ══════════════════════════════════════════════════════════════════════════
    # BOTTOM SECTION LABELS
    # ══════════════════════════════════════════════════════════════════════════
    
    ax_bottom = fig.add_axes([0.02, 0.0, 0.96, 0.07])
    ax_bottom.set_xlim(0, 16)
    ax_bottom.set_ylim(0, 1)
    ax_bottom.set_facecolor(WHITE)
    ax_bottom.axis('off')
    
    sections = [
        (1.35, '(a)', C_INPUT),
        (4.65, '(b)', C_EXTRACT),
        (7.95, '(c)', C_FILTER),
        (11.55, '(d)', C_QUANTUM),
        (14.9, '(e)', C_OUTPUT),
    ]
    
    # Divider lines
    dividers = [2.75, 6.55, 9.55, 13.75]
    for dv in dividers:
        ax_bottom.axvline(dv, color=BORDER_LIGHT, lw=1, ls=':', ymin=0.2, ymax=0.8)
    
    for sx, label, color in sections:
        ax_bottom.text(sx/16, 0.5, label, ha='center', va='center',
                      color=color, fontsize=10, fontweight='bold')
    
    # ══════════════════════════════════════════════════════════════════════════
    # TITLE
    # ══════════════════════════════════════════════════════════════════════════
    
    ax_title = fig.add_axes([0.02, 0.92, 0.96, 0.08])
    ax_title.set_facecolor(WHITE)
    ax_title.axis('off')
    
    ax_title.text(0.5, 0.7,
                  'Hybrid Quantum-Classical Feature Selection for UAV-Based Cotton Defoliation Classification',
                  ha='center', va='center', fontsize=13, fontweight='bold', color=TEXT_PRIMARY)
    
    ax_title.text(0.5, 0.2,
                  'VQC wrapper identifies minimal sufficient feature subset  •  No manual annotation  •  Classical-only deployment',
                  ha='center', va='center', fontsize=9, color=TEXT_SECONDARY, style='italic')
    
    # ══════════════════════════════════════════════════════════════════════════
    # LEGEND
    # ══════════════════════════════════════════════════════════════════════════
    
    legend_items = [
        (C_INPUT, 'Input'),
        (C_EXTRACT, 'Classical'),
        (C_FILTER, 'Filter'),
        (C_QUANTUM, 'Quantum'),
        (C_OUTPUT, 'Output'),
    ]
    
    lx, ly = 0.3, 7.55
    for i, (color, label) in enumerate(legend_items):
        draw_box(ax, lx + i*1.1, ly, 0.15, 0.15, fc=color, ec=color, lw=0, radius=0.005, zorder=10)
        add_text(ax, lx + i*1.1 + 0.22, ly + 0.075, label, color=TEXT_SECONDARY, size=7, ha='left')
    
    # Star legend
    add_text(ax, lx + 5*1.1, ly + 0.075, '★ = quantum-selected', color=C_QUANTUM, size=7, ha='left')
    
    # ══════════════════════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════════════════════
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # PNG (high resolution)
    png_path = os.path.join(OUT_DIR, 'architecture_scientific.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor=WHITE, pad_inches=0.05)
    print(f'✅ Saved: {png_path}')
    
    # PDF (vector)
    pdf_path = os.path.join(OUT_DIR, 'architecture_scientific.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor=WHITE, pad_inches=0.05, format='pdf')
    print(f'✅ Saved: {pdf_path}')
    
    plt.close()


if __name__ == '__main__':
    create_architecture()
