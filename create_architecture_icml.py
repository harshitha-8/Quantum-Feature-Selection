#!/usr/bin/env python3
"""
Publication-Quality Architecture Diagram for ICML/NeurIPS/CVPR
==============================================================
Hybrid Quantum-Classical Feature Selection for Cotton Defoliation

Style: Clean academic figure with minimal colors, thin arrows,
consistent rounded rectangles, and balanced grid layout.
"""

import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.lines as mlines
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
# COLOR PALETTE - Soft academic colors (no gradients)
# ══════════════════════════════════════════════════════════════════════════════

WHITE = "#FFFFFF"
BLACK = "#000000"

# Four soft academic colors
LIGHT_GRAY = "#F5F5F5"      # Data
LIGHT_GREEN = "#E8F5E9"     # Feature extraction
LIGHT_BLUE = "#E3F2FD"      # Quantum blocks
LIGHT_ORANGE = "#FFF3E0"    # Inference

# Border colors (slightly darker versions)
BORDER_GRAY = "#BDBDBD"
BORDER_GREEN = "#81C784"
BORDER_BLUE = "#64B5F6"
BORDER_ORANGE = "#FFB74D"

# Text
TEXT_BLACK = "#212121"
TEXT_GRAY = "#616161"

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_image(folder, idx=0, size=(120, 100)):
    """Load UAV image."""
    folder_path = os.path.join(ICML_ROOT, folder)
    if not os.path.exists(folder_path):
        return np.ones((*size[::-1], 3), dtype=np.uint8) * 200
    
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
    
    # Center crop to square then resize
    m = min(h, w)
    img = img[(h-m)//2:(h-m)//2+m, (w-m)//2:(w-m)//2+m]
    img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    return img


def box(ax, x, y, w, h, fc=WHITE, ec=BORDER_GRAY, lw=1.0, radius=0.03, zorder=2):
    """Draw rounded rectangle."""
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=zorder, transform=ax.transData
    )
    ax.add_patch(b)
    return b


def arrow(ax, x1, y1, x2, y2, color=TEXT_GRAY, lw=1.0, style='-|>', mutation=10):
    """Draw thin arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               mutation_scale=mutation),
                zorder=5)


def txt(ax, x, y, s, color=TEXT_BLACK, size=9, weight='normal', ha='center', 
        va='center', style='normal'):
    """Add text."""
    return ax.text(x, y, s, color=color, fontsize=size, fontweight=weight,
                   ha=ha, va=va, fontstyle=style, zorder=10, multialignment='center')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def create_figure():
    """Create the ICML-style architecture diagram."""
    
    # Figure setup - landscape orientation
    fig = plt.figure(figsize=(14, 10), facecolor=WHITE, dpi=100)
    
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_facecolor(WHITE)
    ax.axis('off')
    
    # ══════════════════════════════════════════════════════════════════════════
    # SECTION (a) - Data Modalities (Left)
    # ══════════════════════════════════════════════════════════════════════════
    
    # Section title
    txt(ax, 1.8, 9.5, '(a) Data Modalities', color=TEXT_BLACK, size=12, weight='bold', ha='left')
    
    # Temporal UAV Database box
    box(ax, 0.3, 5.8, 3.4, 3.4, fc=LIGHT_GRAY, ec=BORDER_GRAY, lw=1.0, radius=0.05)
    txt(ax, 2.0, 8.95, 'Temporal UAV Database', color=TEXT_BLACK, size=10, weight='bold')
    
    # Load and display images
    pre_img = load_image(PRE_FOLDER, idx=5, size=(110, 95))
    post_img = load_image(POST_FOLDER, idx=10, size=(110, 95))
    
    # Pre-defoliation image
    ax_pre = fig.add_axes([0.045, 0.62, 0.085, 0.12])
    ax_pre.imshow(pre_img)
    ax_pre.axis('off')
    for spine in ax_pre.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BORDER_GRAY)
        spine.set_linewidth(1)
    
    # Post-defoliation image
    ax_post = fig.add_axes([0.145, 0.62, 0.085, 0.12])
    ax_post.imshow(post_img)
    ax_post.axis('off')
    for spine in ax_post.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BORDER_GRAY)
        spine.set_linewidth(1)
    
    # Image labels
    txt(ax, 0.95, 6.05, 'Pre-defoliation', color=TEXT_GRAY, size=7)
    txt(ax, 0.95, 5.75, 'RGB canopy', color=TEXT_GRAY, size=6)
    
    txt(ax, 2.55, 6.05, 'Post-defoliation', color=TEXT_GRAY, size=7)
    txt(ax, 2.55, 5.75, 'exposed cotton bolls', color=TEXT_GRAY, size=6)
    
    # Temporal sample label
    txt(ax, 1.75, 5.35, 'Temporal sample', color=TEXT_GRAY, size=7, style='italic')
    
    # Downstream Task box
    box(ax, 4.0, 6.8, 1.8, 1.6, fc=WHITE, ec=BORDER_GRAY, lw=1.0, radius=0.03)
    txt(ax, 4.9, 8.15, 'Downstream Task', color=TEXT_GRAY, size=7)
    txt(ax, 4.9, 7.7, 'Defoliation Score', color=TEXT_BLACK, size=8, weight='bold')
    txt(ax, 4.9, 7.2, r'$Y \in [0, 1]$', color=TEXT_BLACK, size=9)
    
    # Arrow from database to task
    arrow(ax, 3.7, 7.6, 4.0, 7.6, color=TEXT_GRAY, lw=1.0)
    
    # ══════════════════════════════════════════════════════════════════════════
    # SECTION (b) - Classical Feature Matrix (Bottom Left)
    # ══════════════════════════════════════════════════════════════════════════
    
    # Section title
    txt(ax, 1.8, 4.8, '(b) Classical Feature Matrix', color=TEXT_BLACK, size=12, weight='bold', ha='left')
    
    # Main feature extraction box
    box(ax, 0.3, 1.0, 4.5, 3.5, fc=LIGHT_GREEN, ec=BORDER_GREEN, lw=1.0, radius=0.05)
    txt(ax, 2.55, 4.2, 'High-Dimensional Spatial Feature Extraction', 
        color=TEXT_BLACK, size=9, weight='bold')
    
    # Color & Vegetation Indices sub-box
    box(ax, 0.5, 2.6, 2.0, 1.4, fc=WHITE, ec=BORDER_GREEN, lw=0.8, radius=0.03)
    txt(ax, 1.5, 3.8, 'Color & Veg.', color=TEXT_BLACK, size=8, weight='bold')
    txt(ax, 1.5, 3.55, 'Indices', color=TEXT_BLACK, size=8, weight='bold')
    
    txt(ax, 1.5, 3.15, 'ExG', color=TEXT_GRAY, size=8)
    txt(ax, 1.5, 2.9, 'NDVI', color=TEXT_GRAY, size=8)
    txt(ax, 1.5, 2.65, 'RBR', color=TEXT_GRAY, size=8)
    
    # Texture Features sub-box
    box(ax, 2.7, 2.6, 1.9, 1.4, fc=WHITE, ec=BORDER_GREEN, lw=0.8, radius=0.03)
    txt(ax, 3.65, 3.8, 'Texture', color=TEXT_BLACK, size=8, weight='bold')
    txt(ax, 3.65, 3.55, 'Features', color=TEXT_BLACK, size=8, weight='bold')
    
    txt(ax, 3.65, 3.15, 'GLCM', color=TEXT_GRAY, size=8)
    txt(ax, 3.65, 2.9, 'Contrast', color=TEXT_GRAY, size=8)
    txt(ax, 3.65, 2.65, 'Entropy', color=TEXT_GRAY, size=8)
    txt(ax, 3.65, 2.4, 'Homogeneity', color=TEXT_GRAY, size=7)
    
    # Output dimension
    txt(ax, 2.55, 1.7, r'$X \in \mathbb{R}^{700}$', color=TEXT_BLACK, size=11, weight='bold')
    
    # Arrow with "Encode" label
    arrow(ax, 4.8, 2.75, 5.8, 2.75, color=TEXT_GRAY, lw=1.0)
    txt(ax, 5.3, 3.0, 'Encode', color=TEXT_GRAY, size=8)
    
    # ══════════════════════════════════════════════════════════════════════════
    # SECTION (c) - Hybrid Quantum Architecture (Center)
    # ══════════════════════════════════════════════════════════════════════════
    
    # Section title
    txt(ax, 7.8, 9.5, '(c) Hybrid Quantum Architecture', color=TEXT_BLACK, size=12, weight='bold', ha='left')
    
    # Output at top
    txt(ax, 8.0, 9.1, 'Defoliation Score', color=TEXT_BLACK, size=9, weight='bold')
    
    # Arrow down
    arrow(ax, 8.0, 8.9, 8.0, 8.5, color=TEXT_GRAY, lw=1.0)
    
    # Downstream SVM Inference box
    box(ax, 6.0, 7.0, 4.0, 1.4, fc=LIGHT_ORANGE, ec=BORDER_ORANGE, lw=1.0, radius=0.04)
    txt(ax, 8.0, 8.15, 'Downstream SVM Inference', color=TEXT_BLACK, size=9, weight='bold')
    txt(ax, 8.0, 7.75, 'Radial Basis Function Kernel', color=TEXT_GRAY, size=8)
    txt(ax, 8.0, 7.35, r"$K(x, x') = \exp(-\gamma \|x - x'\|^2)$", color=TEXT_BLACK, size=9)
    
    # Arrow down
    arrow(ax, 8.0, 7.0, 8.0, 6.6, color=TEXT_GRAY, lw=1.0)
    
    # Variational Quantum Circuit box
    box(ax, 6.0, 3.0, 4.0, 3.5, fc=LIGHT_BLUE, ec=BORDER_BLUE, lw=1.0, radius=0.04)
    txt(ax, 8.0, 6.25, 'Variational Quantum Circuit', color=TEXT_BLACK, size=9, weight='bold')
    
    # Layer 3: Measurement
    box(ax, 6.3, 5.3, 3.4, 0.7, fc=WHITE, ec=BORDER_BLUE, lw=0.8, radius=0.02)
    txt(ax, 8.0, 5.65, r'3. Measurement $Z_i$', color=TEXT_BLACK, size=8)
    
    # Draw measurement symbols
    for i, mx in enumerate([6.8, 7.5, 8.2, 8.9]):
        # Measurement box symbol
        box(ax, mx-0.15, 5.35, 0.3, 0.25, fc=WHITE, ec=BORDER_BLUE, lw=0.6, radius=0.01)
        # Arc inside
        ax.plot([mx-0.08, mx, mx+0.08], [5.42, 5.52, 5.42], color=BORDER_BLUE, lw=0.6)
        # Arrow up from measurement
        ax.annotate('', xy=(mx, 5.75), xytext=(mx, 5.6),
                   arrowprops=dict(arrowstyle='->', color=TEXT_GRAY, lw=0.6, mutation_scale=6))
    
    # Layer 2: Entangling CNOT
    box(ax, 6.3, 4.4, 3.4, 0.7, fc=WHITE, ec=BORDER_BLUE, lw=0.8, radius=0.02)
    txt(ax, 8.0, 4.75, '2. Entangling CNOT Cascade', color=TEXT_BLACK, size=8)
    
    # Draw CNOT symbols
    for i, cx in enumerate([6.8, 7.5, 8.2]):
        # Control dot
        ax.plot(cx, 4.55, 'o', color=TEXT_BLACK, markersize=3)
        # Vertical line
        ax.plot([cx, cx], [4.45, 4.55], color=TEXT_BLACK, lw=0.6)
        # Target circle with plus
        circle = plt.Circle((cx + 0.35, 4.5), 0.08, fill=False, ec=TEXT_BLACK, lw=0.6)
        ax.add_patch(circle)
        ax.plot([cx+0.27, cx+0.43], [4.5, 4.5], color=TEXT_BLACK, lw=0.6)
        ax.plot([cx+0.35, cx+0.35], [4.42, 4.58], color=TEXT_BLACK, lw=0.6)
    
    # Arrows between CNOT and measurement
    for mx in [6.8, 7.5, 8.2, 8.9]:
        ax.annotate('', xy=(mx, 5.3), xytext=(mx, 5.1),
                   arrowprops=dict(arrowstyle='->', color=TEXT_GRAY, lw=0.5, mutation_scale=5))
    
    # Layer 1: Ry Rotations
    box(ax, 6.3, 3.5, 3.4, 0.7, fc=WHITE, ec=BORDER_BLUE, lw=0.8, radius=0.02)
    txt(ax, 8.0, 3.85, r'1. $R_y$ Parameterized Rotations', color=TEXT_BLACK, size=8)
    
    # Draw Ry gate symbols
    for rx in [6.8, 7.5, 8.2, 8.9]:
        box(ax, rx-0.12, 3.55, 0.24, 0.2, fc=WHITE, ec=BORDER_BLUE, lw=0.6, radius=0.01)
        txt(ax, rx, 3.65, r'$R_y$', color=TEXT_BLACK, size=6)
        # Arrow up
        ax.annotate('', xy=(rx, 4.4), xytext=(rx, 3.75),
                   arrowprops=dict(arrowstyle='->', color=TEXT_GRAY, lw=0.5, mutation_scale=5))
    
    # Arrows from Ry to encoding
    for rx in [6.8, 7.5, 8.2, 8.9]:
        ax.annotate('', xy=(rx, 3.5), xytext=(rx, 3.0),
                   arrowprops=dict(arrowstyle='->', color=TEXT_GRAY, lw=0.5, mutation_scale=5))
    
    # Quantum Data Encoding box
    box(ax, 6.0, 1.5, 4.0, 1.3, fc=LIGHT_BLUE, ec=BORDER_BLUE, lw=1.0, radius=0.04)
    txt(ax, 8.0, 2.5, 'Quantum Data Encoding', color=TEXT_BLACK, size=9, weight='bold')
    txt(ax, 8.0, 2.0, r'ZZFeatureMap $U_\phi(x)$', color=TEXT_BLACK, size=9)
    
    # Arrows up from encoding
    for rx in [6.8, 7.5, 8.2, 8.9]:
        ax.annotate('', xy=(rx, 3.0), xytext=(rx, 2.8),
                   arrowprops=dict(arrowstyle='->', color=TEXT_GRAY, lw=0.5, mutation_scale=5))
    
    # ══════════════════════════════════════════════════════════════════════════
    # SECTION - Optimization Search Space (Right)
    # ══════════════════════════════════════════════════════════════════════════
    
    # Vertical label on right side
    ax.text(12.8, 5.5, 'Optimization Search Space', color=TEXT_BLACK, fontsize=10, 
            fontweight='bold', rotation=90, va='center', ha='center')
    
    # Three subset boxes with mini plots
    subset_configs = [
        (7.8, 'Subset size $k = 2$'),
        (5.5, 'Subset size $k = 4$'),
        (3.2, 'Subset size $k = 6$'),
    ]
    
    for ypos, label in subset_configs:
        # Box
        box(ax, 10.5, ypos, 2.0, 1.8, fc=LIGHT_ORANGE, ec=BORDER_ORANGE, lw=0.8, radius=0.03)
        txt(ax, 11.5, ypos + 1.55, label, color=TEXT_BLACK, size=8)
        
        # Mini optimization curve (simple line plot)
        # Generate a simple convergence-like curve
        np.random.seed(int(ypos * 10))
        x_pts = np.linspace(0, 1, 8)
        y_pts = 1 - 0.7 * np.exp(-3 * x_pts) + 0.05 * np.random.randn(8)
        y_pts = np.clip(y_pts, 0.3, 1.0)
        
        # Scale to box coordinates
        x_scaled = 10.7 + x_pts * 1.6
        y_scaled = ypos + 0.2 + y_pts * 1.1
        
        ax.plot(x_scaled, y_scaled, color=BORDER_ORANGE, lw=1.2)
        
        # Small dots at data points
        ax.scatter(x_scaled[::2], y_scaled[::2], color=BORDER_ORANGE, s=8, zorder=6)
    
    # Arrows connecting to quantum circuit
    arrow(ax, 10.0, 7.6, 10.5, 8.7, color=TEXT_GRAY, lw=0.8)
    arrow(ax, 10.0, 5.5, 10.5, 6.4, color=TEXT_GRAY, lw=0.8)
    arrow(ax, 10.0, 3.5, 10.5, 4.1, color=TEXT_GRAY, lw=0.8)
    
    # ══════════════════════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════════════════════
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # PNG
    png_path = os.path.join(OUT_DIR, 'architecture_icml.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor=WHITE, pad_inches=0.1)
    print(f'✅ Saved: {png_path}')
    
    # PDF
    pdf_path = os.path.join(OUT_DIR, 'architecture_icml.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor=WHITE, pad_inches=0.1, format='pdf')
    print(f'✅ Saved: {pdf_path}')
    
    plt.close()


if __name__ == '__main__':
    create_figure()
