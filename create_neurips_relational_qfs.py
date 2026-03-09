import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path
import cv2
import numpy as np
import os

PRE_IMG  = "/Volumes/T9/ICML/Part_one_pre_def_rgb/DJI_20250929095743_0311_D.JPG"
POST_IMG = "/Volumes/T9/ICML/Post_def_rgb_part1/DJI_20250929124149_0029_D.JPG"
OUT_PNG  = "/Volumes/T9/CottonDefoliationApp/neurips_qfs_relational_architecture.png"

def load_crop(path, size=(300, 200)):
    if not os.path.exists(path):
        return np.ones((*size[::-1], 3), dtype=np.uint8) * 230
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    m = min(h, w)
    img = img[(h-m)//2:(h-m)//2+m, (w-m)//2:(w-m)//2+m]
    return cv2.resize(img, size)

def build():
    # Setup ICML/NeurIPS aesthetic light theme (inspired by Relational Attention)
    fig, ax = plt.subplots(figsize=(24, 13), facecolor='white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']

    # Pastel Palette matching the inspiration image
    C_BG_GRAY   = '#F0F2F5'
    C_GREEN     = '#DDF0D9' # b) Context Window
    C_PINK      = '#F5DCDD' # Per-datatype encoder/decoder
    C_ORANGE    = '#F8E8C0' # Attention layers
    C_ORANGE_BD = '#E89D25' # Transformer Block Border
    C_BLUE      = '#D1E1EF' # MLP
    C_PURPLE_BD = '#BFA3C8' # Task Schema
    C_TEXT      = '#222222'
    C_GRAY_TXT  = '#555555'

    def draw_box(x, y, w, h, bg, outline=None, lw=2, zorder=2):
        ec = outline if outline else bg
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1", 
                           fc=bg, ec=ec, lw=lw, zorder=zorder)
        ax.add_patch(b)

    def txt(x, y, text, size=12, weight='normal', color=C_TEXT, ha='center', va='center'):
        ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va=va)

    def draw_arrow(x1, y1, x2, y2, color='#A0A0A0', lw=2.5, rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.6", 
                                    color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))

    # =========================================================================
    # (a) Input Modality (Schema equivalent)
    # =========================================================================
    txt(2, 95, "(a) Data Modality", size=20, weight='bold', ha='left')
    
    draw_box(4, 60, 22, 30, C_BG_GRAY, C_BG_GRAY)
    txt(15, 88, "UAV Spectral Database", size=14, weight='bold')

    # Images
    pre_img = load_crop(PRE_IMG)
    ax_pre = fig.add_axes([0.06, 0.70, 0.08, 0.15])
    ax_pre.imshow(pre_img); ax_pre.axis('off')
    txt(10, 68, "Pre-Defoliation (Baseline)\nRGB Canopy", size=12)

    post_img = load_crop(POST_IMG)
    ax_post = fig.add_axes([0.16, 0.70, 0.08, 0.15])
    ax_post.imshow(post_img); ax_post.axis('off')
    txt(20, 68, "Post-Defoliation (Target)\nExposed Cotton Bolls", size=12)

    draw_box(29, 68, 10, 15, 'white', C_PURPLE_BD, lw=3)
    txt(34, 80, "Downstream Task", size=11, color=C_GRAY_TXT)
    txt(34, 76, "Defoliation Score", size=13, weight='bold')
    txt(34, 72, "$Y \in [0, 1]$", size=14)
    draw_arrow(26, 75.5, 29, 75.5)

    # =========================================================================
    # (b) Context Window (Classical Extraction equivalent)
    # =========================================================================
    txt(2, 53, "(b) Classical Feature Matrix", size=20, weight='bold', ha='left')
    
    # Large green box
    draw_box(2, 10, 38, 38, C_GREEN, '#B3D6B1', lw=3)
    
    txt(21, 44, "High-Dimensional Spatial Extraction", size=16, weight='bold')

    # Draw a table/matrix representation
    draw_box(4, 15, 8, 25, 'white')
    txt(8, 37, "Color Texture", size=12)
    txt(8, 32, "ExG ($x_1$)\nNDVI ($x_2$)\nRBR ($x_3$)\n...\nGLCM ($x_{100}$)", size=12, va='top', ha='center', color=C_GRAY_TXT)

    draw_box(14, 15, 8, 25, 'white')
    txt(18, 37, "Spatial Corr.", size=12)
    txt(18, 32, "Contrast\nEnergy\nEntropy\n...\nHomog.", size=12, va='top', ha='center', color=C_GRAY_TXT)

    txt(28, 27, "$X \in \mathbb{R}^{700}$", size=20, weight='bold')
    
    # Filter step transitioning out of the matrix
    draw_arrow(40, 29, 48, 29, color='#8DB88D', lw=15)
    txt(44, 29, "Mutual Information Filter", size=12, weight='bold', color='white')

    # =========================================================================
    # (c) Quantum Architecture
    # =========================================================================
    txt(50, 95, "(c) Hybrid Quantum Architecture", size=20, weight='bold', ha='left')

    # Bottom Encoder (Classical to Quantum wrapper)
    draw_box(53, 10, 24, 12, C_PINK)
    txt(65, 18, "Quantum Data Encoding", size=14, weight='bold')
    txt(65, 14, "ZZFeatureMap $U_{\Phi(\mathbf{x})}$\nMaps Classical $X$ to Unitary Hilbert Space", size=12, color=C_GRAY_TXT)

    # Big Orange Variational Block
    draw_box(51, 28, 28, 38, 'white', C_ORANGE_BD, lw=5)
    txt(46, 47, "Variational\nQuantum\nBlock", size=14, weight='bold', ha='right')
    txt(46, 42, "(Parametrized\nOptimized via\nCOBYLA)", size=11, color=C_GRAY_TXT, ha='right')

    # Internal layers (pastel orange and blue)
    draw_box(54, 32, 22, 6, C_ORANGE)
    txt(65, 35, "$R_y$ Parameterized Rotations", size=14, weight='bold')

    draw_box(54, 42, 22, 6, C_ORANGE)
    txt(65, 45, "Entangling CNOT Cascade", size=14, weight='bold')

    draw_box(54, 52, 22, 6, C_BLUE)
    txt(65, 55, "Measurement $\langle Z_i \\rangle$", size=14, weight='bold')

    # Top Decoder (Classical SV-RBF Output)
    draw_box(53, 72, 24, 12, C_PINK)
    txt(65, 80, "Downstream SVM Inference", size=14, weight='bold')
    txt(65, 76, "Radial Basis Function (RBF) Kernel\n$K(x, x') = \exp(-\gamma||x-x'||^2)$", size=12, color=C_GRAY_TXT)

    # Vertical Arrows up the stack
    for x in [56, 60, 65, 70, 74]:
        draw_arrow(x, 23, x, 27)
        draw_arrow(x, 39, x, 41)
        draw_arrow(x, 49, x, 51)
        draw_arrow(x, 59, x, 65)
        draw_arrow(x, 67, x, 71)

    # Final output arrow
    draw_arrow(65, 85, 65, 92)
    txt(65, 95, "Predicted\nDefoliation\nRatio", size=12, weight='bold')

    # Visualizing Relational Attention side panel (Applying it to VQC metrics)
    txt(88, 95, "Optimization Sub-Space", size=16, weight='bold', ha='center')
    
    # Add some miniature subset diagrams on the right
    colors = ['#F5DCDD', '#D1E1EF', '#E89D25']
    for i, y_base in enumerate([70, 45, 20]):
        draw_box(82, y_base, 12, 18, 'white', C_PURPLE_BD, lw=2)
        txt(88, y_base+15, f"Subset Size $k={2+i*2}$", size=12, weight='bold')
        
        # draw dummy connections
        ax.plot([84, 86, 88], [y_base+6, y_base+8, y_base+4], color=colors[i], lw=3)
        ax.plot([86, 90, 92], [y_base+8, y_base+12, y_base+5], color=colors[i], lw=3)
        draw_arrow(79.5, y_base+9, 81.5, y_base+9, color='#E0E0E0') # From main architecture
        
    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Graph generated at: {OUT_PNG}")

if __name__ == "__main__":
    build()
