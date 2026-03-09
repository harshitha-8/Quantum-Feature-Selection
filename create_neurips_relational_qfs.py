import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
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
    # Crop central region
    m = min(h, w)
    img = img[(h-m)//2:(h-m)//2+m, (w-m)//2:(w-m)//2+m]
    return cv2.resize(img, size)

def build():
    fig, ax = plt.subplots(figsize=(24, 13), facecolor='white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']

    C_BG_GRAY   = '#F4F5F7'
    C_GREEN     = '#E6F4EA' 
    C_PINK      = '#FCE8E6' 
    C_ORANGE    = '#FEF7E0' 
    C_ORANGE_BD = '#F9BC04' 
    C_BLUE      = '#E8F0FE' 
    C_PURPLE_BD = '#9333EA' 
    C_TEXT      = '#202124'
    C_GRAY_TXT  = '#5F6368'

    def draw_box(x, y, w, h, bg, outline=None, lw=2, zorder=2):
        ec = outline if outline else bg
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1", 
                           fc=bg, ec=ec, lw=lw, zorder=zorder)
        ax.add_patch(b)

    def txt(x, y, text, size=12, weight='normal', color=C_TEXT, ha='center', va='center', **kwargs):
        ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=10, **kwargs)

    def draw_arrow(x1, y1, x2, y2, color='#BDC1C6', lw=2.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.7", 
                                    color=color, lw=lw), zorder=5)

    # =========================================================================
    # (a) Data Modalities
    # =========================================================================
    txt(2, 95, "(a) Data Modalities", size=20, weight='bold', ha='left')
    
    draw_box(4, 60, 24, 30, C_BG_GRAY, C_BG_GRAY)
    txt(16, 87, "Temporal UAV Database", size=14, weight='bold')

    pre_img = load_crop(PRE_IMG)
    ax.imshow(pre_img, extent=[5, 14, 68, 83], zorder=4)
    txt(9.5, 65, "Pre-defoliation\nRGB canopy", size=12)

    post_img = load_crop(POST_IMG)
    ax.imshow(post_img, extent=[16, 25, 68, 83], zorder=4)
    txt(20.5, 65, "Post-defoliation\nExposed cotton bolls", size=12)

    draw_arrow(28, 75, 33, 75)
    
    draw_box(33, 68, 12, 14, 'white', C_PURPLE_BD, lw=2)
    txt(39, 79, "Downstream Task", size=11, color=C_GRAY_TXT)
    txt(39, 75, "Defoliation Score", size=13, weight='bold')
    txt(39, 71, r"$Y \in [0, 1]$", size=14)

    # =========================================================================
    # (b) Classical Feature Matrix
    # =========================================================================
    txt(2, 53, "(b) Classical Feature Matrix", size=20, weight='bold', ha='left')
    
    draw_box(2, 10, 43, 38, C_GREEN, '#CEEAD6', lw=3)
    txt(23.5, 44, "High-Dimensional Spatial Feature Extraction", size=16, weight='bold')

    draw_box(4, 15, 10, 25, 'white')
    txt(9, 36, "Color & Veg. Indices", size=12, weight='bold')
    txt(9, 25, "ExG\nNDVI\nRBR", size=12, va='center', ha='center', color=C_GRAY_TXT, linespacing=2.0)

    draw_box(16, 15, 10, 25, 'white')
    txt(21, 36, "Texture Features", size=12, weight='bold')
    txt(21, 25, "GLCM\nContrast\nEntropy\nHomogeneity", size=12, va='center', ha='center', color=C_GRAY_TXT, linespacing=2.0)

    txt(35, 27, r"$X \in \mathbb{R}^{700}$", size=20, weight='bold')
    
    draw_arrow(45, 29, 52, 29, color='#A8D8B9', lw=6)
    txt(48.5, 31, "Encode", size=12, weight='bold', color=C_GRAY_TXT)

    # =========================================================================
    # (c) Hybrid Quantum Architecture
    # =========================================================================
    txt(52, 95, "(c) Hybrid Quantum Architecture", size=20, weight='bold', ha='left')

    draw_box(55, 10, 24, 12, C_PINK)
    txt(67, 18, "Quantum Data Encoding", size=14, weight='bold')
    txt(67, 14, r"ZZFeatureMap $U_\Phi(x)$", size=13, color=C_GRAY_TXT)

    draw_box(53, 28, 28, 38, 'white', C_ORANGE_BD, lw=4)
    txt(53, 68, "Variational Quantum Circuit", size=14, weight='bold', ha='left')

    draw_box(56, 32, 22, 6, C_ORANGE)
    txt(67, 35, r"1. $R_y$ Parameterized Rotations", size=14, weight='bold')

    draw_box(56, 42, 22, 6, C_ORANGE)
    txt(67, 45, "2. Entangling CNOT Cascade", size=14, weight='bold')

    draw_box(56, 52, 22, 6, C_BLUE)
    txt(67, 55, r"3. Measurement $Z_i$", size=14, weight='bold')

    draw_box(55, 72, 24, 13, C_PINK)
    txt(67, 81, "Downstream SVM Inference", size=14, weight='bold')
    txt(67, 77, "Radial Basis Function Kernel", size=12, color=C_GRAY_TXT)
    txt(67, 74, r"$K(x, x') = \exp(-\gamma||x - x'||^2)$", size=12, color=C_GRAY_TXT)

    for x in [58, 62.5, 67, 71.5, 76]:
        draw_arrow(x, 23, x, 27)
        draw_arrow(x, 39, x, 41)
        draw_arrow(x, 49, x, 51)
        draw_arrow(x, 59, x, 71)

    draw_arrow(67, 86, 67, 92)
    txt(67, 96, "Defoliation Score\n" + r"$Y \in [0,1]$", size=12, weight='bold')

    # =========================================================================
    # Optimization Search Space
    # =========================================================================
    txt(90, 95, "Optimization Search Space", size=16, weight='bold', ha='center')
    
    colors = ['#FCE8E6', '#E8F0FE', '#FEF7E0']
    for i, y_base in enumerate([70, 45, 20]):
        draw_box(84, y_base, 12, 18, 'white', C_PURPLE_BD, lw=2)
        txt(90, y_base+15, f"Subset size $k = {2+i*2}$", size=12, weight='bold')
        
        ax.plot([86, 88, 90], [y_base+6, y_base+10, y_base+4], color=colors[i], lw=4)
        ax.plot([88, 92, 94], [y_base+10, y_base+14, y_base+5], color=colors[i], lw=4)
        
        draw_arrow(81.5, y_base+9, 83.5, y_base+9)
        
    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Graph generated at: {OUT_PNG}")

if __name__ == "__main__":
    build()
