#!/usr/bin/env python3
"""
Architecture Diagram — Publication Quality (v2)
Clean white background, minimal palette, real drone images,
section labels below like top CVPR/NeurIPS/ICML papers.
"""
import os, warnings
import numpy as np
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import cv2

PRE_IMG  = "/Volumes/T9/ICML/Part_one_pre_def_rgb/DJI_20250929095743_0311_D.JPG"
PRE_IMG2 = "/Volumes/T9/ICML/Part_one_pre_def_rgb/DJI_20250929095749_0314_D.JPG"
POST_IMG = "/Volumes/T9/ICML/Post_def_rgb_part1/DJI_20250929124149_0029_D.JPG"
POST_IMG2= "/Volumes/T9/ICML/205_Post_Def_RGB/DJI_20250929124505_0127_D.JPG"

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "paper_figures", "architecture_v2.png")

# ── Design tokens (minimal palette like CVPR papers) ─────────────────────────
WHITE   = "#FFFFFF"
BG      = "#FFFFFF"          # clean white
GRAY_LT = "#F5F5F5"          # very light fill
GRAY_MD = "#EEEEEE"
GRAY_BG = "#FAFAFA"
GRAY_BRD= "#BDBDBD"          # subtle border
BLUE    = "#1565C0"           # primary accent (classical)
BLUE_LT = "#E3F2FD"
PURPLE  = "#6A1B9A"           # quantum accent
PURPLE_LT="#F3E5F5"
GREEN   = "#2E7D32"
GREEN_LT= "#E8F5E9"
ORANGE  = "#E65100"
ORANGE_LT="#FFF3E0"
TXT_DRK = "#212121"
TXT_MED = "#616161"
TXT_LT  = "#9E9E9E"
RED_LT  = "#FFEBEE"
RED     = "#C62828"

def load_crop(path, size=(256, 192)):
    img = cv2.imread(path)
    if img is None:
        return np.ones((*size[::-1], 3), dtype=np.uint8) * 200
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Crop center square then resize
    h, w = img.shape[:2]
    m = min(h, w)
    img = img[(h-m)//2:(h-m)//2+m, (w-m)//2:(w-m)//2+m]
    img = cv2.resize(img, size)
    return img

def rect(ax, x, y, w, h, fc=GRAY_LT, ec=GRAY_BRD, lw=1.0, radius=0.008, alpha=1.0, zorder=2):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad={radius}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=zorder,
                       transform=ax.transData)
    ax.add_patch(b)

def txt(ax, x, y, s, color=TXT_DRK, fs=9, fw="normal", ha="center", va="center",
        zorder=5, style="normal"):
    ax.text(x, y, s, ha=ha, va=va, color=color, fontsize=fs,
            fontweight=fw, fontstyle=style, zorder=zorder,
            fontfamily="sans-serif", multialignment="center")

def arrow_h(ax, x1, y, x2, color=TXT_MED, lw=1.8):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14),
                zorder=6)

def dashed_rect(ax, x, y, w, h, color=GRAY_BRD):
    r = patches.Rectangle((x, y), w, h, fill=False,
                           edgecolor=color, lw=1.0,
                           linestyle='--', zorder=1)
    ax.add_patch(r)

# ─────────────────────────────────────────────────────────────────────────────
def build():
    fig = plt.figure(figsize=(20, 9), facecolor=BG)

    # ── Two-row layout: main diagram (top 78%) + section labels (bottom 10%)
    # Plus a thin title strip at very top
    ax = fig.add_axes([0.0, 0.09, 1.0, 0.85])
    ax.set_xlim(0, 20); ax.set_ylim(0, 7.6)
    ax.set_facecolor(BG); ax.axis("off")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION A  — Input UAV Images  (x: 0.2 – 2.5)
    # ════════════════════════════════════════════════════════════════════════
    # Two real images stacked vertically  
    pre  = load_crop(PRE_IMG,  (210, 145))
    post = load_crop(POST_IMG, (210, 145))

    ax_pre  = fig.add_axes([0.014, 0.60, 0.093, 0.24])
    ax_post = fig.add_axes([0.014, 0.30, 0.093, 0.24])

    ax_pre.imshow(pre);  ax_pre.axis("off")
    ax_post.imshow(post); ax_post.axis("off")

    # Thin colored border via spines
    for sp in ax_pre.spines.values():
        sp.set_visible(True); sp.set_edgecolor(GREEN); sp.set_linewidth(2.2)
    for sp in ax_post.spines.values():
        sp.set_visible(True); sp.set_edgecolor(RED); sp.set_linewidth(2.2)

    # Image captions
    txt(ax, 1.18, 6.52, "Pre-Defoliation",  color=GREEN, fs=8.5, fw="bold")
    txt(ax, 1.18, 3.55, "Post-Defoliation", color=RED,   fs=8.5, fw="bold")

    # Bracket connecting the two images
    txt(ax, 0.22, 4.95, "{", color=TXT_MED, fs=42, fw="normal")

    # Label below
    txt(ax, 1.18, 2.72, "1,549 UAV images\n6 flight folders", color=TXT_MED, fs=7.8)

    # ════════════════════════════════════════════════════════════════════════
    # ARROW  A → B
    # ════════════════════════════════════════════════════════════════════════
    arrow_h(ax, 2.45, 4.95, 3.0, color=TXT_MED)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION B — Feature Extraction  (x: 3.0 – 6.4)
    # ════════════════════════════════════════════════════════════════════════
    rect(ax, 3.0, 1.8, 3.4, 5.6, fc=BLUE_LT, ec=BLUE, lw=1.4, radius=0.01)

    txt(ax, 4.7, 7.1, "Feature Extraction", color=BLUE, fs=10, fw="bold")

    # Color sub-box
    rect(ax, 3.15, 4.3, 3.1, 2.85, fc=WHITE, ec=BLUE, lw=0.9, radius=0.006)
    txt(ax, 4.7, 6.9,  "Color Indices  (7)", color=BLUE, fs=8.5, fw="bold")

    color_feats = [
        ("ExG",    "Excess Green"),
        ("σ(ExG)", "ExG Variance  ⚛"),
        ("RBR",    "Red-Blue Ratio  ⚛"),
        ("NGRDI",  "Norm. Green-Red"),
        ("R, G, B","Channel Means"),
    ]
    for i, (abbr, full) in enumerate(color_feats):
        yy = 6.55 - i * 0.50
        rect(ax, 3.22, yy-0.18, 0.82, 0.36, fc=BLUE_LT, ec=BLUE, lw=0.6, radius=0.005)
        txt(ax, 3.63, yy+0.003, abbr, color=BLUE, fs=7.5, fw="bold")
        txt(ax, 4.78, yy+0.003, full, color=TXT_MED, fs=7.0, ha="left")

    # Texture sub-box
    rect(ax, 3.15, 2.0, 3.1, 2.05, fc=WHITE, ec=BLUE, lw=0.9, radius=0.006)
    txt(ax, 4.7, 3.83, "Texture — GLCM  (5)", color=BLUE, fs=8.5, fw="bold")

    tex_feats = [
        ("Entr.",  "Entropy"),
        ("Contr.", "Contrast"),
        ("Homog.", "Homogeneity"),
        ("Corr.",  "Correlation  ⚛"),
    ]
    for i, (abbr, full) in enumerate(tex_feats):
        yy = 3.5 - i * 0.41
        rect(ax, 3.22, yy-0.17, 0.82, 0.34, fc=BLUE_LT, ec=BLUE, lw=0.6, radius=0.005)
        txt(ax, 3.63, yy+0.003, abbr, color=BLUE, fs=7.5, fw="bold")
        txt(ax, 4.78, yy+0.003, full, color=TXT_MED, fs=7.0, ha="left")

    # Output badge
    rect(ax, 3.3, 1.88, 2.8, 0.4, fc=BLUE, ec=BLUE, lw=0, radius=0.007)
    txt(ax, 4.7, 2.09, "12-dim feature vector / image", color=WHITE, fs=7.8, fw="bold")

    # ════════════════════════════════════════════════════════════════════════
    # ARROW  B → C
    # ════════════════════════════════════════════════════════════════════════
    arrow_h(ax, 6.4, 4.95, 7.0, color=TXT_MED)
    txt(ax, 6.7, 5.25, "×1,549", color=TXT_LT, fs=7.5)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION C — MI Pre-Filter  (x: 7.0 – 9.6)
    # ════════════════════════════════════════════════════════════════════════
    rect(ax, 7.0, 1.8, 2.6, 5.6, fc=ORANGE_LT, ec=ORANGE, lw=1.4, radius=0.01)
    txt(ax, 8.3, 7.1, "MI Pre-Filter", color=ORANGE, fs=10, fw="bold")

    rect(ax, 7.12, 4.5, 2.36, 2.65, fc=WHITE, ec=ORANGE, lw=0.9, radius=0.006)
    txt(ax, 8.3, 6.9,  "Mutual Information", color=ORANGE, fs=8.5, fw="bold")
    txt(ax, 8.3, 6.55, "SelectKBest  (k = 6)", color=TXT_MED, fs=7.8)

    # MI formula
    rect(ax, 7.25, 5.68, 2.1, 0.52, fc=ORANGE_LT, ec=ORANGE, lw=0.6, radius=0.005)
    txt(ax, 8.3, 5.94, "I(X;Y) = H(Y) − H(Y|X)", color=TXT_DRK, fs=7.5, style="italic")

    top6 = ["1. Std_ExG    ⚛", "2. Mean_ExG", "3. Mean_RBR  ⚛",
            "4. Mean_NGRDI", "5. Mean_B      ⚛", "6. Correlation ⚛"]
    for i, f in enumerate(top6):
        yy = 5.35 - i * 0.32
        bold = "bold" if "⚛" in f else "normal"
        clr  = ORANGE if "⚛" in f else TXT_MED
        txt(ax, 8.3, yy, f.replace("⚛","★"), color=clr, fs=7.4, fw=bold)

    # Why box
    rect(ax, 7.12, 2.1, 2.36, 2.1, fc=WHITE, ec=ORANGE, lw=0.9, radius=0.006)
    txt(ax, 8.3, 3.95, "Why pre-filter?", color=ORANGE, fs=8.0, fw="bold")
    txt(ax, 8.3, 3.55, "VQC circuit depth\nscales 2ⁿ with qubits.\n12 → 6 reduces\ncost exponentially.", color=TXT_MED, fs=7.5)

    rect(ax, 7.24, 1.88, 2.12, 0.4, fc=ORANGE, ec=ORANGE, lw=0, radius=0.007)
    txt(ax, 8.3, 2.09, "Top-6 candidates", color=WHITE, fs=7.8, fw="bold")

    # ════════════════════════════════════════════════════════════════════════
    # ARROW  C → D
    # ════════════════════════════════════════════════════════════════════════
    arrow_h(ax, 9.6, 4.95, 10.2, color=TXT_MED)
    txt(ax, 9.9, 5.25, "6-dim", color=TXT_LT, fs=7.5)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION D — Quantum VQC  (x: 10.2 – 15.6)
    # ════════════════════════════════════════════════════════════════════════
    rect(ax, 10.2, 1.8, 5.4, 5.6, fc=PURPLE_LT, ec=PURPLE, lw=1.8, radius=0.01)
    txt(ax, 12.9, 7.1, "Variational Quantum Classifier  (VQC)", color=PURPLE, fs=10, fw="bold")

    # Top banner
    rect(ax, 10.35, 6.6, 5.1, 0.65, fc=WHITE, ec=PURPLE, lw=0.9, radius=0.006)
    txt(ax, 12.9, 6.93, "Evaluate all  C(6,4) = 15  feature subsets  →  select best by VQC accuracy",
        color=PURPLE, fs=8.5, fw="bold")

    # ── Quantum circuit ───────────────────────────────────────────────────
    rect(ax, 10.35, 3.35, 5.1, 3.0, fc=WHITE, ec=PURPLE, lw=0.9, radius=0.006)
    txt(ax, 12.9, 6.12, "4-Qubit Circuit  (reps = 1)", color=PURPLE, fs=8.5, fw="bold")

    q_names = ["q₀  Std_ExG", "q₁  Mean_RBR", "q₂  Mean_B", "q₃  Correlation"]
    q_colors= ["#7B1FA2","#1565C0","#2E7D32","#BF360C"]
    wire_y  = [5.72, 5.14, 4.57, 4.0]

    for qi, (qn, qc, wy) in enumerate(zip(q_names, q_colors, wire_y)):
        # Wire
        ax.plot([10.55, 15.35], [wy, wy], color="#BDBDBD", lw=1.2, zorder=2)
        # Label
        txt(ax, 10.55, wy+0.20, qn, color=qc, fs=7.2, fw="bold", ha="left")

        # H gate
        rect(ax, 10.98, wy-0.18, 0.40, 0.36, fc=BLUE_LT, ec=BLUE, lw=0.8, radius=0.004, zorder=4)
        txt(ax, 11.18, wy,  "H", color=BLUE, fs=7.5, fw="bold")

        # ZZ Rz gate
        rect(ax, 11.52, wy-0.18, 0.50, 0.36, fc=BLUE_LT, ec=BLUE, lw=0.8, radius=0.004, zorder=4)
        txt(ax, 11.77, wy, f"Rz", color=BLUE, fs=7.5, fw="bold")

        # RealAmp Ry gate
        rect(ax, 13.45, wy-0.18, 0.50, 0.36, fc=PURPLE_LT, ec=PURPLE, lw=0.8, radius=0.004, zorder=4)
        txt(ax, 13.70, wy, f"Ry", color=PURPLE, fs=7.5, fw="bold")

        # Measure
        rect(ax, 14.68, wy-0.18, 0.50, 0.36, fc=GREEN_LT, ec=GREEN, lw=0.8, radius=0.004, zorder=4)
        txt(ax, 14.93, wy, "M", color=GREEN, fs=7.5, fw="bold")

    # CNOT entangling gates (ZZ Feature Map)
    for i in range(3):
        cy = wire_y[i]; ty = wire_y[i+1]
        ax.plot([12.35, 12.35], [ty, cy], color=BLUE, lw=1.0, zorder=3, alpha=0.7)
        ax.plot(12.35, cy, 'o', color=BLUE, markersize=5, zorder=4)
        b = FancyBboxPatch((12.20, ty-0.15), 0.30, 0.30,
                           boxstyle="round,pad=0.02", fc=BLUE_LT, ec=BLUE, lw=0.8, zorder=4)
        ax.add_patch(b)
        txt(ax, 12.35, ty, "⊕", color=BLUE, fs=9, fw="bold")

    # Gate labels under wires
    for gx, gl, gc in [(11.18, "H\ngate", BLUE), (11.77, "ZZ\nmap", BLUE),
                       (12.35, "CNOT\nentangle", BLUE), (13.70, "Real\nAmpl.", PURPLE),
                       (14.93, "Meas.", GREEN)]:
        txt(ax, gx, 3.65, gl, color=gc, fs=6.5, style="italic")

    # Optimizer strip
    rect(ax, 10.35, 3.22, 5.1, 0.40, fc=PURPLE_LT, ec=PURPLE, lw=0.6, radius=0.005)
    txt(ax, 12.9, 3.42, "Optimizer: COBYLA  ·  Sampler: StatevectorSampler  ·  Training only — no quantum at inference",
        color=PURPLE, fs=7.5)

    # Result box
    rect(ax, 10.35, 2.0, 5.1, 1.0, fc=WHITE, ec=PURPLE, lw=0.9, radius=0.006)
    txt(ax, 12.9, 2.73, "Best Subset  (VQC acc = 72%)", color=PURPLE, fs=8.5, fw="bold")
    txt(ax, 12.9, 2.35, "Std_ExG   ·   Mean_RBR   ·   Mean_B   ·   Correlation",
        color=TXT_DRK, fs=9.5, fw="bold")

    rect(ax, 10.48, 1.88, 4.84, 0.40, fc=PURPLE, ec=PURPLE, lw=0, radius=0.007)
    txt(ax, 12.9, 2.09, "4 quantum-selected features  ·  Avg |Pearson r| = 0.596 (vs MI = 0.798)",
        color=WHITE, fs=7.8, fw="bold")

    # ════════════════════════════════════════════════════════════════════════
    # ARROW  D → E
    # ════════════════════════════════════════════════════════════════════════
    arrow_h(ax, 15.6, 4.95, 16.2, color=TXT_MED)
    txt(ax, 15.9, 5.25, "4-dim", color=TXT_LT, fs=7.5)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION E — Classical Deployment  (x: 16.2 – 19.6)
    # ════════════════════════════════════════════════════════════════════════
    rect(ax, 16.2, 1.8, 3.5, 5.6, fc=GREEN_LT, ec=GREEN, lw=1.4, radius=0.01)
    txt(ax, 17.95, 7.1, "Classical Inference", color=GREEN, fs=10, fw="bold")

    # SVM box
    rect(ax, 16.35, 4.55, 3.2, 2.65, fc=WHITE, ec=GREEN, lw=0.9, radius=0.006)
    txt(ax, 17.95, 6.97, "SVM-RBF Classifier", color=GREEN, fs=8.5, fw="bold")
    txt(ax, 17.95, 6.57, "kernel=rbf, C=10, γ=scale", color=TXT_MED, fs=7.8, style="italic")

    # Accuracy result
    rect(ax, 16.5, 5.6, 2.9, 0.70, fc=GREEN_LT, ec=GREEN, lw=0.7, radius=0.006)
    txt(ax, 17.95, 6.01, "5-Fold CV Accuracy", color=TXT_MED, fs=7.5)
    txt(ax, 17.95, 5.72, "100.0%  ±  0.00", color=GREEN, fs=11.5, fw="bold")

    txt(ax, 17.95, 5.27, "All 6 flight folders\nzero annotation required", color=TXT_MED, fs=7.5)

    # Output classes
    rect(ax, 16.35, 2.9, 3.2, 1.45, fc=WHITE, ec=GREEN, lw=0.9, radius=0.006)
    txt(ax, 17.95, 4.18, "Output Classes", color=GREEN, fs=8.5, fw="bold")

    rect(ax, 16.55, 3.6,  2.8, 0.48, fc=GREEN_LT, ec=GREEN, lw=0.7, radius=0.005)
    txt(ax, 17.95, 3.84, "🌿  Pre-Defoliation", color=GREEN, fs=8.5, fw="bold")

    rect(ax, 16.55, 3.05, 2.8, 0.48, fc=RED_LT, ec=RED, lw=0.7, radius=0.005)
    txt(ax, 17.95, 3.29, "🟫  Post-Defoliation", color=RED, fs=8.5, fw="bold")

    # No-quantum badge
    rect(ax, 16.35, 1.88, 3.2, 0.77, fc=GREEN, ec=GREEN, lw=0, radius=0.007)
    txt(ax, 17.95, 2.27, "⚛ Quantum used ONLY at training\nEdge-deployable classical model", color=WHITE, fs=7.8, fw="bold")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION LABELS (very bottom strip, like the drone multi-agent paper)
    # ════════════════════════════════════════════════════════════════════════
    axy = fig.add_axes([0.0, 0.0, 1.0, 0.09])
    axy.set_xlim(0, 20); axy.set_ylim(0, 1)
    axy.set_facecolor(BG); axy.axis("off")

    sections = [
        (1.18,  "(a) UAV Input",        TXT_MED, 0.14),
        (4.70,  "(b) Classical\nFeature Extraction", BLUE,  0.17),
        (8.30,  "(c) Mutual Info\nPre-Filter",        ORANGE, 0.13),
        (12.90, "(d) VQC Quantum\nFeature Selection",  PURPLE, 0.27),
        (17.95, "(e) Classical\nDeployment",  GREEN, 0.175),
    ]
    # Draw section dividers + labels
    dividers = [2.52, 6.45, 9.65, 15.65]
    for dv in dividers:
        axy.plot([dv, dv], [0.05, 0.95], color=GRAY_BRD, lw=1.0, ls=':')

    for sx, slbl, sc, _ in sections:
        axy.text(sx/20, 0.5, slbl, ha="center", va="center", color=sc,
                 fontsize=8.5, fontweight="bold", multialignment="center",
                 fontfamily="sans-serif")

    # ════════════════════════════════════════════════════════════════════════
    # TITLE strip
    # ════════════════════════════════════════════════════════════════════════
    axt = fig.add_axes([0.0, 0.94, 1.0, 0.06])
    axt.set_facecolor(BG); axt.axis("off")
    axt.text(0.5, 0.70,
             "Hybrid Quantum-Classical Feature Discovery for UAV-Based Cotton Defoliation Monitoring",
             ha="center", va="center", fontsize=14, fontweight="bold",
             color=TXT_DRK, fontfamily="sans-serif")
    axt.text(0.5, 0.18,
             "Quantum VQC identifies the minimal sufficient feature subset • No bounding box annotation • Classical inference only",
             ha="center", va="center", fontsize=9, color=TXT_MED,
             fontfamily="sans-serif", style="italic")

    # ════════════════════════════════════════════════════════════════════════
    # LEGEND  (top-right of main axes)
    # ════════════════════════════════════════════════════════════════════════
    legend_items = [
        (BLUE,   "Classical module"),
        (ORANGE, "MI pre-filter"),
        (PURPLE, "Quantum (training only)"),
        (GREEN,  "Classical inference"),
        (None,   "★ = quantum-selected"),
    ]
    lx, ly = 17.0, 7.42
    for i, (lc, ll) in enumerate(legend_items):
        if lc is not None:
            rect(ax, lx, ly - i*0.32, 0.24, 0.22, fc=lc, ec=lc, lw=0, radius=0.004, zorder=8)
        else:
            txt(ax, lx+0.12, ly - i*0.32 + 0.11, "★", color=ORANGE, fs=9, fw="bold")
        txt(ax, lx+0.34, ly - i*0.32 + 0.11, ll, color=TXT_MED, fs=7.5, ha="left")

    plt.savefig(OUT_PNG, dpi=220, bbox_inches="tight", facecolor=BG)
    print(f"✅  Saved: {OUT_PNG}")
    plt.close()

if __name__ == "__main__":
    build()
