import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

def draw_scientific_architecture():
    # Setup ICML/NeurIPS aesthetic dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='#0B0E14')
    ax.set_facecolor('#0B0E14')
    
    # Fonts
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Arial']
    
    # Colors
    c_bg = '#0B0E14'
    c_panel = '#161B22'
    c_stroke = '#30363D'
    c_highlight = '#EAB308' # Gold
    c_quantum = '#A855F7' # Purple
    c_classical = '#38BDF8' # Blue
    c_input = '#10B981' # Green

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def draw_box(x, y, w, h, title, text, color, border_color):
        box = patches.Rectangle((x, y), w, h, fill=True, color=color, alpha=0.9, 
                                ec=border_color, lw=2, transform=ax.transData)
        ax.add_patch(box)
        plt.text(x + 2, y + h - 4, title, fontsize=14, color='white', fontweight='bold', ha='left')
        plt.text(x + 2, y + h - 10, text, fontsize=11, color='#C9D1D9', ha='left', va='top', linespacing=1.6)
        return x + w/2, y + h/2

    def draw_orthogonal_arrow(x1, y1, x2, y2, color, weight=2):
        path = Path([(x1, y1), (x1 + (x2-x1)/2, y1), (x1 + (x2-x1)/2, y2), (x2, y2)],
                    [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])
        patch = patches.PathPatch(path, facecolor='none', lw=weight, edgecolor=color)
        ax.add_patch(patch)
        # arrowhead
        ax.plot(x2, y2, marker='>', color=color, markersize=8)

    # 1. Inputs: Pre and Post Defoliation
    draw_box(2, 75, 20, 18, "Pre-Defoliation", "UAV RGB Imagery\nDense Canopy\nResolution: 4K", c_panel, c_input)
    draw_box(2, 50, 20, 18, "Post-Defoliation", "UAV RGB Imagery\nExposed Bolls\nLabel Generation", c_panel, c_input)

    # 2. Classical Feature Extraction Engine
    draw_box(30, 50, 25, 43, "Visual Feature Engineering", 
             "$X \\in \\mathbb{R}^{700}$\n\n• GLCM Texture Matrices\n  (Contrast, Energy, Entropy)\n• Color Space Histograms\n  (RGB, HSV, LAB, YCrCb)\n• Vegetation Indices\n  (ExG, NGRDI)\n• Spatial Correlations", 
             c_panel, c_classical)
    
    draw_orthogonal_arrow(22, 84, 30, 84, c_classical)
    draw_orthogonal_arrow(22, 59, 30, 59, c_classical)

    # 3. Dimensionality bottleneck / MI Filter
    draw_box(60, 75, 18, 12, "Mutual Information", "Top $k$ Candidate Pool\n$I(X;Y) = \\sum p(x,y)\\log\\frac{p}{p p}$", c_panel, c_stroke)
    draw_orthogonal_arrow(55, 71, 60, 81, c_classical)

    # 4. Hybrid Quantum Processor (VQC)
    draw_box(60, 30, 38, 40, "Variational Quantum Circuit (VQC)", 
             "Quantum Feature Evaluator\n$\\min_{\\theta} \\mathcal{L}(\\theta) = \\mathbb{E}[C(y, \\hat{y})]$\n\n1. State Preparation $|0\\rangle^{\\otimes n}$\n2. ZZFeatureMap (Data Encoding)\n   $U_{\\Phi(x)} = \\exp(i \\sum \\phi_S(x) Z_S)$\n3. RealAmplitudes Ansatz\n   $W(\\theta) = \\prod R_y(\\theta_i) CX$\n4. Measurement $\\langle Z \\rangle$", 
             c_panel, c_quantum)
    
    draw_orthogonal_arrow(69, 75, 69, 70, c_quantum)
    
    # 5. Output Model Pipeline
    draw_box(30, 10, 25, 18, "Combinatorial Loop", "Evaluates subsets mapping\nfeatures to qubits.\nOutput: $X_{opt} \\in \\mathbb{R}^4$", c_panel, c_quantum)
    draw_orthogonal_arrow(75, 30, 75, 19, c_quantum) # Feedback loop visualization part 1
    draw_orthogonal_arrow(75, 19, 55, 19, c_quantum) # Feedback loop visualization part 2

    # 6. Final Inference
    draw_box(60, 5, 38, 15, "Classical Support Vector Machine", "Kernel: $K(x,x') = \\exp(-\\gamma||x-x'||^2)$\nOptimal Defoliation Predictor", c_panel, c_highlight)
    
    draw_orthogonal_arrow(55, 19, 60, 12, c_highlight)


    # Title
    plt.text(50, 95, "Hybrid Quantum-Classical Architecture for UAV-based Defoliation Inference", 
             fontsize=22, color='white', fontweight='bold', ha='center')

    plt.savefig("rigorous_qfs_architecture.png", dpi=300, bbox_inches='tight', facecolor='#0B0E14')
    print("Graph generated at: rigorous_qfs_architecture.png")

if __name__ == "__main__":
    draw_scientific_architecture()
