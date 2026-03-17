#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'icml_features_FULL.csv')
OUT_PNG = os.path.join(HERE, 'neurips_biological_separability.png')

def generate_distribution_plot():
    df = pd.read_csv(CSV_PATH).dropna()
    
    # Setup aesthetic
    plt.style.use('default')
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#FAFAFA"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {"Pre_Defoliation": "#2E7D32", "Post_Defoliation": "#8D6E63"}
    
    # Panel 1: The "Perfect Separator" (Mean_ExG)
    sns.kdeplot(data=df, x='Mean_ExG', hue='Label', fill=True, palette=colors, 
                alpha=0.6, linewidth=2, ax=axes[0], warn_singular=False)
    
    axes[0].set_title('Biological Separability: Excess Green Index (ExG)', fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlabel('Mean ExG Value', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
    # Add vertical separation line
    axes[0].axvline(0.04, color='black', linestyle='--', alpha=0.5)
    axes[0].text(0.045, axes[0].get_ylim()[1]*0.8, "Perfect Linear\nDecision Boundary", fontsize=10, style='italic')
    
    # Panel 2: A "Harder" Feature (Correlation)
    sns.kdeplot(data=df, x='Correlation', hue='Label', fill=True, palette=colors, 
                alpha=0.6, linewidth=2, ax=axes[1], warn_singular=False)
    
    axes[1].set_title('Complex Feature Overlap: GLCM Texture Correlation', fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlabel('Texture Correlation', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Density', fontsize=12, fontweight='bold')
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(title='', labels=['Post-Defoliation (Brown/White)', 'Pre-Defoliation (Lush Green)'], 
                  loc='upper right', frameon=True, fontsize=10)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {OUT_PNG}")

if __name__ == "__main__":
    generate_distribution_plot()
