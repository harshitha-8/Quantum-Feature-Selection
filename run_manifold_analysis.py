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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'icml_features_FULL.csv')
OUT_MANIFOLD = os.path.join(HERE, 'neurips_feature_manifold_k2_4_6.png')
OUT_VIOLIN = os.path.join(HERE, 'neurips_exg_glcm_distribution.png')

SUBSETS = {
    'QFS_k2': ['Std_ExG', 'Mean_RBR'],
    'MI_k2':  ['Std_ExG', 'Mean_ExG'],
    'QFS_k4': ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation'],
    'MI_k4':  ['Std_ExG', 'Mean_ExG', 'Mean_RBR', 'Mean_B'],
    'QFS_k6': ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation', 'Mean_ExG', 'Mean_NGRDI'],
    'MI_k6':  ['Std_ExG', 'Mean_ExG', 'Mean_RBR', 'Mean_B', 'Correlation', 'Mean_NGRDI'],
}

def plot_manifolds():
    df = pd.read_csv(CSV_PATH).dropna()
    y_labels = df['Label'].values
    y_colors = np.array(['#2E7D32' if l == 'Pre_Defoliation' else '#8D6E63' for l in y_labels])
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 18), facecolor='white')
    k_vals = [2, 4, 6]
    methods = ['QFS', 'MI']
    
    for i, k in enumerate(k_vals):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            features = SUBSETS[f'{method}_k{k}']
            
            X = df[features].values
            X_scaled = StandardScaler().fit_transform(X)
            
            # Project to 2D for visualization
            if X_scaled.shape[1] > 2:
                pca = PCA(n_components=2, random_state=42)
                X_proj = pca.fit_transform(X_scaled)
                xlabel, ylabel = 'Principal Component 1', 'Principal Component 2'
            else:
                X_proj = X_scaled
                xlabel, ylabel = features[0], features[1]
                
            # Density Contour (KDE)
            sns.kdeplot(x=X_proj[:, 0], y=X_proj[:, 1], hue=y_labels, fill=True, 
                        palette={'Pre_Defoliation': '#A5D6A7', 'Post_Defoliation': '#D7CCC8'},
                        alpha=0.5, ax=ax, legend=False, levels=5)
            
            # Scatter Points
            ax.scatter(X_proj[:, 0], X_proj[:, 1], c=y_colors, edgecolor='w', s=30, alpha=0.8, linewidth=0.5)
            
            ax.set_title(f'{method} Subset Manifold (k={k} Qubits)\nFeatures: {", ".join(features).replace("Correlation", "GLCM_Corr")}', 
                         fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout(pad=4.0)
    fig.suptitle('2D Feature Space Separability Manifolds: Quantum vs Classical Dimension Scaling', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    plt.savefig(OUT_MANIFOLD, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {OUT_MANIFOLD}")


def plot_exg_glcm_violins():
    df = pd.read_csv(CSV_PATH).dropna()
    
    # We want to show how GLCM Correlation and Std_ExG specifically contrast in the distributions
    features_to_plot = ['Std_ExG', 'Mean_ExG', 'Correlation', 'Homogeneity']
    
    # Melt dataframe for Seaborn
    df_melted = pd.melt(df, id_vars=['Label'], value_vars=features_to_plot, 
                        var_name='Feature', value_name='Value')
                        
    # Normalize values per feature for side-by-side comparison
    df_melted['Scaled_Value'] = df_melted.groupby('Feature')['Value'].transform(lambda x: (x - x.mean()) / x.std())

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    
    sns.violinplot(x='Feature', y='Scaled_Value', hue='Label', data=df_melted, split=True,
                   inner="quart", linewidth=1.5, palette={'Pre_Defoliation': '#2E7D32', 'Post_Defoliation': '#8D6E63'}, ax=ax)
                   
    ax.set_title('Standardized Class Distributions: Vegetation ExG vs GLCM Texture Indices', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel('Standardized Feature Value (Z-Score)', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    
    ax.legend(title='', labels=['Pre-Defoliation (Lush Green)', 'Post-Defoliation (Brown/White)'], 
              loc='upper right', frameon=True, fontsize=11)
              
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUT_VIOLIN, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {OUT_VIOLIN}")

if __name__ == "__main__":
    plot_manifolds()
    plot_exg_glcm_violins()
