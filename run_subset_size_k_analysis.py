#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'icml_features_FULL.csv')
OUT_PNG_NOISE = os.path.join(HERE, 'neurips_k_subset_noise_analysis.png')
OUT_PNG_HEATMAP = os.path.join(HERE, 'neurips_k_subset_redundancy.png')

# The optimal subsets found empirically from QFS vs MI
SUBSETS = {
    'QFS_k2': ['Std_ExG', 'Mean_RBR'],
    'MI_k2':  ['Std_ExG', 'Mean_ExG'],
    'QFS_k4': ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation'],
    'MI_k4':  ['Std_ExG', 'Mean_ExG', 'Mean_RBR', 'Mean_B'],
    'QFS_k6': ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation', 'Mean_ExG', 'Mean_NGRDI'],
    'MI_k6':  ['Std_ExG', 'Mean_ExG', 'Mean_RBR', 'Mean_B', 'Correlation', 'Mean_NGRDI'],
}

def inject_noise(X, noise_std):
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise

def generate_noise_plots():
    print("Loading empirical data for rigorous K-subset analysis...")
    df = pd.read_csv(CSV_PATH).dropna()
    folders = df['Folder'].values
    y = (df['Label'] == 'Pre_Defoliation').astype(int).values
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(df, y, groups=folders))
    
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    results = {k: [] for k in SUBSETS.keys()}
    
    for subset_name, features in SUBSETS.items():
        X_tr = train_df[features].values
        X_te_base = test_df[features].values
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        
        clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        clf.fit(X_tr_sc, y[train_idx])
        
        for nl in noise_levels:
            # Inject noise *after* feature extraction to simulate degraded sensor reads
            X_te_noisy = inject_noise(X_te_base, nl)
            X_te_sc = scaler.transform(X_te_noisy)
            
            pred = clf.predict(X_te_sc)
            acc = accuracy_score(y[test_idx], pred)
            results[subset_name].append(acc)

    # ---------------------------------------------------------
    # PLOT 1: Noise Robustness (NeurIPS Style)
    # ---------------------------------------------------------
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor('white')
    
    colors = {
        'QFS_k2': '#7B1FA2', 'MI_k2': '#E1BEE7',
        'QFS_k4': '#1976D2', 'MI_k4': '#BBDEFB',
        'QFS_k6': '#388E3C', 'MI_k6': '#C8E6C9'
    }
    
    markers = {'QFS_k2': 'o', 'MI_k2': 'x', 'QFS_k4': 's', 'MI_k4': '^', 'QFS_k6': 'D', 'MI_k6': 'v'}
    
    for name, accs in results.items():
        ls = '-' if 'QFS' in name else '--'
        lw = 2.5 if 'QFS' in name else 1.5
        ms = 8 if 'QFS' in name else 6
        label = name.replace('_', ' (Quantum) ' if 'QFS' in name else ' (Classical) ')
        ax.plot(noise_levels, [a*100 for a in accs], marker=markers[name], color=colors[name], 
                linestyle=ls, linewidth=lw, markersize=ms, label=label)
    
    ax.set_xlabel(r'Gaussian Noise Standard Deviation ($\sigma$)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Inference Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Robustness of Quantum vs Classical Feature Subsets over $k \in \{2, 4, 6\}$', fontsize=15, fontweight='bold', pad=15)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Shade structural collapse region
    ax.axvspan(0.15, 0.25, color='#FFF3E0', alpha=0.5, zorder=0)
    ax.text(0.20, 55, "Extreme Sensor\nDegradation Zone", color='#E65100', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylim(45, 105)
    ax.legend(loc='lower left', frameon=True, fontsize=10, ncol=2)
    
    plt.savefig(OUT_PNG_NOISE, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {OUT_PNG_NOISE}")

    # ---------------------------------------------------------
    # PLOT 2: Feature Redundancy Bar Chart
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(9, 5), facecolor='white')
    
    bars_qfs = []
    bars_mi = []
    k_labels = ['k = 2', 'k = 4', 'k = 6']
    
    for k_val in [2, 4, 6]:
        # Calculate mean absolute Pearson correlation
        df_qfs = df[SUBSETS[f'QFS_k{k_val}']].corr().abs().values
        mean_r_qfs = df_qfs[np.triu_indices_from(df_qfs, k=1)].mean()
        bars_qfs.append(mean_r_qfs)
        
        df_mi = df[SUBSETS[f'MI_k{k_val}']].corr().abs().values
        mean_r_mi = df_mi[np.triu_indices_from(df_mi, k=1)].mean()
        bars_mi.append(mean_r_mi)
        
    x = np.arange(len(k_labels))
    width = 0.35
    
    rects1 = ax2.bar(x - width/2, bars_qfs, width, label='Quantum VQC (Ours)', color='#5E35B1', edgecolor='black', linewidth=1)
    rects2 = ax2.bar(x + width/2, bars_mi, width, label='Mutual Information (Baseline)', color='#81C784', edgecolor='black', linewidth=1)
    
    ax2.set_ylabel(r'Average |Pearson $r$| (Lower is better)', fontsize=13, fontweight='bold')
    ax2.set_title('Feature Redundancy (Multi-Collinearity) Across Subset Dimensions', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(k_labels, fontsize=12)
    ax2.legend(fontsize=11)
    
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    plt.savefig(OUT_PNG_HEATMAP, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {OUT_PNG_HEATMAP}")

if __name__ == "__main__":
    generate_noise_plots()
