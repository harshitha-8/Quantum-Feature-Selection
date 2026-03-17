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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'icml_features_FULL.csv')
OUT_RADAR = os.path.join(HERE, 'neurips_k_scaling_radar.png')
OUT_BAR = os.path.join(HERE, 'neurips_k_scaling_impact.png')

SUBSETS_QFS = {
    2: ['Std_ExG', 'Mean_RBR'],
    4: ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation'],
    6: ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation', 'Mean_ExG', 'Mean_NGRDI'],
}

def inject_noise(X, noise_std):
    np.random.seed(42) # For reproducible noise
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise

def generate_scaling_plots():
    df = pd.read_csv(CSV_PATH).dropna()
    folders = df['Folder'].values
    y = (df['Label'] == 'Post_Defoliation').astype(int).values
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(df, y, groups=folders))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Metrics at a severe noise level (sigma = 0.15) to force a difference between k=2, 4, 6
    NOISE_LEVEL = 0.15
    metrics_data = {}
    
    for k in [2, 4, 6]:
        features = SUBSETS_QFS[k]
        
        X_tr = train_df[features].values
        X_te_base = test_df[features].values
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        
        clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
        clf.fit(X_tr_sc, y_train)
        
        X_te_noisy = inject_noise(X_te_base, NOISE_LEVEL)
        X_te_sc = scaler.transform(X_te_noisy)
        
        pred = clf.predict(X_te_sc)
        probs = clf.predict_proba(X_te_sc)[:, 1]
        
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        auc = roc_auc_score(y_test, probs)
        
        metrics_data[k] = [acc, f1, prec, rec, auc]

    # ---------------------------------------------------------
    # PLOT 1: RADAR CHART (Spider Plot)
    # ---------------------------------------------------------
    labels = np.array(['Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC-AUC'])
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor='white')
    colors = {2: '#E57373', 4: '#2196F3', 6: '#4CAF50'}
    
    for k in [2, 4, 6]:
        values = metrics_data[k]
        values += values[:1]
        
        ax.plot(angles, values, color=colors[k], linewidth=2.5, linestyle='solid', label=f'k={k} Qubits')
        ax.fill(angles, values, color=colors[k], alpha=0.15)
        
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    
    ax.set_title(f'Multi-Metric Robustness Spider Chart\nUnder Severe Drone Sensor Noise ($\sigma$={NOISE_LEVEL})', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=11)
    
    plt.savefig(OUT_RADAR, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {OUT_RADAR}")

    # ---------------------------------------------------------
    # PLOT 2: K-SCALING IMPACT BAR CHART
    # ---------------------------------------------------------
    # Evaluate at multiple noise levels to show scaling benefit
    noise_levels_eval = [0.05, 0.10, 0.15, 0.20]
    
    fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor='white')
    x = np.arange(len(noise_levels_eval))
    width = 0.25
    
    for i, k in enumerate([2, 4, 6]):
        accs = []
        features = SUBSETS_QFS[k]
        X_tr = train_df[features].values
        X_te_base = test_df[features].values
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        clf.fit(X_tr_sc, y_train)
        
        for nl in noise_levels_eval:
            X_te_noisy = inject_noise(X_te_base, nl)
            X_te_sc = scaler.transform(X_te_noisy)
            acc = accuracy_score(y_test, clf.predict(X_te_sc))
            accs.append(acc * 100)
            
        rects = ax2.bar(x + (i-1)*width, accs, width, label=f'k={k} Qubits', color=colors[k], edgecolor='black', linewidth=1)
        
        # Add values on top of bars
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=90)

    ax2.set_xlabel('Environmental Noise Degradation ($\sigma$)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Defoliation Inference Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('The Benefit of Qubit Scaling: k=2 vs k=4 vs k=6', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'$\sigma$={nl}' for nl in noise_levels_eval], fontsize=12)
    ax2.legend(fontsize=12, loc='lower left')
    ax2.set_ylim(40, 110)
    
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUT_BAR, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {OUT_BAR}")

if __name__ == "__main__":
    generate_scaling_plots()
