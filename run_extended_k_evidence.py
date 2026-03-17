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
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'icml_features_FULL.csv')
OUT_ROC_PR = os.path.join(HERE, 'neurips_roc_pr_k2_4_6.png')
OUT_BOXPLOT = os.path.join(HERE, 'neurips_cv_stability_boxplot.png')

SUBSETS = {
    'QFS_k2': ['Std_ExG', 'Mean_RBR'],
    'MI_k2':  ['Std_ExG', 'Mean_ExG'],
    'QFS_k4': ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation'],
    'MI_k4':  ['Std_ExG', 'Mean_ExG', 'Mean_RBR', 'Mean_B'],
    'QFS_k6': ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation', 'Mean_ExG', 'Mean_NGRDI'],
    'MI_k6':  ['Std_ExG', 'Mean_ExG', 'Mean_RBR', 'Mean_B', 'Correlation', 'Mean_NGRDI'],
}

def load_dataset():
    df = pd.read_csv(CSV_PATH).dropna()
    folders = df['Folder'].values
    y = (df['Label'] == 'Post_Defoliation').astype(int).values # Positive class: Post_Defoliation
    return df, y, folders

def generate_roc_pr_plots():
    df, y, folders = load_dataset()
    
    # We use a single strict grouped split for the ROC/PR curve
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(df, y, groups=folders))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')
    
    colors = {'QFS': '#6A1B9A', 'MI': '#2E7D32'}
    
    k_vals = [2, 4, 6]
    
    for i, k in enumerate(k_vals):
        ax_roc = axes[0, i]
        ax_pr = axes[1, i]
        
        for method in ['QFS', 'MI']:
            features = SUBSETS[f'{method}_k{k}']
            
            X_tr = train_df[features].values
            X_te = test_df[features].values
            
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)
            
            clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
            clf.fit(X_tr_sc, y_train)
            
            y_probs = clf.predict_proba(X_te_sc)[:, 1]
            
            # ROC
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            
            ax_roc.plot(fpr, tpr, color=colors[method], lw=2.5 if method=='QFS' else 2.0, 
                        linestyle='-' if method=='QFS' else '--',
                        label=f'{method} (AUC = {roc_auc:.3f})')
            
            # PR
            precision, recall, _ = precision_recall_curve(y_test, y_probs)
            pr_auc = average_precision_score(y_test, y_probs)
            
            ax_pr.plot(recall, precision, color=colors[method], lw=2.5 if method=='QFS' else 2.0, 
                       linestyle='-' if method=='QFS' else '--',
                       label=f'{method} (AP = {pr_auc:.3f})')
        
        # ROC Formatting
        ax_roc.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
        ax_roc.set_xlim([-0.02, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate', fontsize=12)
        ax_roc.set_ylabel('True Positive Rate', fontsize=12)
        ax_roc.set_title(f'ROC Curve (k={k} Qubits)', fontsize=14, fontweight='bold')
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, linestyle='--', alpha=0.5)
        ax_roc.spines['top'].set_visible(False)
        ax_roc.spines['right'].set_visible(False)
        
        # PR Formatting
        ax_pr.set_xlim([-0.02, 1.02])
        ax_pr.set_ylim([0.4, 1.05])
        ax_pr.set_xlabel('Recall', fontsize=12)
        ax_pr.set_ylabel('Precision', fontsize=12)
        ax_pr.set_title(f'Precision-Recall Curve (k={k} Qubits)', fontsize=14, fontweight='bold')
        ax_pr.legend(loc="lower left")
        ax_pr.grid(True, linestyle='--', alpha=0.5)
        ax_pr.spines['top'].set_visible(False)
        ax_pr.spines['right'].set_visible(False)
        
    plt.tight_layout(pad=3.0)
    fig.suptitle('Classification Performance Profiles Across Quantum & Classical Dimensionalities', 
                 fontsize=18, fontweight='bold', y=1.03)
                 
    plt.savefig(OUT_ROC_PR, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {OUT_ROC_PR}")

def generate_stability_boxplot():
    df, y, folders = load_dataset()
    
    # Stratified Group K-Fold for strict cross-folder evaluation
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    
    for k in [2, 4, 6]:
        for method in ['QFS', 'MI']:
            features = SUBSETS[f'{method}_k{k}']
            accs = []
            
            for train_idx, test_idx in sgkf.split(df, y, groups=folders):
                X_tr = df.iloc[train_idx][features].values
                X_te = df.iloc[test_idx][features].values
                
                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr)
                X_te_sc = scaler.transform(X_te)
                
                clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
                clf.fit(X_tr_sc, y[train_idx])
                
                pred = clf.predict(X_te_sc)
                accs.append(accuracy_score(y[test_idx], pred))
                
            for acc in accs:
                results.append({
                    'Method': 'Quantum VQC' if method == 'QFS' else 'Classical MI',
                    'k': f'k={k}',
                    'Accuracy': acc * 100
                })
                
    res_df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.gca()
    
    # Seaborn Boxplot (NeurIPS Style)
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    
    sns.boxplot(x='k', y='Accuracy', hue='Method', data=res_df, palette=['#9C27B0', '#4CAF50'], 
                width=0.6, linewidth=1.5, ax=ax, showmeans=True, 
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"})
    
    sns.swarmplot(x='k', y='Accuracy', hue='Method', data=res_df, dodge=True, color='0.2', size=5, ax=ax, alpha=0.6)
    
    ax.set_title('Cross-Folder Generalization Stability (5-Fold CV)', fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel('Inference Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Feature Subset Size (# Qubits)', fontsize=13, fontweight='bold')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title='', loc='lower right', fontsize=11)
    
    ax.set_ylim(40, 105)
    sns.despine()
    
    plt.savefig(OUT_BOXPLOT, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {OUT_BOXPLOT}")

if __name__ == "__main__":
    generate_roc_pr_plots()
    generate_stability_boxplot()
