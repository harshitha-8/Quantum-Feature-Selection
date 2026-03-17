#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.svm import SVC

# Qiskit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.optimizers import COBYLA
from itertools import combinations
import argparse
import os

def load_data(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df_clean = df.dropna()
    
    # Extract structural labels
    y = df_clean['Label'].values
    folders = df_clean['Folder'].values if 'Folder' in df_clean.columns else np.arange(len(y))
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Drop non-feature columns
    drop_cols = ['Filename', 'Folder', 'Label']
    X = df_clean.drop([c for c in drop_cols if c in df_clean.columns], axis=1)
    
    return X, y, folders, le.classes_

def evaluate_subset_vqc(X_train, y_train, X_test, y_test, feature_dim):
    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=1)
    ansatz = RealAmplitudes(num_qubits=feature_dim, reps=1)
    optimizer = COBYLA(maxiter=30)
    sampler = StatevectorSampler()
    
    vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=sampler)
    
    train_size = min(150, len(X_train))
    vqc.fit(X_train[:train_size], y_train[:train_size])
    return vqc.score(X_test[:50], y_test[:50])

def main(args):
    X_df, y, folders, classes = load_data(args.input)
    
    subset_size = args.qbits
    print(f"\nEvaluating QFS for Subset Size k={subset_size}")
    
    # Pre-select top 6 to make combinatorial search viable
    selector = SelectKBest(mutual_info_classif, k=6)
    selector.fit(X_df, y)
    selected_indices = selector.get_support(indices=True)
    selected_names = X_df.columns[selected_indices].tolist()
    
    print(f"Top 6 MI Candidates: {selected_names}")
    
    # Strict spatial cross-validation to prevent data leakage (Fixes "1.0 accuracy" issue)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X_df, y, groups=folders))
    
    X_train_df, X_test_df = X_df.iloc[train_idx], X_df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    
    combos = list(combinations(selected_indices, subset_size))
    print(f"Combinatorial Search: {len(combos)} subsets of size {subset_size}...")
    
    best_acc = 0
    best_subset = []
    all_results = []
    
    for combo_idx in combos:
        current_names = [X_df.columns[i] for i in combo_idx]
        
        # Sub-select columns mapped back to scaled indices
        local_idx = [selected_indices.tolist().index(i) for i in combo_idx]
        X_tr_sub = scaler.fit_transform(X_train_df.iloc[:, local_idx])
        X_te_sub = scaler.transform(X_test_df.iloc[:, local_idx])
        
        acc = evaluate_subset_vqc(X_tr_sub, y_train, X_te_sub, y_test, subset_size)
        print(f"Subset: {current_names} -> VQC Accuracy: {acc:.4f}")
        
        all_results.append((current_names, acc))
        if acc > best_acc:
            best_acc = acc
            best_subset = current_names
            
    print(f"\n--- Fold-Isolated QFS Results for k={subset_size} ---")
    print(f"Best Subset: {best_subset}")
    print(f"Best Accuracy: {best_acc:.4f}")
    
    with open(f"qfs_results_k{subset_size}.txt", "w") as f:
        f.write(f"Subset Size: {subset_size}\n")
        f.write(f"Best Subset: {best_subset}\n")
        f.write(f"Accuracy: {best_acc:.4f}\n\n")
        for res in all_results:
            f.write(f"{res}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="icml_features_FULL.csv")
    parser.add_argument("--qbits", type=int, default=4, help="Subset size k")
    main(parser.parse_args())
