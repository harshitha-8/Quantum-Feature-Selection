
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Qiskit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.optimizers import COBYLA

import time
import argparse

def load_and_preprocess(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Drop filename/folder columns if they exist
    drop_cols = ['Filename', 'Folder']
    df_clean = df.drop([c for c in drop_cols if c in df.columns], axis=1)
    
    # Encode Label
    le = LabelEncoder()
    df_clean['Label'] = le.fit_transform(df_clean['Label'])
    print(f"Classes: {le.classes_}")
    
    # Handle NaNs
    df_clean = df_clean.dropna()
    
    X = df_clean.drop('Label', axis=1)
    y = df_clean['Label']
    
    return X, y, le.classes_

def pre_filter_features(X, y, k=10):
    """
    Reduces feature space to top K using Mutual Information (Classical Filter).
    Quantum Feature Selection is expensive, so we cannot search 700+ LLM features.
    """
    print(f"Pre-filtering top {k} features using Mutual Information...")
    selector = SelectKBest(mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)
    
    selected_indices = selector.get_support(indices=True)
    selected_names = X.columns[selected_indices].tolist()
    
    return X_new, selected_names, selected_indices

def evaluate_subset_vqc(X_train, y_train, X_test, y_test, feature_dim):
    """
    Trains a VQC on the subset and returns accuracy.
    """
    # 1. Feature Map & Ansatz
    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=1)
    ansatz = RealAmplitudes(num_qubits=feature_dim, reps=1)
    
    # 2. Optimizer & Sampler
    optimizer = COBYLA(maxiter=40)
    sampler = StatevectorSampler()
    
    # 3. VQC
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler
    )
    
    # Train
    # Limit training size for speed
    train_size = min(200, len(X_train))
    vqc.fit(X_train[:train_size], y_train[:train_size])
    
    # Test
    acc = vqc.score(X_test[:50], y_test[:50])
    return acc

def main(args):
    # 1. Load Data
    X, y, classes = load_and_preprocess(args.input)
    
    # 2. Pre-filter (e.g., from 700 -> 8 best candidates)
    # We select a small number because VQC simulation scales exponentially with qubits
    # For a real TACC run, we could go higher, but for Mac demo, stay < 8-10.
    n_pre_select = 6 
    X_filtered, selected_names, _ = pre_filter_features(X, y, k=n_pre_select)
    
    print(f"Candidates for Quantum Selection: {selected_names}")
    
    # 3. Quantum wrapper Loop (Toy Example)
    # We will try to find the best subset of size 4 from these 6 candidates
    # using a VQC as the evaluator.
    
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.3, random_state=42)
    
    # Scale Data (crucial for Quantum Rotation Gates)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert y to numpy to avoid index errors in VQC
    y_train = y_train.values
    y_test = y_test.values
    
    print("\n--- Starting Quantum Feature Selection Wrapper ---")
    best_acc = 0
    best_subset = []
    
    # Combinatorial search (nCr)
    from itertools import combinations
    subset_size = 4
    
    combos = list(combinations(range(n_pre_select), subset_size))
    print(f"Evaluating {len(combos)} subsets of size {subset_size}...")
    
    results = []
    
    for indices in combos:
        indices = list(indices)
        current_names = [selected_names[i] for i in indices]
        
        # Train VQC on this subset
        acc = evaluate_subset_vqc(
            X_train[:, indices], y_train, 
            X_test[:, indices], y_test, 
            feature_dim=subset_size
        )
        
        print(f"Subset: {current_names} -> Accuracy: {acc:.4f}")
        results.append((current_names, acc))
        
        if acc > best_acc:
            best_acc = acc
            best_subset = current_names
            
    print("\n--- Results ---")
    print(f"Best Quantum Subset: {best_subset}")
    print(f"Best Accuracy: {best_acc:.4f}")
    
    # Save results
    with open("cotton_qfs_results.txt", "w") as f:
        f.write(f"Best Subset: {best_subset}\n")
        f.write(f"Accuracy: {best_acc}\n")
        f.write("All Candidates:\n")
        for res in results:
            f.write(f"{res}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="icml_cotton_features.csv", help="Input extracted features CSV")
    args = parser.parse_args()
    main(args)
