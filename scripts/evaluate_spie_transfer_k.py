#!/usr/bin/env python3
"""
Evaluate how SPIE-derived transfer features change QFS subset selection at k=2/4/6.

Workflow:
1. Load the augmented feature CSV built from raw ICML images.
2. Rank a candidate feature pool with mutual information.
3. Search best subsets of size k using the legacy VQC surrogate from the repo.
4. Measure downstream pre/post-defoliation detection with a classical SVM on
   group-isolated splits so folder leakage stays controlled.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_INPUT = ROOT / "icml_features_spie_augmented.csv"
RESULTS_DIR = ROOT / "results" / "spie_transfer"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NON_FEATURE_COLUMNS = {"Filename", "Folder", "Label"}


def load_data(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Label"].values)
    groups = df["Folder"].values
    feature_columns = [col for col in df.columns if col not in NON_FEATURE_COLUMNS]
    return df[feature_columns], y, groups


def make_svm() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
        ]
    )


def evaluate_subset_vqc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_dim: int,
    train_cap: int,
    test_cap: int,
) -> float:
    train_sel = stratified_cap_indices(y_train, train_cap)
    test_sel = stratified_cap_indices(y_test, test_cap)

    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=1)
    ansatz = RealAmplitudes(num_qubits=feature_dim, reps=1)
    optimizer = COBYLA(maxiter=30)
    sampler = StatevectorSampler()
    vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=sampler)
    vqc.fit(x_train[train_sel], y_train[train_sel])
    return float(vqc.score(x_test[test_sel], y_test[test_sel]))


def stratified_cap_indices(y: np.ndarray, cap: int) -> np.ndarray:
    if len(y) <= cap:
        return np.arange(len(y))

    rng = np.random.RandomState(42)
    chosen: list[int] = []
    classes = np.unique(y)
    per_class = max(1, cap // len(classes))

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        take = min(len(cls_idx), per_class)
        chosen.extend(rng.choice(cls_idx, size=take, replace=False).tolist())

    remaining = cap - len(chosen)
    if remaining > 0:
        available = np.array(sorted(set(range(len(y))) - set(chosen)))
        if len(available) > 0:
            extra_take = min(remaining, len(available))
            chosen.extend(rng.choice(available, size=extra_take, replace=False).tolist())

    return np.array(sorted(chosen))


def collect_valid_splits(x_df: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupShuffleSplit(n_splits=n_splits * 4, test_size=0.30, random_state=42)
    valid_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in splitter.split(x_df, y, groups=groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        valid_splits.append((train_idx, test_idx))
        if len(valid_splits) >= n_splits:
            break
    if not valid_splits:
        raise RuntimeError("Could not build any valid group-isolated split with both classes present.")
    return valid_splits


def search_best_subsets(
    x_df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    k_values: list[int],
    candidate_pool: int,
    n_eval_splits: int,
    train_cap: int,
    test_cap: int,
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    feature_names = x_df.columns.tolist()
    mi_scores = mutual_info_classif(x_df.values, y, random_state=42)
    ranked_features = [name for name, _ in sorted(zip(feature_names, mi_scores), key=lambda item: item[1], reverse=True)]
    candidate_features = ranked_features[: min(candidate_pool, len(ranked_features))]

    valid_splits = collect_valid_splits(x_df, y, groups, n_eval_splits)
    selected_subsets: dict[str, list[str]] = {}
    summary_rows: list[dict[str, object]] = []

    for k in k_values:
        combinations = list(itertools.combinations(candidate_features, k))
        if not combinations:
            continue
        print(f"Searching k={k} across {len(combinations)} subsets from candidate pool: {candidate_features}", flush=True)

        best_subset: tuple[str, ...] | None = None
        best_vqc_mean = float("-inf")

        for subset in combinations:
            split_scores = []
            subset_cols = list(subset)
            for train_idx, test_idx in valid_splits:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_df.iloc[train_idx][subset_cols].values)
                x_test = scaler.transform(x_df.iloc[test_idx][subset_cols].values)
                split_scores.append(
                    evaluate_subset_vqc(
                        x_train,
                        y[train_idx],
                        x_test,
                        y[test_idx],
                        feature_dim=k,
                        train_cap=min(train_cap, len(train_idx)),
                        test_cap=min(test_cap, len(test_idx)),
                    )
                )

            vqc_mean = float(np.mean(split_scores))
            if vqc_mean > best_vqc_mean:
                best_vqc_mean = vqc_mean
                best_subset = subset

        assert best_subset is not None
        selected_subsets[f"k={k}"] = list(best_subset)

        accs, f1s, aucs = [], [], []
        for train_idx, test_idx in valid_splits:
            clf = make_svm()
            x_train = x_df.iloc[train_idx][list(best_subset)].values
            x_test = x_df.iloc[test_idx][list(best_subset)].values
            clf.fit(x_train, y[train_idx])
            pred = clf.predict(x_test)
            prob = clf.predict_proba(x_test)[:, 1]
            accs.append(accuracy_score(y[test_idx], pred))
            f1s.append(f1_score(y[test_idx], pred, zero_division=0))
            aucs.append(roc_auc_score(y[test_idx], prob))

        summary_rows.append(
            {
                "k": k,
                "candidate_pool": candidate_features,
                "best_subset": list(best_subset),
                "vqc_mean_accuracy": best_vqc_mean,
                "svm_accuracy_mean": float(np.mean(accs)),
                "svm_accuracy_std": float(np.std(accs)),
                "svm_f1_mean": float(np.mean(f1s)),
                "svm_auc_mean": float(np.mean(aucs)),
            }
        )

    return summary_rows, selected_subsets


def write_outputs(summary_rows: list[dict[str, object]], selected_subsets: dict[str, list[str]]) -> tuple[Path, Path]:
    summary_path = RESULTS_DIR / "spie_transfer_k_summary.csv"
    subsets_path = RESULTS_DIR / "spie_transfer_selected_subsets.json"

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "k",
                "candidate_pool",
                "best_subset",
                "vqc_mean_accuracy",
                "svm_accuracy_mean",
                "svm_accuracy_std",
                "svm_f1_mean",
                "svm_auc_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    subsets_path.write_text(json.dumps(selected_subsets, indent=2))
    return summary_path, subsets_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--candidate-pool", type=int, default=8)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--train-cap", type=int, default=120)
    parser.add_argument("--test-cap", type=int, default=60)
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 4, 6])
    args = parser.parse_args()

    x_df, y, groups = load_data(args.input)
    summary_rows, selected_subsets = search_best_subsets(
        x_df=x_df,
        y=y,
        groups=groups,
        k_values=args.k_values,
        candidate_pool=args.candidate_pool,
        n_eval_splits=args.splits,
        train_cap=args.train_cap,
        test_cap=args.test_cap,
    )

    summary_path, subsets_path = write_outputs(summary_rows, selected_subsets)

    print(f"Input: {args.input}")
    for row in summary_rows:
        print(
            f"k={row['k']}: subset={row['best_subset']} | "
            f"VQC={row['vqc_mean_accuracy']:.4f} | "
            f"SVM acc={row['svm_accuracy_mean']:.4f} +/- {row['svm_accuracy_std']:.4f} | "
            f"F1={row['svm_f1_mean']:.4f} | AUC={row['svm_auc_mean']:.4f}"
        )
    print(f"Saved summary to {summary_path}")
    print(f"Saved subsets to {subsets_path}")


if __name__ == "__main__":
    main()
