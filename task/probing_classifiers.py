#!/usr/bin/env python
"""
Train probing classifiers on molecular embeddings to evaluate learned representations.

This script trains simple linear probes (logistic regression for classification,
linear regression for continuous targets) on extracted embeddings to evaluate
what molecular properties are captured in the latent space.

================================================================================
What is Probing?
================================================================================
Probing is a technique to understand what information is encoded in learned
representations. By training simple classifiers to predict properties from
embeddings, we can assess whether the model has learned to capture those properties.

- High accuracy → The property is well-encoded in the latent space
- Low accuracy → The property is not captured (or not linearly separable)

================================================================================
Embedding Types:
================================================================================
The extract_embeddings.py script produces four embedding types:
- atom_from_atom: Atom embeddings from atom-centric message passing
- atom_from_bond: Atom embeddings from bond-centric message passing
- bond_from_atom: Bond embeddings from atom-centric message passing
- bond_from_bond: Bond embeddings from bond-centric message passing

You can select which embedding to use with --embedding_type, or use "concat" to
concatenate all four, or "mean" to average them.

================================================================================
Supported Tasks:
================================================================================
Classification:
    - NumRings, NumAromaticRings (discrete counts)
    - Lipinski, Veber (binary druglikeness rules)
    - Custom categorical properties

Regression:
    - MolecularWeight, LogP, TPSA (continuous)
    - Custom continuous properties

================================================================================
Usage:
================================================================================
    # Train probes using atom_from_atom embeddings (default)
    python task/probing_classifiers.py \
        --embeddings embeddings_dir/ \
        --properties properties.csv \
        --output_dir probe_results/

    # Use concatenated embeddings (all four types)
    python task/probing_classifiers.py \
        --embeddings embeddings_dir/ \
        --properties properties.csv \
        --output_dir probe_results/ \
        --embedding_type concat

    # Use averaged embeddings
    python task/probing_classifiers.py \
        --embeddings embeddings_dir/ \
        --properties properties.csv \
        --output_dir probe_results/ \
        --embedding_type mean

    # Train for specific properties
    python task/probing_classifiers.py \
        --embeddings embeddings_dir/ \
        --properties properties.csv \
        --output_dir probe_results/ \
        --targets LogP TPSA NumRings Lipinski

    # Custom classification with binning
    python task/probing_classifiers.py \
        --embeddings embeddings_dir/ \
        --properties properties.csv \
        --output_dir probe_results/ \
        --targets MolecularWeight \
        --bin_continuous 5
"""

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import numpy as np
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score, mean_squared_error,
    mean_absolute_error
)

from embedding_utils import EMBEDDING_TYPES, load_embeddings, load_properties

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)


# Default properties for probing (from molecular_properties.py)
DEFAULT_CLASSIFICATION_TARGETS = [
    "NumRings", "NumAromaticRings", "NumHDonors", "NumHAcceptors",
    "Lipinski", "Veber"
]

DEFAULT_REGRESSION_TARGETS = [
    "MolecularWeight", "LogP", "TPSA", "FractionCSP3", "BertzCT"
]


@dataclass
class ProbeResult:
    """Results from training a probing classifier."""
    target: str
    task_type: str  # "classification" or "regression"
    n_samples: int
    n_classes: Optional[int] = None  # For classification
    
    # Metrics
    train_score: float = 0.0
    test_score: float = 0.0
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    # Classification-specific
    accuracy: Optional[float] = None
    f1_macro: Optional[float] = None
    f1_weighted: Optional[float] = None
    
    # Regression-specific
    r2: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    
    # Additional info
    class_distribution: Optional[Dict[str, int]] = None


def bin_continuous_values(
    values: np.ndarray,
    n_bins: int = 5,
    strategy: str = "quantile"
) -> Tuple[np.ndarray, List[str]]:
    """
    Bin continuous values into discrete categories.

    Args:
        values: Array of continuous values
        n_bins: Number of bins
        strategy: "quantile" or "uniform"

    Returns:
        Tuple of (binned labels, bin names)
    """
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    
    if strategy == "quantile":
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(valid_values, percentiles)
    else:  # uniform
        bin_edges = np.linspace(valid_values.min(), valid_values.max(), n_bins + 1)
    
    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)
    n_bins = len(bin_edges) - 1
    
    # Create bin labels
    binned = np.full_like(values, np.nan)
    binned[valid_mask] = np.clip(
        np.digitize(valid_values, bin_edges[1:-1]),
        0, n_bins - 1
    )
    
    # Create bin names
    bin_names = []
    for i in range(n_bins):
        if i == 0:
            bin_names.append(f"<{bin_edges[1]:.2g}")
        elif i == n_bins - 1:
            bin_names.append(f"≥{bin_edges[-2]:.2g}")
        else:
            bin_names.append(f"{bin_edges[i]:.2g}-{bin_edges[i+1]:.2g}")
    
    return binned, bin_names


def train_classification_probe(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[ProbeResult, Any]:
    """
    Train a classification probe.

    Args:
        X: Features (embeddings)
        y: Labels
        cv_folds: Number of cross-validation folds
        test_size: Test set size for final evaluation
        random_state: Random seed

    Returns:
        Tuple of (ProbeResult, trained model)
    """
    # Remove NaN samples
    valid_mask = ~np.isnan(y)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask].astype(int)
    
    if len(np.unique(y_valid)) < 2:
        raise ValueError("Need at least 2 classes for classification")
    
    # Merge rare classes (< 5 samples) into neighboring class
    unique, counts = np.unique(y_valid, return_counts=True)
    rare_classes = unique[counts < 5]
    if len(rare_classes) > 0:
        for rare_class in rare_classes:
            # Find nearest non-rare class
            non_rare = unique[counts >= 5]
            if len(non_rare) == 0:
                raise ValueError("All classes have fewer than 5 samples")
            nearest = non_rare[np.argmin(np.abs(non_rare - rare_class))]
            y_valid[y_valid == rare_class] = nearest
        # Update unique counts after merging
        unique, counts = np.unique(y_valid, return_counts=True)
    
    if len(np.unique(y_valid)) < 2:
        raise ValueError("After merging rare classes, fewer than 2 classes remain")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    
    # Train/test split (use stratify only if all classes have >= 2 samples)
    min_class_count = counts.min()
    use_stratify = min_class_count >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_valid, test_size=test_size, random_state=random_state,
        stratify=y_valid if use_stratify else None
    )
    
    # Train logistic regression
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
        solver="lbfgs",
    )
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y_valid, cv=cv_folds, scoring="accuracy")
    
    # Final training and evaluation
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute metrics
    result = ProbeResult(
        target="",  # Will be set by caller
        task_type="classification",
        n_samples=len(y_valid),
        n_classes=len(np.unique(y_valid)),
        train_score=model.score(X_train, y_train),
        test_score=model.score(X_test, y_test),
        cv_scores=cv_scores.tolist(),
        cv_mean=cv_scores.mean(),
        cv_std=cv_scores.std(),
        accuracy=accuracy_score(y_test, y_pred),
        f1_macro=f1_score(y_test, y_pred, average="macro", zero_division=0),
        f1_weighted=f1_score(y_test, y_pred, average="weighted", zero_division=0),
    )
    
    # Class distribution
    unique, counts = np.unique(y_valid, return_counts=True)
    result.class_distribution = {str(int(u)): int(c) for u, c in zip(unique, counts)}
    
    return result, (model, scaler)


def train_regression_probe(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    alpha: float = 1.0,
) -> Tuple[ProbeResult, Any]:
    """
    Train a regression probe.

    Args:
        X: Features (embeddings)
        y: Target values
        cv_folds: Number of cross-validation folds
        test_size: Test set size
        random_state: Random seed
        alpha: Ridge regression regularization

    Returns:
        Tuple of (ProbeResult, trained model)
    """
    # Remove NaN samples
    valid_mask = ~np.isnan(y)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    # Scale features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_valid)
    y_scaled = scaler_y.fit_transform(y_valid.reshape(-1, 1)).ravel()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state
    )
    
    # Train Ridge regression
    model = Ridge(alpha=alpha, random_state=random_state)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y_scaled, cv=cv_folds, scoring="r2")
    
    # Final training and evaluation
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Inverse transform for metrics in original scale
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    
    result = ProbeResult(
        target="",  # Will be set by caller
        task_type="regression",
        n_samples=len(y_valid),
        train_score=model.score(X_train, y_train),
        test_score=model.score(X_test, y_test),
        cv_scores=cv_scores.tolist(),
        cv_mean=cv_scores.mean(),
        cv_std=cv_scores.std(),
        r2=r2_score(y_test_orig, y_pred_orig),
        mse=mean_squared_error(y_test_orig, y_pred_orig),
        mae=mean_absolute_error(y_test_orig, y_pred_orig),
    )
    
    return result, (model, scaler_X, scaler_y)


def run_probing(
    embeddings: np.ndarray,
    properties: Dict[str, np.ndarray],
    targets: Optional[List[str]] = None,
    task_types: Optional[Dict[str, str]] = None,
    cv_folds: int = 5,
    bin_continuous: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, ProbeResult]:
    """
    Run probing experiments for multiple targets.

    Args:
        embeddings: Embedding array [N, latent_dim]
        properties: Dictionary of property arrays
        targets: List of targets to probe (None = all)
        task_types: Dict mapping target to "classification" or "regression"
        cv_folds: Number of CV folds
        bin_continuous: If set, bin continuous targets into this many classes
        random_state: Random seed

    Returns:
        Dictionary of ProbeResults
    """
    if targets is None:
        targets = list(properties.keys())
    
    # Auto-detect task types if not provided
    if task_types is None:
        task_types = {}
        for target in targets:
            if target not in properties:
                continue
            values = properties[target]
            valid_values = values[~np.isnan(values)]
            
            # Heuristic: if fewer than 20 unique values, treat as classification
            n_unique = len(np.unique(valid_values))
            if n_unique <= 20 or target in DEFAULT_CLASSIFICATION_TARGETS:
                task_types[target] = "classification"
            else:
                task_types[target] = "regression"
    
    results = {}
    
    for target in targets:
        if target not in properties:
            print(f"Warning: Target '{target}' not found in properties, skipping")
            continue
        
        y = properties[target]
        valid_count = np.sum(~np.isnan(y))
        
        if valid_count < 50:
            print(f"Warning: Target '{target}' has only {valid_count} valid samples, skipping")
            continue
        
        task_type = task_types.get(target, "regression")
        
        # Optionally bin continuous targets
        if task_type == "regression" and bin_continuous is not None:
            y, bin_names = bin_continuous_values(y, n_bins=bin_continuous)
            task_type = "classification"
            print(f"  Binned '{target}' into {len(bin_names)} classes: {bin_names}")
        
        print(f"\nTraining probe for '{target}' ({task_type})...")
        
        try:
            if task_type == "classification":
                result, model = train_classification_probe(
                    embeddings, y, cv_folds=cv_folds, random_state=random_state
                )
            else:
                result, model = train_regression_probe(
                    embeddings, y, cv_folds=cv_folds, random_state=random_state
                )
            
            result.target = target
            results[target] = result
            
            # Print summary
            if task_type == "classification":
                print(f"  Samples: {result.n_samples}, Classes: {result.n_classes}")
                print(f"  CV Accuracy: {result.cv_mean:.3f} ± {result.cv_std:.3f}")
                print(f"  Test Accuracy: {result.accuracy:.3f}, F1 (macro): {result.f1_macro:.3f}")
            else:
                print(f"  Samples: {result.n_samples}")
                print(f"  CV R²: {result.cv_mean:.3f} ± {result.cv_std:.3f}")
                print(f"  Test R²: {result.r2:.3f}, MAE: {result.mae:.3f}")
        
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return results


def save_results(
    results: Dict[str, ProbeResult],
    output_dir: str,
    format: str = "all"
):
    """
    Save probing results.

    Args:
        results: Dictionary of ProbeResults
        output_dir: Output directory
        format: "json", "csv", or "all"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to dicts
    results_dict = {k: asdict(v) for k, v in results.items()}
    
    if format in ["json", "all"]:
        json_path = output_dir / "probe_results.json"
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to {json_path}")
    
    if format in ["csv", "all"]:
        csv_path = output_dir / "probe_results.csv"
        
        # Define all possible fieldnames upfront
        fieldnames = [
            "target", "task_type", "n_samples", "n_classes",
            "cv_mean", "cv_std", "test_score",
            # Classification metrics
            "accuracy", "f1_macro", "f1_weighted",
            # Regression metrics
            "r2", "mse", "mae"
        ]
        
        # Flatten for CSV
        rows = []
        for target, result in results.items():
            row = {
                "target": target,
                "task_type": result.task_type,
                "n_samples": result.n_samples,
                "n_classes": result.n_classes,
                "cv_mean": result.cv_mean,
                "cv_std": result.cv_std,
                "test_score": result.test_score,
                # Initialize all metrics as None
                "accuracy": None,
                "f1_macro": None,
                "f1_weighted": None,
                "r2": None,
                "mse": None,
                "mae": None,
            }
            
            if result.task_type == "classification":
                row["accuracy"] = result.accuracy
                row["f1_macro"] = result.f1_macro
                row["f1_weighted"] = result.f1_weighted
            else:
                row["r2"] = result.r2
                row["mse"] = result.mse
                row["mae"] = result.mae
            
            rows.append(row)
        
        if rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Results saved to {csv_path}")


def print_summary(results: Dict[str, ProbeResult]):
    """Print a summary table of probing results."""
    print(f"\n{'=' * 80}")
    print("PROBING RESULTS SUMMARY")
    print(f"{'=' * 80}")
    
    # Classification results
    clf_results = {k: v for k, v in results.items() if v.task_type == "classification"}
    if clf_results:
        print("\n--- Classification Tasks ---")
        print(f"{'Target':<25} {'Samples':>8} {'Classes':>8} {'CV Acc':>10} {'Test Acc':>10} {'F1 macro':>10}")
        print("-" * 80)
        for target, r in sorted(clf_results.items(), key=lambda x: -x[1].cv_mean):
            print(f"{target:<25} {r.n_samples:>8} {r.n_classes:>8} "
                  f"{r.cv_mean:>9.3f} {r.accuracy:>10.3f} {r.f1_macro:>10.3f}")
    
    # Regression results
    reg_results = {k: v for k, v in results.items() if v.task_type == "regression"}
    if reg_results:
        print("\n--- Regression Tasks ---")
        print(f"{'Target':<25} {'Samples':>8} {'CV R²':>10} {'Test R²':>10} {'MAE':>12}")
        print("-" * 80)
        for target, r in sorted(reg_results.items(), key=lambda x: -x[1].cv_mean):
            print(f"{target:<25} {r.n_samples:>8} {r.cv_mean:>10.3f} {r.r2:>10.3f} {r.mae:>12.3f}")
    
    print(f"\n{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Train probing classifiers on molecular embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input files
    parser.add_argument(
        "--embeddings", "-e",
        type=str,
        required=True,
        help="Path to embeddings directory or file",
    )
    parser.add_argument(
        "--properties", "-p",
        type=str,
        required=True,
        help="Path to properties CSV file",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="atom_from_atom",
        choices=EMBEDDING_TYPES + ["concat", "mean"],
        help="Which embedding type to use: atom_from_atom, atom_from_bond, "
             "bond_from_atom, bond_from_bond, concat (all 4), or mean (average)",
    )
    
    # Output
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Output directory for results",
    )
    
    # Target selection
    parser.add_argument(
        "--targets", "-t",
        nargs="+",
        default=None,
        help="Target properties to probe (default: all)",
    )
    parser.add_argument(
        "--classification_only",
        action="store_true",
        help="Only run classification probes",
    )
    parser.add_argument(
        "--regression_only",
        action="store_true",
        help="Only run regression probes",
    )
    
    # Options
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--bin_continuous",
        type=int,
        default=None,
        help="Bin continuous targets into N classes for classification",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading embeddings from: {args.embeddings}")
    print(f"  Embedding type: {args.embedding_type}")
    embeddings, embed_meta = load_embeddings(args.embeddings, args.embedding_type)
    
    # Handle multi-sample embeddings
    if embeddings.ndim == 3:
        print(f"  Shape: {embeddings.shape} (using first sample)")
        embeddings = embeddings[:, 0, :]
    else:
        print(f"  Shape: {embeddings.shape}")
    
    print(f"\nLoading properties from: {args.properties}")
    smiles_list, properties, validity = load_properties(
        args.properties, columns=args.targets, include_validity=True
    )
    print(f"  Loaded {len(smiles_list)} molecules, {len(properties)} properties")
    
    # Verify alignment
    if len(smiles_list) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: {len(smiles_list)} SMILES vs {embeddings.shape[0]} embeddings"
        )
    
    # Filter targets based on options
    targets = args.targets
    if args.classification_only:
        targets = [t for t in (targets or properties.keys()) 
                   if t in DEFAULT_CLASSIFICATION_TARGETS]
    elif args.regression_only:
        targets = [t for t in (targets or properties.keys()) 
                   if t in DEFAULT_REGRESSION_TARGETS]
    
    # Run probing
    results = run_probing(
        embeddings=embeddings,
        properties=properties,
        targets=targets,
        cv_folds=args.cv_folds,
        bin_continuous=args.bin_continuous,
        random_state=args.random_state,
    )
    
    # Save and print results
    if results:
        save_results(results, args.output_dir)
        print_summary(results)
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
