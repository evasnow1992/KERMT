#!/usr/bin/env python
"""
Visualize latent representations using dimensionality reduction (t-SNE, UMAP, PCA).

This script creates 2D visualizations of molecular embeddings to understand
the structure of the learned latent space. Points can be colored by molecular
properties to see how different properties cluster in the latent space.

================================================================================
Visualization Methods:
================================================================================
- t-SNE: Good for preserving local structure, reveals clusters
- UMAP: Faster than t-SNE, preserves both local and global structure
- PCA: Linear reduction, good baseline, fastest

================================================================================
Usage:
================================================================================
    # Basic t-SNE visualization
    python task/latent_visualization.py \
        --embeddings embeddings.npy \
        --output_dir viz_results/

    # Color by molecular properties
    python task/latent_visualization.py \
        --embeddings embeddings.npy \
        --properties properties.csv \
        --output_dir viz_results/ \
        --color_by LogP NumRings Lipinski

    # Use UMAP (faster for large datasets)
    python task/latent_visualization.py \
        --embeddings embeddings.npy \
        --output_dir viz_results/ \
        --method umap

    # Interactive HTML output
    python task/latent_visualization.py \
        --embeddings embeddings.npy \
        --properties properties.csv \
        --output_dir viz_results/ \
        --interactive

    # Subsample for large datasets
    python task/latent_visualization.py \
        --embeddings embeddings.npy \
        --output_dir viz_results/ \
        --max_samples 10000

    # PCA preprocessing before t-SNE (faster, often better results)
    python task/latent_visualization.py \
        --embeddings embeddings.npy \
        --output_dir viz_results/ \
        --method tsne \
        --pca_preprocess \
        --pca_preprocess_n_components 20

    # PCA preprocessing with UMAP and property coloring
    python task/latent_visualization.py \
        --embeddings embeddings.npy \
        --properties properties.csv \
        --output_dir viz_results/ \
        --method umap \
        --pca_preprocess \
        --pca_preprocess_n_components 50 \
        --color_by LogP NumRings
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from embedding_utils import EMBEDDING_TYPES, load_embeddings, load_properties

# Optional imports
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "tsne",
    n_components: int = 2,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    verbose: bool = True,
    pca_n_variance_components: int = 10,
    pca_preprocess: bool = False,
    pca_preprocess_n_components: int = 20,
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Reduce embedding dimensions for visualization.

    Args:
        embeddings: High-dimensional embeddings [N, D]
        method: "tsne", "umap", or "pca"
        n_components: Output dimensions (usually 2)
        perplexity: t-SNE perplexity
        n_neighbors: UMAP n_neighbors
        min_dist: UMAP min_dist
        random_state: Random seed
        verbose: Print progress
        pca_n_variance_components: Number of top PCA components to report variance for
        pca_preprocess: If True, apply PCA first before t-SNE/UMAP (ignored if method="pca")
        pca_preprocess_n_components: Number of PCA components to keep for preprocessing

    Returns:
        Tuple of:
            - Reduced embeddings [N, n_components]
            - PCA variance info dict (for PCA method or when pca_preprocess=True)
              Contains: explained_variance_ratio, cumulative_variance, n_components_for_95
    """
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    pca_variance_info = None
    
    # Apply PCA preprocessing if requested (for t-SNE/UMAP only)
    if pca_preprocess and method in ["tsne", "umap"]:
        n_pca_comps = min(pca_preprocess_n_components, embeddings_scaled.shape[1], embeddings_scaled.shape[0])
        if verbose:
            print(f"Applying PCA preprocessing: {embeddings_scaled.shape[1]}D -> {n_pca_comps}D")
        
        pca_reducer = PCA(n_components=n_pca_comps, random_state=random_state)
        embeddings_scaled = pca_reducer.fit_transform(embeddings_scaled)
        
        # Capture PCA variance info
        explained_variance = pca_reducer.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Find n_components needed for 95% variance
        n_for_95 = np.searchsorted(cumulative_variance, 0.95) + 1
        n_for_95 = min(n_for_95, len(cumulative_variance))
        
        pca_variance_info = {
            "preprocessing": True,
            "n_components_computed": n_pca_comps,
            "explained_variance_ratio": explained_variance.tolist(),
            "cumulative_variance": cumulative_variance.tolist(),
            "n_components_for_95_percent": int(n_for_95),
            "total_variance_explained": float(cumulative_variance[-1]),
        }
        
        if verbose:
            print(f"  PCA preprocessing variance breakdown (top {n_pca_comps} components):")
            for i, (ev, cv) in enumerate(zip(explained_variance, cumulative_variance)):
                print(f"    PC{i+1}: {ev:.4f} (cumulative: {cv:.4f})")
            print(f"  Total variance captured by {n_pca_comps} components: {cumulative_variance[-1]*100:.1f}%")
            print(f"  Components needed for 95% variance: {n_for_95}")
    
    if method == "tsne":
        if verbose:
            print(f"Running t-SNE (perplexity={perplexity})...")
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
            max_iter=1000,  # renamed from n_iter in scikit-learn 1.5+
        )
        reduced = reducer.fit_transform(embeddings_scaled)
    
    elif method == "umap":
        if not HAS_UMAP:
            raise ImportError("UMAP not installed. Install with: pip install umap-learn")
        if verbose:
            print(f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        reduced = reducer.fit_transform(embeddings_scaled)
    
    elif method == "pca":
        if verbose:
            print(f"Running PCA (n_components={n_components})...")
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(embeddings_scaled)
        
        # Compute full PCA to get top N variance information
        n_variance_comps = min(pca_n_variance_components, embeddings_scaled.shape[1])
        full_pca = PCA(n_components=n_variance_comps, random_state=random_state)
        full_pca.fit(embeddings_scaled)
        
        explained_variance = full_pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Find n_components needed for 95% variance
        n_for_95 = np.searchsorted(cumulative_variance, 0.95) + 1
        n_for_95 = min(n_for_95, len(cumulative_variance))
        
        pca_variance_info = {
            "n_components_computed": n_variance_comps,
            "explained_variance_ratio": explained_variance.tolist(),
            "cumulative_variance": cumulative_variance.tolist(),
            "n_components_for_95_percent": int(n_for_95),
            "total_variance_explained_by_2d": float(cumulative_variance[1]) if len(cumulative_variance) >= 2 else float(cumulative_variance[0]),
        }
        
        if verbose:
            print(f"  Explained variance (top {n_variance_comps} components):")
            for i, (ev, cv) in enumerate(zip(explained_variance, cumulative_variance)):
                print(f"    PC{i+1}: {ev:.4f} (cumulative: {cv:.4f})")
            print(f"  Components needed for 95% variance: {n_for_95}")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return reduced, pca_variance_info


def create_static_plot(
    reduced: np.ndarray,
    color_values: Optional[np.ndarray] = None,
    color_name: Optional[str] = None,
    title: str = "Latent Space Visualization",
    output_path: str = "latent_viz.png",
    figsize: Tuple[int, int] = (10, 8),
    point_size: float = 1.0,
    alpha: float = 0.6,
    cmap: str = "viridis",
    discrete_cmap: str = "viridis",
    discrete_threshold: int = 20,
):
    """
    Create a static matplotlib visualization.

    Args:
        reduced: 2D coordinates [N, 2]
        color_values: Optional values for coloring points
        color_name: Name of the color variable
        title: Plot title
        output_path: Output file path
        figsize: Figure size
        point_size: Point size
        alpha: Point transparency
        cmap: Colormap for continuous values
        discrete_cmap: Colormap for discrete/ordinal values (gradient-based)
        discrete_threshold: Max unique values to treat as discrete
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_values is not None:
        valid_mask = ~np.isnan(color_values)
        valid_colors = color_values[valid_mask]
        n_unique = len(np.unique(valid_colors))
        
        if n_unique <= discrete_threshold:
            # Discrete coloring with gradient colormap for ordinal values
            # Sort numerically (not as strings) to preserve ordinal relationships
            unique_vals = np.unique(valid_colors)
            # Sort as numbers: convert to int if they're whole numbers, otherwise float
            try:
                if np.all(unique_vals == unique_vals.astype(int)):
                    unique_vals = sorted(unique_vals.astype(int))
                else:
                    unique_vals = sorted(unique_vals)
            except (ValueError, TypeError):
                unique_vals = sorted(unique_vals)
            
            # Use a sequential/gradient colormap instead of categorical
            # This helps visualize ordinal relationships (e.g., NumHAcceptors 1-16)
            gradient_cmap = plt.colormaps.get_cmap(discrete_cmap)
            
            for i, val in enumerate(unique_vals):
                # Map index to colormap position (0 to 1)
                color_pos = i / max(1, len(unique_vals) - 1)
                mask = (color_values == val) & valid_mask
                ax.scatter(
                    reduced[mask, 0], reduced[mask, 1],
                    c=[gradient_cmap(color_pos)],
                    s=point_size,
                    alpha=alpha,
                    label=f"{int(val)}" if isinstance(val, (int, float)) and val == int(val) else str(val),
                )
            
            # Plot invalid points in gray
            if (~valid_mask).any():
                ax.scatter(
                    reduced[~valid_mask, 0], reduced[~valid_mask, 1],
                    c="lightgray",
                    s=point_size * 0.5,
                    alpha=alpha * 0.5,
                    label="N/A",
                )
            
            # Use smaller legend if many categories
            legend_kwargs = {"title": color_name, "markerscale": 3}
            if n_unique > 10:
                legend_kwargs["fontsize"] = 8
                legend_kwargs["ncol"] = 2 if n_unique > 15 else 1
                legend_kwargs["loc"] = "upper right"
            else:
                legend_kwargs["loc"] = "upper right"
            ax.legend(**legend_kwargs)
        
        else:
            # Continuous coloring
            scatter = ax.scatter(
                reduced[valid_mask, 0], reduced[valid_mask, 1],
                c=valid_colors,
                s=point_size,
                alpha=alpha,
                cmap=cmap,
            )
            
            # Plot invalid points in gray
            if (~valid_mask).any():
                ax.scatter(
                    reduced[~valid_mask, 0], reduced[~valid_mask, 1],
                    c="lightgray",
                    s=point_size * 0.5,
                    alpha=alpha * 0.5,
                )
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_name)
    
    else:
        ax.scatter(
            reduced[:, 0], reduced[:, 1],
            s=point_size,
            alpha=alpha,
            c="steelblue",
        )
    
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved plot to {output_path}")


def create_interactive_plot(
    reduced: np.ndarray,
    smiles_list: Optional[List[str]] = None,
    properties: Optional[Dict[str, np.ndarray]] = None,
    color_by: Optional[str] = None,
    title: str = "Latent Space Visualization",
    output_path: str = "latent_viz.html",
    point_size: float = 3.0,
):
    """
    Create an interactive Plotly visualization.

    Args:
        reduced: 2D coordinates [N, 2]
        smiles_list: SMILES strings for hover info
        properties: Property dict for coloring/hover
        color_by: Property to color by
        title: Plot title
        output_path: Output HTML file path
        point_size: Point size
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly not installed. Install with: pip install plotly")
    
    # Build dataframe-like dict
    data = {
        "x": reduced[:, 0],
        "y": reduced[:, 1],
    }
    
    if smiles_list is not None:
        data["smiles"] = smiles_list
    
    if properties is not None:
        for prop, values in properties.items():
            data[prop] = values
    
    # Create hover text
    hover_cols = ["smiles"] if smiles_list else []
    if properties:
        hover_cols.extend(list(properties.keys())[:5])  # Limit to 5 properties
    
    # Create figure
    if color_by and color_by in data:
        color_vals = data[color_by]
        valid_mask = ~np.isnan(color_vals) if isinstance(color_vals[0], float) else [True] * len(color_vals)
        
        n_unique = len(np.unique([v for v, m in zip(color_vals, valid_mask) if m]))
        
        if n_unique <= 20:
            # Discrete coloring
            fig = px.scatter(
                x=data["x"], y=data["y"],
                color=[str(int(v)) if not np.isnan(v) else "N/A" for v in color_vals],
                hover_data={col: data[col] for col in hover_cols if col in data},
                title=title,
                labels={"x": "Dimension 1", "y": "Dimension 2", "color": color_by},
            )
        else:
            # Continuous coloring
            fig = px.scatter(
                x=data["x"], y=data["y"],
                color=color_vals,
                hover_data={col: data[col] for col in hover_cols if col in data},
                title=title,
                labels={"x": "Dimension 1", "y": "Dimension 2", "color": color_by},
                color_continuous_scale="Viridis",
            )
    else:
        fig = px.scatter(
            x=data["x"], y=data["y"],
            hover_data={col: data[col] for col in hover_cols if col in data},
            title=title,
            labels={"x": "Dimension 1", "y": "Dimension 2"},
        )
    
    fig.update_traces(marker=dict(size=point_size))
    fig.update_layout(
        width=900,
        height=700,
        template="plotly_white",
    )
    
    fig.write_html(output_path)
    print(f"Saved interactive plot to {output_path}")


def compute_cluster_metrics(
    reduced: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute clustering quality metrics.

    Args:
        reduced: 2D coordinates
        labels: Cluster labels (discrete property values)

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    # Remove NaN labels
    valid_mask = ~np.isnan(labels)
    if valid_mask.sum() < 10:
        return {}
    
    X = reduced[valid_mask]
    y = labels[valid_mask].astype(int)
    
    n_unique = len(np.unique(y))
    if n_unique < 2:
        return {}
    
    metrics = {}
    
    try:
        metrics["silhouette"] = silhouette_score(X, y)
    except:
        pass
    
    try:
        metrics["davies_bouldin"] = davies_bouldin_score(X, y)
    except:
        pass
    
    try:
        metrics["calinski_harabasz"] = calinski_harabasz_score(X, y)
    except:
        pass
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Visualize latent representations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input
    parser.add_argument(
        "--embeddings", "-e",
        type=str,
        required=True,
        help="Path to embeddings directory or file",
    )
    parser.add_argument(
        "--properties", "-p",
        type=str,
        default=None,
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
        help="Output directory",
    )
    
    # Visualization options
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["tsne", "umap", "pca"],
        default="tsne",
        help="Dimensionality reduction method",
    )
    parser.add_argument(
        "--color_by", "-c",
        nargs="+",
        default=None,
        help="Properties to color by (creates separate plots)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Create interactive HTML plots (requires plotly)",
    )
    parser.add_argument(
        "--discrete_cmap",
        type=str,
        default="viridis",
        help="Colormap for discrete/ordinal values (e.g., viridis, plasma, rainbow, coolwarm)",
    )
    
    # Method-specific parameters
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors",
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist",
    )
    
    # Other options
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to visualize (random subsample)",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=1.0,
        help="Point size for static plots",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--pca_n_variance_components",
        type=int,
        default=10,
        help="Number of top PCA components to report variance for (only used with --method pca)",
    )
    parser.add_argument(
        "--pca_preprocess",
        action="store_true",
        help="Apply PCA preprocessing before t-SNE/UMAP. This reduces dimensions first "
             "(default: 20 components), then applies t-SNE/UMAP. Often produces better "
             "results and is faster for high-dimensional embeddings.",
    )
    parser.add_argument(
        "--pca_preprocess_n_components",
        type=int,
        default=20,
        help="Number of PCA components to keep when using --pca_preprocess (default: 20)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    print(f"Loading embeddings from: {args.embeddings}")
    print(f"  Embedding type: {args.embedding_type}")
    embeddings, embed_meta = load_embeddings(args.embeddings, args.embedding_type)
    
    if embeddings.ndim == 3:
        print(f"  Shape: {embeddings.shape} (using first sample)")
        embeddings = embeddings[:, 0, :]
    else:
        print(f"  Shape: {embeddings.shape}")
    
    # Load properties if provided
    smiles_list = None
    properties = None
    
    if args.properties:
        print(f"Loading properties from: {args.properties}")
        smiles_list, properties = load_properties(args.properties, args.color_by)
        print(f"  Loaded {len(smiles_list)} molecules")
        
        if len(smiles_list) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(smiles_list)} SMILES vs {embeddings.shape[0]} embeddings"
            )
    elif embed_meta and "smiles" in embed_meta:
        smiles_list = embed_meta["smiles"]
    
    # Subsample if needed
    n_samples = embeddings.shape[0]
    sample_indices = np.arange(n_samples)
    
    if args.max_samples and n_samples > args.max_samples:
        print(f"Subsampling from {n_samples} to {args.max_samples} samples")
        np.random.seed(args.random_state)
        sample_indices = np.random.choice(n_samples, args.max_samples, replace=False)
        embeddings = embeddings[sample_indices]
        
        if smiles_list:
            smiles_list = [smiles_list[i] for i in sample_indices]
        if properties:
            properties = {k: v[sample_indices] for k, v in properties.items()}
    
    # Reduce dimensions
    reduced, pca_variance_info = reduce_dimensions(
        embeddings=embeddings,
        method=args.method,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        pca_n_variance_components=args.pca_n_variance_components,
        pca_preprocess=args.pca_preprocess,
        pca_preprocess_n_components=args.pca_preprocess_n_components,
    )
    
    # Determine output prefix based on method and preprocessing
    if args.pca_preprocess and args.method in ["tsne", "umap"]:
        method_prefix = f"pca{args.pca_preprocess_n_components}_{args.method}"
    else:
        method_prefix = args.method
    
    # Save reduced coordinates
    np.save(output_dir / f"{method_prefix}_coords.npy", reduced)
    print(f"Saved coordinates to {output_dir / f'{method_prefix}_coords.npy'}")
    
    # Save PCA variance information if using PCA or PCA preprocessing
    if pca_variance_info is not None:
        pca_variance_path = output_dir / f"{method_prefix}_pca_variance_info.json"
        with open(pca_variance_path, "w") as f:
            json.dump(pca_variance_info, f, indent=2)
        print(f"Saved PCA variance info to {pca_variance_path}")
    
    # Create visualizations
    color_properties = args.color_by or []
    
    # Base plot (no coloring)
    if args.pca_preprocess and args.method in ["tsne", "umap"]:
        base_title = f"Latent Space (PCA-{args.pca_preprocess_n_components} + {args.method.upper()})"
    else:
        base_title = f"Latent Space ({args.method.upper()})"
    create_static_plot(
        reduced=reduced,
        title=base_title,
        output_path=str(output_dir / f"{method_prefix}_base.png"),
        point_size=args.point_size,
        discrete_cmap=args.discrete_cmap,
    )
    
    if args.interactive:
        create_interactive_plot(
            reduced=reduced,
            smiles_list=smiles_list,
            properties=properties,
            title=base_title,
            output_path=str(output_dir / f"{method_prefix}_base.html"),
        )
    
    # Colored plots
    cluster_metrics = {}
    
    # Method label for titles
    if args.pca_preprocess and args.method in ["tsne", "umap"]:
        method_label = f"PCA-{args.pca_preprocess_n_components} + {args.method.upper()}"
    else:
        method_label = args.method.upper()
    
    for prop in color_properties:
        if properties and prop in properties:
            color_values = properties[prop]
            
            # Static plot
            create_static_plot(
                reduced=reduced,
                color_values=color_values,
                color_name=prop,
                title=f"Latent Space by {prop} ({method_label})",
                output_path=str(output_dir / f"{method_prefix}_{prop}.png"),
                point_size=args.point_size,
                discrete_cmap=args.discrete_cmap,
            )
            
            # Interactive plot
            if args.interactive:
                create_interactive_plot(
                    reduced=reduced,
                    smiles_list=smiles_list,
                    properties=properties,
                    color_by=prop,
                    title=f"Latent Space by {prop} ({method_label})",
                    output_path=str(output_dir / f"{method_prefix}_{prop}.html"),
                )
            
            # Compute clustering metrics for discrete properties
            n_unique = len(np.unique(color_values[~np.isnan(color_values)]))
            if n_unique <= 20:
                metrics = compute_cluster_metrics(reduced, color_values)
                if metrics:
                    cluster_metrics[prop] = metrics
                    print(f"\nClustering metrics for {prop}:")
                    for name, val in metrics.items():
                        print(f"  {name}: {val:.4f}")
    
    # Save clustering metrics
    if cluster_metrics:
        with open(output_dir / "cluster_metrics.json", "w") as f:
            json.dump(cluster_metrics, f, indent=2)
        print(f"\nSaved clustering metrics to {output_dir / 'cluster_metrics.json'}")
    
    print(f"\n{'=' * 60}")
    print(f"Visualization complete! Results saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
