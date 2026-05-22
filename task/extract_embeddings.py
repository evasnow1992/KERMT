#!/usr/bin/env python
"""
Extract molecular embeddings from pretrained GROVER/KERMT model checkpoints.

This script loads any pretrained checkpoint (vocab, CMIM, or hybrid mode) and extracts
molecular embeddings from the shared KERMT encoder. It works with all model variants
since they all share the same encoder architecture.

================================================================================
Embedding Types:
================================================================================
The KERMT encoder produces four types of embeddings (dual-view architecture):
- atom_from_atom: Atom embeddings from the atom-centric message passing
- atom_from_bond: Atom embeddings from the bond-centric message passing
- bond_from_atom: Bond embeddings from the atom-centric message passing
- bond_from_bond: Bond embeddings from the bond-centric message passing

These are aggregated to molecule-level using mean pooling (no learnable parameters).

================================================================================
Output Formats:
================================================================================
- NPY: NumPy array files (one per embedding type, efficient for large datasets)
- PKL: Single pickle file with all embeddings and metadata

================================================================================
Usage:
================================================================================
    # Extract embeddings from any checkpoint (auto-detects model type)
    python task/extract_embeddings.py \
        -c model.pt \
        -i molecules.csv \
        -o embeddings_dir/

    # Output as pickle (single file with all embeddings)
    python task/extract_embeddings.py \
        -c model.pt \
        -i molecules.csv \
        -o embeddings.pkl \
        --format pkl

    # With batch processing for large datasets
    python task/extract_embeddings.py \
        -c model.pt \
        -i molecules.csv \
        -o embeddings_dir/ \
        --batch_size 128
"""

import argparse
import csv
import pickle
import time
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)

from kermt.data.molgraph import mol2graph
from kermt.model.models import KERMTEmbedding
from kermt.model.layers import Readout
from collections import OrderedDict


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES using RDKit. Returns None if invalid."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def load_encoder_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda"
) -> Tuple[Any, Readout, argparse.Namespace]:
    """
    Load KERMT encoder from any checkpoint (vocab, CMIM, hybrid, or grover_base).
    
    This is a minimal loader that only builds KERMTEmbedding (encoder),
    not the full finetune model. This avoids needing all finetune-specific args.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Tuple of (encoder, readout, args)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = state["args"]
    loaded_state_dict = state["state_dict"]

    print(f"Checkpoint from epoch {state.get('epoch', 'unknown')}, "
          f"step {state.get('scheduler_step', 'unknown')}")
    
    # === Transform parameter names for compatibility (same as utils.py) ===
    # 1. Replace old "grover" naming with "kermt"
    loaded_state_dict = OrderedDict([
        (k.replace("grover", "kermt"), v) for k, v in loaded_state_dict.items()
    ])
    
    # 2. Handle CMIM checkpoint format: strip "latent_dist." prefix
    has_cmim_params = any("latent_dist." in k for k in loaded_state_dict.keys())
    if has_cmim_params:
        print("Detected CMIM checkpoint format. Transforming 'latent_dist.kermt.*' -> 'kermt.*'")
    loaded_state_dict = OrderedDict([
        (k.replace("latent_dist.", ""), v) for k, v in loaded_state_dict.items()
    ])
    
    # === Set default args for KERMTEmbedding (if missing from old checkpoints) ===
    # These are the args required by KERMTEmbedding.__init__ and mol2graph
    encoder_defaults = {
        # Core model architecture
        'hidden_size': 800,
        'depth': 6,
        'dropout': 0.0,
        'activation': 'PReLU',
        'bias': False,
        'undirected': False,
        'dense': False,
        'atom_message': False,
        
        # Transformer / attention args
        'backbone': 'gtrans',
        'num_mt_block': 1,
        'num_attn_head': 4,
        'embedding_output_type': 'both',  # Need all 4 embeddings
        
        # Other
        'cuda': device == "cuda",
        'nencoders': 3,
        'coord': 15,
        'input_layer': 'fc',
        'no_attach_fea': False,
        'self_attention': False,
        'attn_hidden': 4,
        'attn_out': 8,
        
        # Featurization (used by mol2graph)
        'use_cuikmolmaker_featurization': False,  # Not used in launch scripts
        'bond_drop_rate': 0.0,
        'no_cache': True,
    }
    
    for key, value in encoder_defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
            print(f"  Set default: {key} = {value}")
    
    # Force embedding_output_type to 'both' for full embedding extraction
    args.embedding_output_type = 'both'
    args.cuda = device == "cuda"
    
    # === Build encoder ===
    encoder = KERMTEmbedding(args)
    
    # === Extract encoder weights from checkpoint ===
    # Encoder is stored under "kermt." prefix after transformations
    encoder_state_dict = {}
    for key, value in loaded_state_dict.items():
        if key.startswith("kermt."):
            encoder_key = key[len("kermt."):]
            encoder_state_dict[encoder_key] = value
    
    if not encoder_state_dict:
        raise ValueError(
            f"Could not find encoder weights (kermt.*) in checkpoint. "
            f"Available keys: {list(loaded_state_dict.keys())[:10]}..."
        )
    
    print(f"Found {len(encoder_state_dict)} encoder parameters")
    
    # === Load weights with flexible matching (skip mismatched) ===
    model_state_dict = encoder.state_dict()
    pretrained_state_dict = {}
    skipped = []
    
    for param_name, param_value in encoder_state_dict.items():
        if param_name not in model_state_dict:
            skipped.append(f"'{param_name}' (not in model)")
        elif model_state_dict[param_name].shape != param_value.shape:
            skipped.append(f"'{param_name}' (shape mismatch)")
        else:
            pretrained_state_dict[param_name] = param_value
    
    if skipped:
        print(f"Skipped {len(skipped)} incompatible parameters: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    
    model_state_dict.update(pretrained_state_dict)
    encoder.load_state_dict(model_state_dict)
    print(f"Loaded {len(pretrained_state_dict)}/{len(encoder_state_dict)} encoder parameters")
    
    if device == "cuda":
        encoder = encoder.cuda()
    encoder.eval()
    
    # === Create readout layer (mean pooling, stateless) ===
    hidden_size = args.hidden_size
    readout = Readout(rtype="mean", hidden_size=hidden_size)
    if device == "cuda":
        readout = readout.cuda()

    print(f"Encoder loaded successfully. Hidden size: {hidden_size}")

    return encoder, readout, args


def smiles_to_graph_batch(smiles_list: List[str], args: argparse.Namespace) -> Tuple:
    """
    Convert SMILES strings to graph batch format for the encoder.

    Args:
        smiles_list: List of SMILES strings
        args: Model arguments

    Returns:
        Graph batch tuple
    """
    shared_dict = {}
    batch_graph = mol2graph(smiles_list, shared_dict, args)
    return batch_graph.get_components()


def extract_embeddings_batch(
    encoder: torch.nn.Module,
    readout: Readout,
    smiles_batch: List[str],
    args: argparse.Namespace,
    device: str = "cuda",
) -> Tuple[Dict[str, np.ndarray], List[bool]]:
    """
    Extract all four molecular embeddings for a batch of SMILES.

    Args:
        encoder: The KERMT encoder
        readout: Readout layer (mean pooling, stateless)
        smiles_batch: List of SMILES strings
        args: Model arguments
        device: Device for computation

    Returns:
        Tuple of (embeddings dict with 4 arrays, validity mask)
    """
    # Filter valid SMILES
    valid_smiles = []
    valid_indices = []
    
    for i, smi in enumerate(smiles_batch):
        can_smi = canonicalize_smiles(smi)
        if can_smi is not None:
            valid_smiles.append(can_smi)
            valid_indices.append(i)
    
    batch_size = len(smiles_batch)
    hidden_size = args.hidden_size
    
    # Initialize output arrays for all four embedding types
    embedding_types = ["atom_from_atom", "atom_from_bond", "bond_from_atom", "bond_from_bond"]
    embeddings = {
        etype: np.zeros((batch_size, hidden_size), dtype=np.float32)
        for etype in embedding_types
    }
    validity = [False] * batch_size
    
    if not valid_smiles:
        return embeddings, validity
    
    # Convert to graph batch
    graph_batch = smiles_to_graph_batch(valid_smiles, args)
    
    # Extract scope information for readout
    # graph_batch = (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)
    f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = graph_batch
    
    # Move tensors to device
    graph_batch_device = tuple(
        t.to(device) if isinstance(t, torch.Tensor) else t for t in graph_batch
    )
    
    with torch.no_grad():
        # Get embeddings from encoder
        encoder_output = encoder(graph_batch_device)
        
        # Extract molecule-level embeddings using mean readout
        # atom_from_atom and atom_from_bond use a_scope
        # bond_from_atom and bond_from_bond use b_scope
        # Note: same readout instance works for all since mean pooling is stateless
        
        mol_embeddings = {}
        
        if encoder_output["atom_from_atom"] is not None:
            mol_embeddings["atom_from_atom"] = readout(
                encoder_output["atom_from_atom"], a_scope
            )
        
        if encoder_output["atom_from_bond"] is not None:
            mol_embeddings["atom_from_bond"] = readout(
                encoder_output["atom_from_bond"], a_scope
            )
        
        if encoder_output["bond_from_atom"] is not None:
            mol_embeddings["bond_from_atom"] = readout(
                encoder_output["bond_from_atom"], b_scope
            )
        
        if encoder_output["bond_from_bond"] is not None:
            mol_embeddings["bond_from_bond"] = readout(
                encoder_output["bond_from_bond"], b_scope
            )
        
        # Place embeddings at correct indices
        for batch_idx, orig_idx in enumerate(valid_indices):
            validity[orig_idx] = True
            for etype in embedding_types:
                if etype in mol_embeddings and mol_embeddings[etype] is not None:
                    embeddings[etype][orig_idx] = mol_embeddings[etype][batch_idx].cpu().numpy()
    
    return embeddings, validity


def extract_all_embeddings(
    encoder: torch.nn.Module,
    readout: Readout,
    smiles_list: List[str],
    args: argparse.Namespace,
    batch_size: int = 64,
    device: str = "cuda",
    show_progress: bool = True,
) -> Tuple[Dict[str, np.ndarray], List[str], List[bool]]:
    """
    Extract all four embeddings for all SMILES in a list.

    Args:
        encoder: The KERMT encoder
        readout: Readout layer (mean pooling, stateless)
        smiles_list: List of SMILES strings
        args: Model arguments
        batch_size: Batch size for processing
        device: Device for computation
        show_progress: Show progress bar

    Returns:
        Tuple of (embeddings dict [N, hidden_size] x 4, 
                  canonical SMILES list, validity list)
    """
    n_molecules = len(smiles_list)
    hidden_size = args.hidden_size
    
    embedding_types = ["atom_from_atom", "atom_from_bond", "bond_from_atom", "bond_from_bond"]
    all_embeddings = {
        etype: np.zeros((n_molecules, hidden_size), dtype=np.float32)
        for etype in embedding_types
    }
    all_validity = [False] * n_molecules
    all_canonical = [None] * n_molecules
    
    # Pre-canonicalize to store canonical SMILES
    for i, smi in enumerate(smiles_list):
        all_canonical[i] = canonicalize_smiles(smi)
    
    # Process in batches
    n_batches = (n_molecules + batch_size - 1) // batch_size
    iterator = range(0, n_molecules, batch_size)
    
    if show_progress:
        iterator = tqdm(iterator, total=n_batches, desc="Extracting embeddings")
    
    for start_idx in iterator:
        end_idx = min(start_idx + batch_size, n_molecules)
        batch_smiles = smiles_list[start_idx:end_idx]
        
        batch_embeddings, batch_validity = extract_embeddings_batch(
            encoder=encoder,
            readout=readout,
            smiles_batch=batch_smiles,
            args=args,
            device=device,
        )
        
        for etype in embedding_types:
            all_embeddings[etype][start_idx:end_idx] = batch_embeddings[etype]
        all_validity[start_idx:end_idx] = batch_validity
    
    return all_embeddings, all_canonical, all_validity


def save_embeddings(
    output_path: str,
    embeddings: Dict[str, np.ndarray],
    smiles_list: List[str],
    canonical_list: List[str],
    validity_list: List[bool],
    format: str = "npy",
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save embeddings to file(s).

    Args:
        output_path: Output path (directory for npy, file for pkl)
        embeddings: Dict of embedding arrays {type: [N, hidden_size]}
        smiles_list: Original SMILES list
        canonical_list: Canonical SMILES list
        validity_list: Validity list
        format: Output format ('npy' or 'pkl')
        metadata: Optional metadata dictionary
    """
    n_molecules = len(smiles_list)
    hidden_size = list(embeddings.values())[0].shape[1]
    embedding_types = list(embeddings.keys())
    
    if format == "npy":
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each embedding type as separate .npy file
        for etype, emb_array in embeddings.items():
            np.save(output_dir / f"{etype}.npy", emb_array)
            print(f"Saved {etype} embeddings to {output_dir / f'{etype}.npy'}")
        
        # Save metadata
        meta = {
            "smiles": smiles_list,
            "canonical_smiles": canonical_list,
            "valid": validity_list,
            "n_molecules": n_molecules,
            "hidden_size": hidden_size,
            "embedding_types": embedding_types,
            **(metadata or {}),
        }
        meta_path = output_dir / "metadata.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        print(f"Saved metadata to {meta_path}")
    
    elif format == "pkl":
        # Save everything in a single pickle file
        data = {
            "embeddings": embeddings,
            "smiles": smiles_list,
            "canonical_smiles": canonical_list,
            "valid": validity_list,
            "n_molecules": n_molecules,
            "hidden_size": hidden_size,
            "embedding_types": embedding_types,
            **(metadata or {}),
        }
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved all embeddings to {output_path}")
    
    else:
        raise ValueError(f"Unknown format: {format}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract molecular embeddings from pretrained GROVER/KERMT checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint (works with vocab, CMIM, or hybrid)",
    )

    # Input
    parser.add_argument(
        "--input_file", "-i",
        type=str,
        required=True,
        help="Input CSV file with SMILES (first column)",
    )
    parser.add_argument(
        "--smiles_column",
        type=int,
        default=0,
        help="Column index containing SMILES (0-indexed)",
    )

    # Output
    parser.add_argument(
        "--output_path", "-o",
        type=str,
        required=True,
        help="Output path (directory for npy format, file for pkl format)",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["npy", "pkl"],
        default="npy",
        help="Output format: 'npy' saves 4 separate files, 'pkl' saves single file",
    )

    # Processing options
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=64,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for computation",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load encoder from checkpoint
    encoder, readout, model_args = load_encoder_from_checkpoint(
        args.checkpoint, device=args.device
    )

    # Read input SMILES
    smiles_list = []
    with open(args.input_file, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if row and len(row) > args.smiles_column:
                smiles_list.append(row[args.smiles_column].strip())

    print(f"Loaded {len(smiles_list)} SMILES from {args.input_file}")

    # Extract embeddings
    start_time = time.time()
    
    embeddings, canonical_list, validity_list = extract_all_embeddings(
        encoder=encoder,
        readout=readout,
        smiles_list=smiles_list,
        args=model_args,
        batch_size=args.batch_size,
        device=args.device,
        show_progress=not args.no_progress,
    )

    elapsed = time.time() - start_time

    # Print summary
    n_valid = sum(validity_list)
    n_invalid = len(smiles_list) - n_valid
    print(f"\nExtraction complete in {elapsed:.1f}s")
    print(f"  Valid molecules:   {n_valid} ({100 * n_valid / len(smiles_list):.1f}%)")
    print(f"  Invalid molecules: {n_invalid} ({100 * n_invalid / len(smiles_list):.1f}%)")
    print("  Embedding shapes:")
    for etype, emb in embeddings.items():
        print(f"    {etype}: {emb.shape}")

    # Save
    metadata = {
        "checkpoint": args.checkpoint,
        "embedding_output_type": model_args.embedding_output_type,
    }
    
    save_embeddings(
        output_path=args.output_path,
        embeddings=embeddings,
        smiles_list=smiles_list,
        canonical_list=canonical_list,
        validity_list=validity_list,
        format=args.format,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
