#!/usr/bin/env python
"""
Print model size and parameter counts for KERMT models.

Supports all three pretraining modes:
- vocab: Original vocabulary-based pretraining (KermtTask)
  - Uses atom_vocab, bond_vocab, and fg_size (default: 85)
- cmim: CMIM pretraining with SMILES decoder (KermtCMIMTask)
  - Uses smiles_vocab
- hybrid: Combined CMIM + vocab pretraining (KermtHybridTask)
  - Uses smiles_vocab, atom_vocab, bond_vocab, and fg_size
  - Combines CMIM (contrastive + reconstruction) with vocab prediction

Uses model configuration matching the default pretraining setup:
- Defaults from parsing.py
- Overridden by launch-KERMT-pretrain-slurm.sh values

Usage (from grover_fork directory):
    # CMIM mode (default):
    python task/print_model_size.py --mode cmim
    
    # Vocab mode (uses default fg_size=85):
    python task/print_model_size.py --mode vocab
    
    # Hybrid mode (CMIM + vocab combined):
    python task/print_model_size.py --mode hybrid
    
    # With custom parameters:
    python task/print_model_size.py --mode vocab --hidden_size 1000 --depth 8 --fg_size 100
"""

import argparse
import pickle
from kermt.model.models import KermtCMIMTask, KermtTask, KermtHybridTask, KERMTEmbedding


def load_vocab(vocab_path):
    """Load vocabulary from pickle file and return its size."""
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return len(vocab)




def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_parameters_by_component_cmim(model):
    """Count parameters by component for CMIM model."""
    component_params = {}
    
    # Latent distribution (encoder + readout + projection heads)
    if hasattr(model, 'latent_dist'):
        component_params['Latent Distribution (Encoder + Readout + Heads)'] = \
            sum(p.numel() for p in model.latent_dist.parameters())
        
        # Break down latent_dist components
        if hasattr(model.latent_dist, 'kermt'):
            component_params['  ├─ Encoder (GROVER)'] = \
                sum(p.numel() for p in model.latent_dist.kermt.parameters())
        if hasattr(model.latent_dist, 'readout'):
            component_params['  ├─ Readout'] = \
                sum(p.numel() for p in model.latent_dist.readout.parameters())
        if hasattr(model.latent_dist, 'fc_mean_logscale'):
            component_params['  └─ Mean/LogScale Projection'] = \
                sum(p.numel() for p in model.latent_dist.fc_mean_logscale.parameters())
    
    # Decoder
    if hasattr(model, 'decoder'):
        component_params['Decoder (SMILES Transformer)'] = \
            sum(p.numel() for p in model.decoder.parameters())
    
    return component_params


def count_parameters_by_component_vocab(model):
    """Count parameters by component for vocab-based model."""
    component_params = {}
    
    # Encoder
    if hasattr(model, 'kermt'):
        component_params['Encoder (GROVER)'] = \
            sum(p.numel() for p in model.kermt.parameters())
    
    # Atom vocab prediction tasks
    if hasattr(model, 'av_task_atom'):
        component_params['Atom Vocab Prediction (from atom)'] = \
            sum(p.numel() for p in model.av_task_atom.parameters())
    if hasattr(model, 'av_task_bond'):
        component_params['Atom Vocab Prediction (from bond)'] = \
            sum(p.numel() for p in model.av_task_bond.parameters())
    
    # Bond vocab prediction tasks
    if hasattr(model, 'bv_task_atom'):
        component_params['Bond Vocab Prediction (from atom)'] = \
            sum(p.numel() for p in model.bv_task_atom.parameters())
    if hasattr(model, 'bv_task_bond'):
        component_params['Bond Vocab Prediction (from bond)'] = \
            sum(p.numel() for p in model.bv_task_bond.parameters())
    
    # Functional group prediction
    if hasattr(model, 'fg_task_all'):
        component_params['Functional Group Prediction'] = \
            sum(p.numel() for p in model.fg_task_all.parameters())
    
    return component_params


def count_parameters_by_component_hybrid(model):
    """Count parameters by component for hybrid model (CMIM + vocab)."""
    component_params = {}
    
    # Latent distribution (encoder + readout + projection heads)
    if hasattr(model, 'latent_dist'):
        component_params['Latent Distribution (Encoder + Readout + Heads)'] = \
            sum(p.numel() for p in model.latent_dist.parameters())
        
        # Break down latent_dist components
        if hasattr(model.latent_dist, 'kermt'):
            component_params['  ├─ Encoder (GROVER)'] = \
                sum(p.numel() for p in model.latent_dist.kermt.parameters())
        if hasattr(model.latent_dist, 'readout'):
            component_params['  ├─ Readout'] = \
                sum(p.numel() for p in model.latent_dist.readout.parameters())
        if hasattr(model.latent_dist, 'fc_mean_logscale'):
            component_params['  └─ Mean/LogScale Projection'] = \
                sum(p.numel() for p in model.latent_dist.fc_mean_logscale.parameters())
    
    # Decoder
    if hasattr(model, 'decoder'):
        component_params['Decoder (SMILES Transformer)'] = \
            sum(p.numel() for p in model.decoder.parameters())
    
    # Vocab prediction module
    if hasattr(model, 'vocab_module'):
        component_params['Vocab Prediction Module (Total)'] = \
            sum(p.numel() for p in model.vocab_module.parameters())
        
        # Break down vocab module components
        if hasattr(model.vocab_module, 'av_task_atom'):
            component_params['  ├─ Atom Vocab (from atom)'] = \
                sum(p.numel() for p in model.vocab_module.av_task_atom.parameters())
        if hasattr(model.vocab_module, 'av_task_bond'):
            component_params['  ├─ Atom Vocab (from bond)'] = \
                sum(p.numel() for p in model.vocab_module.av_task_bond.parameters())
        if hasattr(model.vocab_module, 'bv_task_atom'):
            component_params['  ├─ Bond Vocab (from atom)'] = \
                sum(p.numel() for p in model.vocab_module.bv_task_atom.parameters())
        if hasattr(model.vocab_module, 'bv_task_bond'):
            component_params['  ├─ Bond Vocab (from bond)'] = \
                sum(p.numel() for p in model.vocab_module.bv_task_bond.parameters())
        if hasattr(model.vocab_module, 'fg_task_all'):
            component_params['  └─ Functional Group'] = \
                sum(p.numel() for p in model.vocab_module.fg_task_all.parameters())
    
    return component_params


def format_size(num_params):
    """Format parameter count in human-readable form."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def create_cmim_model(args, smiles_vocab_size):
    """Create GROVER+CMIM model from configuration."""
    print("Creating CMIM model from configuration...")
    
    # Set non-configurable required args
    args.atom_vocab_size = 100
    args.bond_vocab_size = 100
    args.backbone = 'gtrans'
    args.embedding_output_type = 'both'
    args.dense = False
    args.bias = False
    args.undirected = False
    args.cuda = False
    args.features_dim = 0
    args.no_cache = True
    args.smiles_vocab_path = None
    args.tensorboard = False
    
    # Ensure gating args exist (default to False if not provided)
    if not hasattr(args, 'decoder_gate_self_attn'):
        args.decoder_gate_self_attn = False
    if not hasattr(args, 'decoder_gate_cross_attn'):
        args.decoder_gate_cross_attn = False
    
    model = KermtCMIMTask(args, smiles_vocab_size=smiles_vocab_size)
    return model, args


def create_vocab_model(args, atom_vocab_size, bond_vocab_size, fg_size):
    """Create original vocab-based KERMT model from configuration."""
    print("Creating vocab-based KERMT model from configuration...")
    
    # Set non-configurable required args
    args.backbone = 'gtrans'
    args.embedding_output_type = 'both'
    args.dense = False
    args.bias = False
    args.undirected = False
    args.cuda = False
    args.features_dim = 0
    args.no_cache = True
    
    # Create encoder
    kermt = KERMTEmbedding(args)
    
    # Create model
    model = KermtTask(args, kermt, atom_vocab_size, bond_vocab_size, fg_size)
    return model, args


def create_hybrid_model(args, smiles_vocab_size, atom_vocab_size, bond_vocab_size, fg_size):
    """Create hybrid KERMT model (CMIM + vocab) from configuration."""
    print("Creating hybrid KERMT model from configuration...")
    
    # Set non-configurable required args
    args.atom_vocab_size = atom_vocab_size
    args.bond_vocab_size = bond_vocab_size
    args.backbone = 'gtrans'
    args.embedding_output_type = 'both'
    args.dense = False
    args.bias = False
    args.undirected = False
    args.cuda = False
    args.features_dim = 0
    args.no_cache = True
    args.smiles_vocab_path = None
    args.tensorboard = False
    
    # Ensure gating args exist (default to False if not provided)
    if not hasattr(args, 'decoder_gate_self_attn'):
        args.decoder_gate_self_attn = False
    if not hasattr(args, 'decoder_gate_cross_attn'):
        args.decoder_gate_cross_attn = False
    
    # Create encoder
    kermt = KERMTEmbedding(args)
    
    # Create model
    model = KermtHybridTask(
        args,
        kermt=kermt,
        latent_dim=args.latent_dim,
        contrastive_temperature=args.contrastive_temperature,
        smiles_vocab_size=smiles_vocab_size,
        atom_vocab_size=atom_vocab_size,
        bond_vocab_size=bond_vocab_size,
        fg_size=fg_size
    )
    return model, args


def print_model_info_cmim(model, args, smiles_vocab_size):
    """Print comprehensive model information for CMIM model."""
    print("\n" + "="*70)
    print("GROVER+CMIM MODEL SIZE")
    print("="*70)
    
    # Configuration
    print("\nModel Configuration:")
    print("  Encoder (GROVER):")
    print(f"    Hidden Size:        {args.hidden_size}")
    print(f"    Depth (Layers):     {args.depth}")
    print(f"    Attention Heads:    {args.num_attn_head}")
    print(f"    MT Blocks:          {args.num_mt_block}")
    print(f"    Activation:         {args.activation}")
    print(f"    Dropout:            {args.dropout}")
    print(f"    Readout:            {'Self-Attention' if args.self_attention else 'Mean'}")
    if args.self_attention:
        print(f"      Attn Hidden:      {args.attn_hidden}")
        print(f"      Attn Out:         {args.attn_out}")
    
    print("\n  Decoder (CMIM):")
    print(f"    SMILES Vocab Size:  {smiles_vocab_size}")
    print(f"    Latent Dim:         {args.latent_dim}")
    print(f"    Decoder Layers:     {args.decoder_num_layers}")
    print(f"    Decoder Heads:      {args.decoder_num_attention_heads}")
    print(f"    Decoder FFN Dim:    {args.decoder_ffn_hidden_size}")
    print(f"    Decoder Dropout:    {args.decoder_dropout}")
    print(f"    Positional Encoding: {args.decoder_positional_encoding}")
    
    # Total parameters
    total_params, trainable_params = count_parameters(model)
    print("\nParameter Count:")
    print(f"  Total Parameters:      {total_params:,} ({format_size(total_params)})")
    print(f"  Trainable Parameters:  {trainable_params:,} ({format_size(trainable_params)})")
    
    # Component breakdown
    print("\nParameter Breakdown by Component:")
    component_params = count_parameters_by_component_cmim(model)
    for component, count in component_params.items():
        percentage = (count / total_params) * 100
        print(f"  {component:50s}: {count:12,} ({format_size(count):>8s}) [{percentage:5.1f}%]")
    
    # Model size in memory
    param_size_mb = (total_params * 4) / (1024 ** 2)
    print("\nApproximate Model Size:")
    print(f"  FP32: {param_size_mb:.2f} MB")
    print(f"  FP16: {param_size_mb / 2:.2f} MB")
    
    print("="*70 + "\n")


def print_model_info_vocab(model, args, atom_vocab_size, bond_vocab_size, fg_size):
    """Print comprehensive model information for vocab-based model."""
    print("\n" + "="*70)
    print("KERMT (VOCAB-BASED) MODEL SIZE")
    print("="*70)
    
    # Configuration
    print("\nModel Configuration:")
    print("  Encoder (GROVER):")
    print(f"    Hidden Size:        {args.hidden_size}")
    print(f"    Depth (Layers):     {args.depth}")
    print(f"    Attention Heads:    {args.num_attn_head}")
    print(f"    MT Blocks:          {args.num_mt_block}")
    print(f"    Activation:         {args.activation}")
    print(f"    Dropout:            {args.dropout}")
    
    print("\n  Prediction Tasks:")
    print(f"    Atom Vocab Size:    {atom_vocab_size}")
    print(f"    Bond Vocab Size:    {bond_vocab_size}")
    print(f"    FG Size:            {fg_size}")
    
    # Total parameters
    total_params, trainable_params = count_parameters(model)
    print("\nParameter Count:")
    print(f"  Total Parameters:      {total_params:,} ({format_size(total_params)})")
    print(f"  Trainable Parameters:  {trainable_params:,} ({format_size(trainable_params)})")
    
    # Component breakdown
    print("\nParameter Breakdown by Component:")
    component_params = count_parameters_by_component_vocab(model)
    for component, count in component_params.items():
        percentage = (count / total_params) * 100
        print(f"  {component:35s}: {count:12,} ({format_size(count):>8s}) [{percentage:5.1f}%]")
    
    # Model size in memory
    param_size_mb = (total_params * 4) / (1024 ** 2)
    print("\nApproximate Model Size:")
    print(f"  FP32: {param_size_mb:.2f} MB")
    print(f"  FP16: {param_size_mb / 2:.2f} MB")
    
    print("="*70 + "\n")


def print_model_info_hybrid(model, args, smiles_vocab_size, atom_vocab_size, bond_vocab_size, fg_size):
    """Print comprehensive model information for hybrid model."""
    print("\n" + "="*70)
    print("KERMT HYBRID (CMIM + VOCAB) MODEL SIZE")
    print("="*70)
    
    # Configuration
    print("\nModel Configuration:")
    print("  Encoder (GROVER):")
    print(f"    Hidden Size:        {args.hidden_size}")
    print(f"    Depth (Layers):     {args.depth}")
    print(f"    Attention Heads:    {args.num_attn_head}")
    print(f"    MT Blocks:          {args.num_mt_block}")
    print(f"    Activation:         {args.activation}")
    print(f"    Dropout:            {args.dropout}")
    print(f"    Readout:            {'Self-Attention' if args.self_attention else 'Mean'}")
    if args.self_attention:
        print(f"      Attn Hidden:      {args.attn_hidden}")
        print(f"      Attn Out:         {args.attn_out}")
    
    print("\n  CMIM Components:")
    print(f"    SMILES Vocab Size:  {smiles_vocab_size}")
    print(f"    Latent Dim:         {args.latent_dim}")
    print(f"    Decoder Layers:     {args.decoder_num_layers}")
    print(f"    Decoder Heads:      {args.decoder_num_attention_heads}")
    print(f"    Decoder FFN Dim:    {args.decoder_ffn_hidden_size}")
    print(f"    Decoder Dropout:    {args.decoder_dropout}")
    print(f"    Positional Encoding: {args.decoder_positional_encoding}")
    
    print("\n  Vocab Components:")
    print(f"    Atom Vocab Size:    {atom_vocab_size}")
    print(f"    Bond Vocab Size:    {bond_vocab_size}")
    print(f"    FG Size:            {fg_size}")
    print(f"    Vocab Loss Weight:  {getattr(args, 'vocab_loss_weight', 1.0)}")
    
    # Total parameters
    total_params, trainable_params = count_parameters(model)
    print("\nParameter Count:")
    print(f"  Total Parameters:      {total_params:,} ({format_size(total_params)})")
    print(f"  Trainable Parameters:  {trainable_params:,} ({format_size(trainable_params)})")
    
    # Component breakdown
    print("\nParameter Breakdown by Component:")
    component_params = count_parameters_by_component_hybrid(model)
    for component, count in component_params.items():
        percentage = (count / total_params) * 100
        print(f"  {component:50s}: {count:12,} ({format_size(count):>8s}) [{percentage:5.1f}%]")
    
    # Model size in memory
    param_size_mb = (total_params * 4) / (1024 ** 2)
    print("\nApproximate Model Size:")
    print(f"  FP32: {param_size_mb:.2f} MB")
    print(f"  FP16: {param_size_mb / 2:.2f} MB")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Print KERMT model size (supports vocab, cmim, and hybrid modes)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='cmim',
                        choices=['vocab', 'cmim', 'hybrid'],
                        help='Pretraining mode: vocab (original), cmim (with decoder), '
                             'or hybrid (cmim + vocab combined)')
    
    # Vocabulary paths (relative to grover_fork)
    parser.add_argument('--atom_vocab_path', type=str,
                        default='tests/data/pretrain_atom_vocab.pkl',
                        help='Path to atom vocabulary (vocab/hybrid modes)')
    parser.add_argument('--bond_vocab_path', type=str,
                        default='tests/data/pretrain_bond_vocab.pkl',
                        help='Path to bond vocabulary (vocab/hybrid modes)')
    parser.add_argument('--fg_size', type=int, default=85,
                        help='Functional group size (vocab/hybrid modes, default: 85)')
    parser.add_argument('--smiles_vocab_path', type=str,
                        default='tests/data/pretrain_smiles_vocab.pkl',
                        help='Path to SMILES vocabulary (cmim/hybrid modes)')
    
    # Encoder configuration (defaults match launch-KERMT-pretrain-slurm.sh)
    parser.add_argument('--hidden_size', type=int, default=800,
                        help='Encoder hidden size')
    parser.add_argument('--depth', type=int, default=6,
                        help='Number of encoder message passing layers')
    parser.add_argument('--num_attn_head', type=int, default=4,
                        help='Number of attention heads in encoder MTBlock')
    parser.add_argument('--num_mt_block', type=int, default=1,
                        help='Number of MTBlocks in encoder')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='PReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    
    # Readout configuration
    parser.add_argument('--self_attention', action='store_true', default=False,
                        help='Use self-attention readout (default: mean readout)')
    parser.add_argument('--attn_hidden', type=int, default=4,
                        help='Self-attention hidden size (only if --self_attention)')
    parser.add_argument('--attn_out', type=int, default=128,
                        help='Self-attention output size (only if --self_attention)')
    
    # CMIM configuration (for cmim and hybrid modes)
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Latent space dimension (cmim/hybrid only)')
    parser.add_argument('--contrastive_temperature', type=float, default=0.1,
                        help='Temperature for contrastive loss (cmim/hybrid only)')
    parser.add_argument('--reconstruction_loss_weight', type=float, default=1.0,
                        help='Weight for reconstruction loss (cmim/hybrid only)')
    parser.add_argument('--normalize_gradient', action='store_true', default=False,
                        help='Normalize gradients by latent dim (cmim/hybrid only)')
    parser.add_argument('--normalize_loss', action='store_true', default=False,
                        help='Normalize loss by latent dim (cmim/hybrid only)')
    
    # Decoder configuration (for cmim and hybrid modes)
    parser.add_argument('--decoder_num_layers', type=int, default=3,
                        help='Number of decoder layers (cmim/hybrid only)')
    parser.add_argument('--decoder_num_attention_heads', type=int, default=8,
                        help='Number of decoder attention heads (cmim/hybrid only)')
    parser.add_argument('--decoder_ffn_hidden_size', type=int, default=2048,
                        help='Decoder feedforward size (cmim/hybrid only)')
    parser.add_argument('--decoder_dropout', type=float, default=0.1,
                        help='Decoder dropout (cmim/hybrid only)')
    parser.add_argument('--decoder_max_seq_len', type=int, default=512,
                        help='Max SMILES sequence length (cmim/hybrid only)')
    parser.add_argument('--decoder_positional_encoding', type=str, default='rope',
                        choices=['rope', 'sinusoidal'],
                        help='Decoder positional encoding type (cmim/hybrid only)')
    parser.add_argument('--decoder_gate_self_attn', action='store_true', default=False,
                        help='Use gating on decoder self-attention (cmim/hybrid only)')
    parser.add_argument('--decoder_gate_cross_attn', action='store_true', default=False,
                        help='Use gating on decoder cross-attention (cmim/hybrid only)')
    
    # Hybrid-specific configuration
    parser.add_argument('--vocab_loss_weight', type=float, default=1.0,
                        help='Weight for vocab loss in hybrid mode (default: 1.0)')
    
    args = parser.parse_args()
    
    # Create model based on mode
    if args.mode == 'cmim':
        # CMIM mode
        print(f"Loading SMILES vocabulary from: {args.smiles_vocab_path}")
        smiles_vocab_size = load_vocab(args.smiles_vocab_path)
        print(f"SMILES vocabulary size: {smiles_vocab_size}")
        
        model, model_args = create_cmim_model(args, smiles_vocab_size)
        print_model_info_cmim(model, model_args, smiles_vocab_size)
        
    elif args.mode == 'vocab':
        # Vocab mode
        print(f"Loading atom vocabulary from: {args.atom_vocab_path}")
        atom_vocab_size = load_vocab(args.atom_vocab_path)
        print(f"Atom vocabulary size: {atom_vocab_size}")
        
        print(f"Loading bond vocabulary from: {args.bond_vocab_path}")
        bond_vocab_size = load_vocab(args.bond_vocab_path)
        print(f"Bond vocabulary size: {bond_vocab_size}")
        
        print(f"Using functional group size: {args.fg_size}")
        
        model, model_args = create_vocab_model(args, atom_vocab_size, bond_vocab_size, args.fg_size)
        print_model_info_vocab(model, model_args, atom_vocab_size, bond_vocab_size, args.fg_size)
    
    elif args.mode == 'hybrid':
        # Hybrid mode (CMIM + vocab)
        print(f"Loading SMILES vocabulary from: {args.smiles_vocab_path}")
        smiles_vocab_size = load_vocab(args.smiles_vocab_path)
        print(f"SMILES vocabulary size: {smiles_vocab_size}")
        
        print(f"Loading atom vocabulary from: {args.atom_vocab_path}")
        atom_vocab_size = load_vocab(args.atom_vocab_path)
        print(f"Atom vocabulary size: {atom_vocab_size}")
        
        print(f"Loading bond vocabulary from: {args.bond_vocab_path}")
        bond_vocab_size = load_vocab(args.bond_vocab_path)
        print(f"Bond vocabulary size: {bond_vocab_size}")
        
        print(f"Using functional group size: {args.fg_size}")
        
        model, model_args = create_hybrid_model(args, smiles_vocab_size, atom_vocab_size, bond_vocab_size, args.fg_size)
        print_model_info_hybrid(model, model_args, smiles_vocab_size, atom_vocab_size, bond_vocab_size, args.fg_size)


if __name__ == '__main__':
    main()
