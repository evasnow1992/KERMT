"""
The dataset used in training KERMT.
"""
import math
import os
import csv
from typing import Union, List
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from rdkit import Chem

import kermt.util.utils as feautils
from kermt.data import mol2graph
from kermt.data.moldataset import MoleculeDatapoint
from kermt.data.task_labels import atom_to_vocab, bond_to_vocab
from kermt.util.features import get_feature_range

import cuik_molmaker


def setup_cuik_molmaker_features(args):
    """
    Helper function to set up cuik-molmaker feature tensors and ranges.
    Shared by all collators.
    
    Args:
        args: Training arguments with use_cuikmolmaker_featurization flag
        
    Returns:
        tuple: (cmm_feature_tensors dict, cmm_feature_range) or (None, None)
    """
    if not args.use_cuikmolmaker_featurization:
        return None, None
    
    # Define feature properties
    atom_onehot_props = [
        "atomic-number", "total-degree", "formal-charge", "chirality",
        "num-hydrogens", "hybridization", "implicit-valence", "ring-size"
    ]
    atom_float_props = [
        "aromatic", "mass", "hydrogen-bond-acceptor",
        "hydrogen-bond-donor", "acidic", "basic"
    ]
    bond_props = ["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"]
    
    # Form feature tensors
    cmm_feature_tensors = {
        "atom_onehot": cuik_molmaker.atom_onehot_feature_names_to_tensor(atom_onehot_props),
        "atom_float": cuik_molmaker.atom_float_feature_names_to_tensor(atom_float_props),
        "bond": cuik_molmaker.bond_feature_names_to_tensor(bond_props)
    }
    
    # Get feature ranges
    cmm_feature_range = get_feature_range(atom_onehot_props, atom_float_props)
    
    return cmm_feature_tensors, cmm_feature_range


# ============================================================================
# Shared helper functions for collators
# ============================================================================

def atom_random_mask(smiles_batch, atom_vocab, percent=0.15):
    """
    Perform random mask operation on atoms for vocabulary prediction.
    Shared by KermtCollator and KermtHybridCollator.
    
    Args:
        smiles_batch: List of SMILES strings
        atom_vocab: MolVocab instance for atom vocabulary
        percent: Fraction of atoms to mask (default 0.15)
        
    Returns:
        List of atom vocabulary labels (0 = not masked)
    """
    vocab_label = [0]  # Zero padding at start
    for smi in smiles_batch:
        mol = Chem.MolFromSmiles(smi)
        mlabel = [0] * mol.GetNumAtoms()
        n_mask = math.ceil(mol.GetNumAtoms() * percent)
        perm = np.random.permutation(mol.GetNumAtoms())[:n_mask]
        for p in perm:
            atom = mol.GetAtomWithIdx(int(p))
            mlabel[p] = atom_vocab.stoi.get(atom_to_vocab(mol, atom), atom_vocab.other_index)
        vocab_label.extend(mlabel)
    return vocab_label


def bond_random_mask(smiles_batch, bond_vocab, percent=0.15):
    """
    Perform random mask operation on bonds for vocabulary prediction.
    Shared by KermtCollator and KermtHybridCollator.
    
    Args:
        smiles_batch: List of SMILES strings
        bond_vocab: MolVocab instance for bond vocabulary
        percent: Fraction of bonds to mask (default 0.15)
        
    Returns:
        List of bond vocabulary labels (0 = not masked)
    """
    vocab_label = [0]  # Zero padding at start
    for smi in smiles_batch:
        mol = Chem.MolFromSmiles(smi)
        nm_atoms = mol.GetNumAtoms()
        nm_bonds = mol.GetNumBonds()
        mlabel = []
        n_mask = math.ceil(nm_bonds * percent)
        perm = np.random.permutation(nm_bonds)[:n_mask]
        virtual_bond_id = 0
        for a1 in range(nm_atoms):
            for a2 in range(a1 + 1, nm_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is None:
                    continue
                if virtual_bond_id in perm:
                    label = bond_vocab.stoi.get(bond_to_vocab(mol, bond), bond_vocab.other_index)
                    mlabel.extend([label])
                else:
                    mlabel.extend([0])
                virtual_bond_id += 1
        vocab_label.extend(mlabel)
    return vocab_label


def tokenize_and_pad_smiles(smiles_batch, smiles_vocab, max_seq_len):
    """
    Tokenize SMILES strings, truncate if needed, and pad to same length.
    Shared by KermtDecoderCollator and KermtHybridCollator.
    
    Args:
        smiles_batch: List of SMILES strings
        smiles_vocab: SMILESVocab instance for tokenization
        max_seq_len: Maximum sequence length
        
    Returns:
        tokens: [batch_size, max_len] padded token IDs with <start> and <end>
        lengths: [batch_size] sequence lengths after truncation
        padding_mask: [batch_size, max_len] True where padded
    """
    batch_size = len(smiles_batch)
    
    # Tokenize all SMILES (includes <start> and <end>)
    tokenized = [
        smiles_vocab.smiles_to_ids(smi, add_special_tokens=True)
        for smi in smiles_batch
    ]
    
    # Truncate sequences that exceed max_seq_len
    end_token_id = smiles_vocab.end_index
    for i, ids in enumerate(tokenized):
        if len(ids) > max_seq_len:
            tokenized[i] = ids[:max_seq_len-1] + [end_token_id]
    
    # Get lengths (after truncation)
    lengths = torch.tensor([len(ids) for ids in tokenized], dtype=torch.long)
    
    # Pad to same length
    max_len = max(lengths)
    pad_idx = smiles_vocab.pad_index
    
    tokens = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    for i, ids in enumerate(tokenized):
        tokens[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    
    # Create padding mask (True where padding)
    padding_mask = tokens == pad_idx
    
    return tokens, lengths, padding_mask


def prepare_decoder_sequences(tokens):
    """
    Prepare decoder input and target sequences for teacher forcing.
    Shared by KermtDecoderCollator and KermtHybridCollator.
    
    Input:  [<start>, C, C, (, =, O, ), <end>, <pad>, <pad>]
    Decoder input:  [<start>, C, C, (, =, O, ), <end>, <pad>]  (all except last)
    Decoder target: [C, C, (, =, O, ), <end>, <pad>, <pad>]     (all except first)
    
    Args:
        tokens: [batch, seq_len] full sequences with <start> and <end>
        
    Returns:
        decoder_input: [batch, seq_len-1] input to decoder
        decoder_target: [batch, seq_len-1] target for loss
    """
    decoder_input = tokens[:, :-1].contiguous()
    decoder_target = tokens[:, 1:].contiguous()
    return decoder_input, decoder_target


def create_causal_mask(seq_len, positional_encoding='rope', device='cpu'):
    """
    Create causal mask for autoregressive generation.
    Shared by KermtDecoderCollator and KermtHybridCollator.
    
    Args:
        seq_len: Sequence length
        positional_encoding: Type of positional encoding ('rope' or 'sinusoidal')
        device: Device to create mask on
        
    Returns:
        mask: [seq_len, seq_len] causal mask in appropriate format
    """
    if positional_encoding == 'sinusoidal':
        # PyTorch's nn.TransformerDecoder expects BoolTensor
        # True = masked (cannot attend), False = allowed
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    elif positional_encoding == 'rope':
        # Custom RoPE implementation expects FloatTensor with additive masking
        # -inf = masked (becomes 0 after softmax), 0 = allowed
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
    else:
        raise ValueError(f"Unknown positional_encoding: {positional_encoding}")
    
    return mask


def get_data(data_path, logger=None, load_features=True, max_cached_files=100):
    """
    Load data from the data_path.
    :param data_path: the data_path.
    :param logger: the logger.
    :param load_features: whether to load feature files (.npz). Set to False for CMIM
                          pretraining which doesn't use features.
    :param max_cached_files: Maximum number of files to keep in memory (LRU cache).
                             Set to 0 or None to disable cache limit.
    :return:
    """
    debug = logger.debug if logger is not None else print
    summary_path = os.path.join(data_path, "summary.txt")
    smiles_path = os.path.join(data_path, "graph")
    feature_path = os.path.join(data_path, "feature")

    fin = open(summary_path)
    n_files = int(fin.readline().strip().split(":")[-1])
    n_samples = int(fin.readline().strip().split(":")[-1])
    sample_per_file = int(fin.readline().strip().split(":")[-1])
    debug("Loading data:")
    debug("Number of files: %d" % n_files)
    debug("Number of samples: %d" % n_samples)
    debug("Samples/file: %d" % sample_per_file)
    debug("Load features: %s" % load_features)
    debug("Max cached files: %s" % max_cached_files)

    datapoints = []
    for i in range(n_files):
        smiles_path_i = os.path.join(smiles_path, str(i) + ".csv")
        feature_path_i = os.path.join(feature_path, str(i) + ".npz")
        n_samples_i = sample_per_file if i != (n_files - 1) else n_samples % sample_per_file
        datapoints.append(BatchDatapoint(smiles_path_i, feature_path_i, n_samples_i, load_features))
    return BatchMolDataset(datapoints, max_cached_files=max_cached_files), sample_per_file


def split_data(data,
               split_type='random',
               sizes=(0.8, 0.1, 0.1),
               seed=0,
               logger=None):
    """
    Split data with given train/validation/test ratio.
    :param data:
    :param split_type:
    :param sizes:
    :param seed:
    :param logger:
    :return:
    """
    assert len(sizes) == 3 and sum(sizes) == 1

    if split_type == "random":
        data.shuffle(seed=seed)
        data = data.data

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = data[:train_size]
        val = data[train_size:train_val_size]
        test = data[train_val_size:]

        return BatchMolDataset(train), BatchMolDataset(val), BatchMolDataset(test)
    else:
        raise NotImplementedError("Do not support %s splits" % split_type)


class BatchDatapoint:
    def __init__(self,
                 smiles_file,
                 feature_file,
                 n_samples,
                 load_features=True,
                 ):
        self.smiles_file = smiles_file
        self.feature_file = feature_file
        # deal with the last batch graph numbers.
        self.n_samples = n_samples
        self.load_features = load_features
        self.datapoints = None

    def load_datapoints(self):
        # Skip loading features for CMIM pretraining (features not used)
        if self.load_features:
            features = self.load_feature()
        else:
            features = [None] * self.n_samples
        self.datapoints = []

        with open(self.smiles_file) as f:
            reader = csv.reader(f)
            next(reader)
            for i, line in enumerate(reader):
                # line = line[0]
                d = MoleculeDatapoint(line=line,
                                      features=features[i])
                self.datapoints.append(d)

        assert len(self.datapoints) == self.n_samples

    def load_feature(self):
        return feautils.load_features(self.feature_file)

    def shuffle(self):
        pass

    def clean_cache(self):
        del self.datapoints
        self.datapoints = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Lazy loading: load data on first access if not already loaded
        if self.datapoints is None:
            self.load_datapoints()
        return self.datapoints[idx]

    def is_loaded(self):
        return self.datapoints is not None


class BatchMolDataset(Dataset):
    def __init__(self, data: List[BatchDatapoint],
                 graph_per_file=None,
                 max_cached_files=100):
        """
        Args:
            data: List of BatchDatapoint objects
            graph_per_file: Number of samples per file
            max_cached_files: Maximum number of files to keep in memory (LRU cache).
                              Set to None or 0 to disable cache limit (keep all).
        """
        self.data = data
        self.max_cached_files = max_cached_files if max_cached_files else None
        self.loaded_file_order = []  # Track order for LRU eviction

        self.len = 0
        for d in self.data:
            self.len += len(d)
        if graph_per_file is not None:
            self.sample_per_file = graph_per_file
        else:
            self.sample_per_file = len(self.data[0]) if len(self.data) != 0 else None

    def shuffle(self, seed: int = None):
        pass

    def clean_cache(self):
        for d in self.data:
            d.clean_cache()
        self.loaded_file_order = []

    def __len__(self) -> int:
        return self.len

    def _ensure_loaded_with_cache(self, dp_idx):
        """Load file at dp_idx, evicting old files if cache is full."""
        if self.data[dp_idx].is_loaded():
            # Already loaded - move to end of LRU list (most recently used)
            if dp_idx in self.loaded_file_order:
                self.loaded_file_order.remove(dp_idx)
            self.loaded_file_order.append(dp_idx)
            return
        
        # Need to load - first check if we need to evict
        if self.max_cached_files and len(self.loaded_file_order) >= self.max_cached_files:
            # Evict least recently used file (first in list)
            evict_idx = self.loaded_file_order.pop(0)
            self.data[evict_idx].clean_cache()
        
        # Load the new file
        self.data[dp_idx].load_datapoints()
        self.loaded_file_order.append(dp_idx)

    def __getitem__(self, idx) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        dp_idx = int(idx / self.sample_per_file)
        real_idx = idx % self.sample_per_file
        
        # Ensure file is loaded (with LRU cache management)
        self._ensure_loaded_with_cache(dp_idx)
        
        return self.data[dp_idx][real_idx], idx

    def load_data(self, idx):
        dp_idx = int(idx / self.sample_per_file)
        self._ensure_loaded_with_cache(dp_idx)

    def count_loaded_datapoints(self):
        res = 0
        for d in self.data:
            if d.is_loaded():
                res += 1
        return res


# ============================================================================
# Pre-tokenized data classes for memory-efficient CMIM training
# ============================================================================

def get_pretokenized_data(data_path, tokens_dir, logger=None, max_smiles_cache_files=50):
    """
    Load pre-tokenized data from memory-mapped .npy files.
    
    This is the memory-efficient alternative to get_data() for CMIM pretraining.
    Uses memory-mapped numpy arrays that can be shared across DataLoader workers.
    
    Args:
        data_path: Base data path (for summary.txt and graph/ SMILES)
        tokens_dir: Path to pre-tokenized .npy files (e.g., ZINC15/train/tokens)
        logger: Optional logger
        max_smiles_cache_files: Max files to keep SMILES cached (LRU eviction).
                                Each file ~50MB. Default 50 = ~2.5GB max per worker.
        
    Returns:
        (PreTokenizedBatchMolDataset, sample_per_file)
    """
    debug = logger.debug if logger is not None else print
    
    # Read summary from tokens directory
    summary_path = os.path.join(tokens_dir, "summary.txt")
    smiles_path = os.path.join(data_path, "graph")
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}\n"
            f"Make sure you've run pretokenize_zinc15.py to completion."
        )
    
    with open(summary_path) as fin:
        n_files = int(fin.readline().strip().split(":")[-1])
        n_samples = int(fin.readline().strip().split(":")[-1])
        sample_per_file = int(fin.readline().strip().split(":")[-1])
    
    debug("Loading pre-tokenized data:")
    debug(f"  Number of files: {n_files}")
    debug(f"  Number of samples: {n_samples}")
    debug(f"  Samples/file: {sample_per_file}")
    debug(f"  Tokens directory: {tokens_dir}")
    debug(f"  Max SMILES cache files: {max_smiles_cache_files}")
    
    datapoints = []
    for i in range(n_files):
        tokens_path_i = os.path.join(tokens_dir, str(i) + ".npy")
        smiles_path_i = os.path.join(smiles_path, str(i) + ".csv")
        n_samples_i = sample_per_file if i != (n_files - 1) else n_samples % sample_per_file
        if n_samples_i == 0:
            n_samples_i = sample_per_file  # Handle exact division case
        datapoints.append(PreTokenizedBatchDatapoint(tokens_path_i, smiles_path_i, n_samples_i))
    
    return PreTokenizedBatchMolDataset(datapoints, max_smiles_cache_files=max_smiles_cache_files), sample_per_file


class PreTokenizedBatchDatapoint:
    """
    A batch datapoint that uses memory-mapped pre-tokenized numpy arrays.
    
    Key difference from BatchDatapoint: uses np.load(mmap_mode='r') so the
    same physical memory is shared across all DataLoader workers via OS page cache.
    """
    
    def __init__(self, tokens_file: str, smiles_file: str, n_samples: int):
        """
        Args:
            tokens_file: Path to pre-tokenized .npy file
            smiles_file: Path to SMILES .csv file (for graph construction)
            n_samples: Number of samples in this file
        """
        self.tokens_file = tokens_file
        self.smiles_file = smiles_file
        self.n_samples = n_samples
        
        # Memory-mapped arrays - shared across all workers
        self._tokens_mmap = None
        self._smiles_cache = None
    
    def _ensure_mmap(self):
        """Lazily create memory-mapped view of tokens."""
        if self._tokens_mmap is None:
            # mmap_mode='r' = read-only memory-mapped
            # This is THE key for multi-worker efficiency:
            # All workers share the same underlying file pages via OS page cache
            self._tokens_mmap = np.load(self.tokens_file, mmap_mode='r')
    
    def _ensure_smiles(self):
        """Lazily load SMILES strings (still needed for graph construction)."""
        if self._smiles_cache is None:
            self._smiles_cache = []
            with open(self.smiles_file) as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for line in reader:
                    self._smiles_cache.append(line[0])
    
    def get_tokens(self, idx: int) -> np.ndarray:
        """Get pre-tokenized tokens for a sample (memory-mapped, efficient)."""
        self._ensure_mmap()
        return self._tokens_mmap[idx]
    
    def get_smiles(self, idx: int) -> str:
        """Get SMILES string for graph construction."""
        self._ensure_smiles()
        return self._smiles_cache[idx]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """Return (tokens, smiles, local_idx) for a sample."""
        return self.get_tokens(idx), self.get_smiles(idx), idx
    
    def is_loaded(self):
        """Memory-mapped files are always 'loaded' (lazy access)."""
        return True
    
    def clean_cache(self):
        """Clear cached SMILES strings (mmap remains efficient)."""
        self._smiles_cache = None


class PreTokenizedBatchMolDataset(Dataset):
    """
    Dataset for pre-tokenized data with memory-mapped numpy arrays.
    
    Designed for multi-worker DataLoader efficiency:
    - Token arrays are memory-mapped and shared via OS page cache
    - SMILES strings use LRU cache to limit memory (only keep recent files)
    """
    
    def __init__(self, data: List[PreTokenizedBatchDatapoint], graph_per_file=None,
                 max_smiles_cache_files: int = 50):
        """
        Args:
            data: List of PreTokenizedBatchDatapoint
            graph_per_file: Samples per file (inferred if None)
            max_smiles_cache_files: Max number of files to keep SMILES cached.
                                    Each file is ~50MB of SMILES strings.
                                    Default 50 files = ~2.5GB max per worker.
        """
        self.data = data
        self.len = sum(len(d) for d in self.data)
        self.max_smiles_cache_files = max_smiles_cache_files
        self._smiles_cache_order = []  # Track LRU order of cached files
        
        if graph_per_file is not None:
            self.sample_per_file = graph_per_file
        else:
            self.sample_per_file = len(self.data[0]) if len(self.data) != 0 else None
    
    def __len__(self) -> int:
        return self.len
    
    def _ensure_smiles_with_cache(self, dp_idx: int):
        """
        Ensure SMILES are loaded for file at dp_idx, with LRU cache eviction.
        """
        datapoint = self.data[dp_idx]
        
        # If already cached, move to end of LRU order
        if datapoint._smiles_cache is not None:
            if dp_idx in self._smiles_cache_order:
                self._smiles_cache_order.remove(dp_idx)
            self._smiles_cache_order.append(dp_idx)
            return
        
        # Need to load - first check if we need to evict
        if (self.max_smiles_cache_files > 0 and 
            len(self._smiles_cache_order) >= self.max_smiles_cache_files):
            # Evict oldest (least recently used) file
            evict_idx = self._smiles_cache_order.pop(0)
            self.data[evict_idx].clean_cache()
        
        # Load SMILES for this file
        datapoint._ensure_smiles()
        self._smiles_cache_order.append(dp_idx)
    
    def __getitem__(self, idx):
        """
        Returns (tokens, smiles, global_idx) for use by collator.
        
        Note: Unlike BatchMolDataset which returns MoleculeDatapoint,
        this returns raw tokens and SMILES for graph construction.
        """
        dp_idx = int(idx / self.sample_per_file)
        real_idx = idx % self.sample_per_file
        
        # Ensure SMILES are loaded with LRU cache management
        self._ensure_smiles_with_cache(dp_idx)
        
        tokens, smiles, local_idx = self.data[dp_idx][real_idx]
        return tokens, smiles, idx
    
    def clean_cache(self):
        """Clear all SMILES caches (mmap remains efficient)."""
        for d in self.data:
            d.clean_cache()
        self._smiles_cache_order = []
    
    def shuffle(self, seed: int = None):
        """Shuffling handled by DistributedSampler."""
        pass


class KermtPreTokenizedDecoderCollator(object):
    """
    Collator for CMIM training with pre-tokenized data.
    
    Key difference from KermtDecoderCollator:
    - Receives pre-tokenized token arrays instead of SMILES strings
    - Only needs to do graph construction (mol2graph) - tokenization already done
    - Much faster collation for large datasets
    """
    
    def __init__(self, shared_dict, smiles_vocab, args):
        """
        Args:
            shared_dict: Shared dictionary for mol2graph caching
            smiles_vocab: SMILESVocab instance (for pad_index, etc.)
            args: Training arguments
        """
        self.args = args
        self.shared_dict = shared_dict
        self.smiles_vocab = smiles_vocab
        
        # Setup cuik-molmaker features if needed
        self.cmm_feature_tensors, self.cmm_feature_range = setup_cuik_molmaker_features(args)
    
    def __call__(self, batch_data):
        """
        Collate batch for CMIM decoder training with pre-tokenized data.
        
        Args:
            batch_data: List of (tokens, smiles, idx) tuples from dataset
            
        Returns:
            dict with graph_input, decoder_input, decoder_target, etc.
        """
        tokens_batch, smiles_batch, idx = zip(*batch_data)
        
        # Stack pre-tokenized tokens into batch
        # tokens_batch is list of numpy arrays with potentially different lengths
        # Need to pad to max length in batch
        max_len = max(len(t) for t in tokens_batch)
        batch_size = len(tokens_batch)
        pad_idx = self.smiles_vocab.pad_index
        
        tokens = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
        for i, t in enumerate(tokens_batch):
            tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        
        # Build graph input for encoder
        if self.args.use_cuikmolmaker_featurization:
            batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args,
                                  cmm_feature_range=self.cmm_feature_range,
                                  cmm_tensors=self.cmm_feature_tensors).get_components()
        else:
            batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args).get_components()
        
        # Prepare decoder sequences (same as KermtDecoderCollator)
        decoder_input, decoder_target = prepare_decoder_sequences(tokens)
        
        # Create padding mask
        padding_mask = tokens == pad_idx
        decoder_padding_mask = padding_mask[:, 1:].contiguous()
        
        # Create causal mask
        seq_len = decoder_input.size(1)
        positional_encoding = getattr(self.args, 'decoder_positional_encoding', 'rope')
        causal_mask = create_causal_mask(seq_len, positional_encoding)
        
        res = {
            "graph_input": batchgraph,
            "decoder_input": decoder_input,
            "decoder_target": decoder_target,
            "decoder_padding_mask": decoder_padding_mask,
            "causal_mask": causal_mask,
            "smiles": smiles_batch,
            "idx": idx
        }
        return res


class KermtDecoderCollator(object):
    """
    Collator for CMIM training with transformer decoder.
    Prepares graph input for encoder and tokenized SMILES for decoder.
    """
    
    def __init__(self, shared_dict, smiles_vocab, args):
        """
        Args:
            shared_dict: Shared dictionary for mol2graph caching
            smiles_vocab: SMILESVocab instance for tokenization
            args: Training arguments
        """
        self.args = args
        self.shared_dict = shared_dict
        self.smiles_vocab = smiles_vocab
        
        # Setup cuik-molmaker features if needed
        self.cmm_feature_tensors, self.cmm_feature_range = setup_cuik_molmaker_features(args)
    
    def __call__(self, batch_idx):
        """
        Collate batch for CMIM decoder training.
        
        Args:
            batch_idx: List of (data, idx) tuples
            
        Returns:
            dict with:
                - graph_input: Graph components for encoder
                - decoder_input: [batch, seq_len] tokenized SMILES input
                - decoder_target: [batch, seq_len] tokenized SMILES target
                - decoder_padding_mask: [batch, seq_len] padding mask
                - causal_mask: [seq_len, seq_len] causal mask
                - smiles: List of original SMILES strings
                - idx: Batch indices
        """
        batch, idx = zip(*batch_idx)
        smiles_batch = [d.smiles for d in batch]
        
        # Build graph input for encoder
        if self.args.use_cuikmolmaker_featurization:
            batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args, 
                                  cmm_feature_range=self.cmm_feature_range, 
                                  cmm_tensors=self.cmm_feature_tensors).get_components()
        else:
            batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args).get_components()
        
        # Tokenize and pad SMILES using helper function
        tokens, lengths, padding_mask = tokenize_and_pad_smiles(
            smiles_batch, self.smiles_vocab, self.args.decoder_max_seq_len
        )
        
        # Prepare decoder sequences using helper function
        decoder_input, decoder_target = prepare_decoder_sequences(tokens)
        
        # Update padding mask to match decoder sequences (shifted)
        decoder_padding_mask = padding_mask[:, 1:].contiguous()
        
        # Create causal mask using helper function
        seq_len = decoder_input.size(1)
        positional_encoding = getattr(self.args, 'decoder_positional_encoding', 'rope')
        causal_mask = create_causal_mask(seq_len, positional_encoding)
        
        res = {
            "graph_input": batchgraph,
            "decoder_input": decoder_input,
            "decoder_target": decoder_target,
            "decoder_padding_mask": decoder_padding_mask,
            "causal_mask": causal_mask,
            "smiles": smiles_batch,
            "idx": idx
        }
        return res


class KermtCollator(object):
    """
    Collator for vocabulary-based pretraining (original GROVER).
    Prepares graph input and vocab prediction targets.
    """
    
    def __init__(self, shared_dict, atom_vocab, bond_vocab, args):
        self.args = args
        self.shared_dict = shared_dict
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab

        # Setup cuik-molmaker features if needed
        self.cmm_feature_tensors, self.cmm_feature_range = setup_cuik_molmaker_features(args)

    def __call__(self, batch_idx):
        batch, idx = zip(*batch_idx)
        smiles_batch = [d.smiles for d in batch]
        
        # Build graph input
        if self.args.use_cuikmolmaker_featurization:
            batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args,
                                  cmm_feature_range=self.cmm_feature_range, 
                                  cmm_tensors=self.cmm_feature_tensors).get_components()
        else:
            batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args).get_components()

        # Generate vocab labels using helper functions
        atom_vocab_label = torch.Tensor(atom_random_mask(smiles_batch, self.atom_vocab)).long()
        bond_vocab_label = torch.Tensor(bond_random_mask(smiles_batch, self.bond_vocab)).long()
        fgroup_label = torch.Tensor([d.features for d in batch]).float()
        
        res = {
            "graph_input": batchgraph,
            "targets": {
                "av_task": atom_vocab_label,
                "bv_task": bond_vocab_label,
                "fg_task": fgroup_label
            },
            "idx": idx
        }
        return res


class KermtHybridCollator(object):
    """
    Collator for hybrid CMIM + vocabulary pretraining.
    Combines functionality from KermtCollator (vocab targets) and 
    KermtDecoderCollator (SMILES tokenization for decoder).
    """
    
    def __init__(self, shared_dict, smiles_vocab, atom_vocab, bond_vocab, args):
        """
        Args:
            shared_dict: Shared dictionary for mol2graph caching
            smiles_vocab: SMILESVocab instance for tokenization (CMIM decoder)
            atom_vocab: MolVocab for atom vocabulary (GROVER)
            bond_vocab: MolVocab for bond vocabulary (GROVER)
            args: Training arguments
        """
        self.args = args
        self.shared_dict = shared_dict
        self.smiles_vocab = smiles_vocab
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        
        # Setup cuik-molmaker features if needed
        self.cmm_feature_tensors, self.cmm_feature_range = setup_cuik_molmaker_features(args)
    
    def __call__(self, batch_idx):
        """
        Collate batch for hybrid CMIM + vocab training.
        
        Args:
            batch_idx: List of (data, idx) tuples
            
        Returns:
            dict with:
                - graph_input: Graph components for encoder
                - decoder_input: Tokenized SMILES input for decoder
                - decoder_target: Tokenized SMILES target
                - decoder_padding_mask: Padding mask for decoder
                - causal_mask: Causal attention mask
                - targets: Dict with av_task, bv_task, fg_task vocab labels
                - smiles: List of original SMILES strings
                - idx: Batch indices
        """
        batch, idx = zip(*batch_idx)
        smiles_batch = [d.smiles for d in batch]
        
        # Build graph input
        if self.args.use_cuikmolmaker_featurization:
            batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args,
                                  cmm_feature_range=self.cmm_feature_range,
                                  cmm_tensors=self.cmm_feature_tensors).get_components()
        else:
            batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args).get_components()
        
        # === Vocab targets (GROVER) using helper functions ===
        atom_vocab_label = torch.Tensor(atom_random_mask(smiles_batch, self.atom_vocab)).long()
        bond_vocab_label = torch.Tensor(bond_random_mask(smiles_batch, self.bond_vocab)).long()
        fgroup_label = torch.Tensor([d.features for d in batch]).float()
        
        # === Decoder sequences (CMIM) using helper functions ===
        tokens, lengths, padding_mask = tokenize_and_pad_smiles(
            smiles_batch, self.smiles_vocab, self.args.decoder_max_seq_len
        )
        decoder_input, decoder_target = prepare_decoder_sequences(tokens)
        decoder_padding_mask = padding_mask[:, 1:].contiguous()
        seq_len = decoder_input.size(1)
        positional_encoding = getattr(self.args, 'decoder_positional_encoding', 'rope')
        causal_mask = create_causal_mask(seq_len, positional_encoding)
        
        res = {
            "graph_input": batchgraph,
            # Decoder inputs for CMIM
            "decoder_input": decoder_input,
            "decoder_target": decoder_target,
            "decoder_padding_mask": decoder_padding_mask,
            "causal_mask": causal_mask,
            # Vocab targets for GROVER
            "targets": {
                "av_task": atom_vocab_label,
                "bv_task": bond_vocab_label,
                "fg_task": fgroup_label
            },
            "smiles": smiles_batch,
            "idx": idx
        }
        return res
