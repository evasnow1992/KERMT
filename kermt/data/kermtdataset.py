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
from kermt.util.features import FeatureRange, get_feature_range

import cuik_molmaker


def setup_cuik_molmaker_features(args):
    """
    Helper function to set up cuik-molmaker feature tensors and ranges.
    Shared by both KermtCollator and KermtDecoderCollator.
    
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


def get_data(data_path, logger=None):
    """
    Load data from the data_path.
    :param data_path: the data_path.
    :param logger: the logger.
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

    datapoints = []
    for i in range(n_files):
        smiles_path_i = os.path.join(smiles_path, str(i) + ".csv")
        feature_path_i = os.path.join(feature_path, str(i) + ".npz")
        n_samples_i = sample_per_file if i != (n_files - 1) else n_samples % sample_per_file
        datapoints.append(BatchDatapoint(smiles_path_i, feature_path_i, n_samples_i))
    return BatchMolDataset(datapoints), sample_per_file


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
                 ):
        self.smiles_file = smiles_file
        self.feature_file = feature_file
        # deal with the last batch graph numbers.
        self.n_samples = n_samples
        self.datapoints = None

    def load_datapoints(self):
        features = self.load_feature()
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
        assert self.datapoints is not None

        return self.datapoints[idx]

    def is_loaded(self):
        return self.datapoints is not None


class BatchMolDataset(Dataset):
    def __init__(self, data: List[BatchDatapoint],
                 graph_per_file=None):
        self.data = data

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

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        # print(idx)
        dp_idx = int(idx / self.sample_per_file)
        real_idx = idx % self.sample_per_file
        return self.data[dp_idx][real_idx], idx

    def load_data(self, idx):
        dp_idx = int(idx / self.sample_per_file)
        if not self.data[dp_idx].is_loaded():
            self.data[dp_idx].load_datapoints()

    def count_loaded_datapoints(self):
        res = 0
        for d in self.data:
            if d.is_loaded():
                res += 1
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
    
    def tokenize_and_pad_smiles(self, smiles_batch):
        """
        Tokenize SMILES strings, truncate if needed, and pad to same length.
        
        Args:
            smiles_batch: List of SMILES strings
            
        Returns:
            tokens: [batch_size, max_len] padded token IDs with <start> and <end>
            lengths: [batch_size] sequence lengths after truncation (including special tokens)
            padding_mask: [batch_size, max_len] True where padded
        """
        batch_size = len(smiles_batch)
        
        # Get max sequence length from args (default 512 from parsing.py)
        max_seq_len = self.args.decoder_max_seq_len
        
        # Tokenize all SMILES (includes <start> and <end>)
        tokenized = [
            self.smiles_vocab.smiles_to_ids(smi, add_special_tokens=True)
            for smi in smiles_batch
        ]
        
        # Truncate sequences that exceed max_seq_len
        # Keep <start>, truncate middle, ensure <end> is at the end
        end_token_id = self.smiles_vocab.end_index
        for i, ids in enumerate(tokenized):
            if len(ids) > max_seq_len:
                # Keep first tokens and add <end> at position max_seq_len-1
                tokenized[i] = ids[:max_seq_len-1] + [end_token_id]
        
        # Get lengths (after truncation)
        lengths = torch.tensor([len(ids) for ids in tokenized], dtype=torch.long)
        
        # Pad to same length
        max_len = max(lengths)
        pad_idx = self.smiles_vocab.pad_index
        
        tokens = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
        for i, ids in enumerate(tokenized):
            tokens[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        
        # Create padding mask (True where padding)
        padding_mask = tokens == pad_idx
        
        return tokens, lengths, padding_mask
    
    def prepare_decoder_sequences(self, tokens):
        """
        Prepare decoder input and target sequences for teacher forcing.
        
        Input:  [<start>, C, C, (, =, O, ), <end>, <pad>, <pad>]
        Decoder input:  [<start>, C, C, (, =, O, ), <end>, <pad>]  (all except last)
        Decoder target: [C, C, (, =, O, ), <end>, <pad>, <pad>]     (all except first)
        
        Args:
            tokens: [batch, seq_len] full sequences with <start> and <end>
            
        Returns:
            decoder_input: [batch, seq_len-1] input to decoder
            decoder_target: [batch, seq_len-1] target for loss
        """
        # Decoder input: all tokens except last
        decoder_input = tokens[:, :-1].contiguous()
        
        # Decoder target: all tokens except first (<start>)
        decoder_target = tokens[:, 1:].contiguous()
        
        return decoder_input, decoder_target
    
    def create_causal_mask(self, seq_len, device='cpu'):
        """
        Create causal mask for autoregressive generation.
        
        The mask format depends on the decoder's positional encoding type:
        - For 'sinusoidal' (PyTorch nn.TransformerDecoder): 
            BoolTensor where True = masked position (cannot attend)
        - For 'rope' (custom RoPETransformerDecoderLayer):
            FloatTensor where -inf = masked position (used with additive masking)
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            mask: [seq_len, seq_len] causal mask in appropriate format
        """
        # Get positional encoding type from args
        positional_encoding = getattr(self.args, 'decoder_positional_encoding', 'rope')
        
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
        
        # Build graph input for encoder (same as KermtCollator)
        if self.args.use_cuikmolmaker_featurization:
            batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args, 
                                  cmm_feature_range=self.cmm_feature_range, 
                                  cmm_tensors=self.cmm_feature_tensors).get_components()
        else:
            batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args).get_components()
        
        # Tokenize and pad SMILES
        tokens, lengths, padding_mask = self.tokenize_and_pad_smiles(smiles_batch)
        
        # Prepare decoder sequences (teacher forcing)
        decoder_input, decoder_target = self.prepare_decoder_sequences(tokens)
        
        # Update padding mask to match decoder sequences (shifted)
        decoder_padding_mask = padding_mask[:, 1:].contiguous()
        
        # Create causal mask (same for all samples in batch)
        seq_len = decoder_input.size(1)
        causal_mask = self.create_causal_mask(seq_len)
        
        res = {
            "graph_input": batchgraph,
            "decoder_input": decoder_input,           # [batch, seq_len]
            "decoder_target": decoder_target,         # [batch, seq_len]
            "decoder_padding_mask": decoder_padding_mask,  # [batch, seq_len]
            "causal_mask": causal_mask,               # [seq_len, seq_len]
            "smiles": smiles_batch,                   # List of strings
            "idx": idx
        }
        return res


class KermtCollator(object):
    def __init__(self, shared_dict, atom_vocab, bond_vocab, args):
        self.args = args
        self.shared_dict = shared_dict
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab

        # Setup cuik-molmaker features if needed
        self.cmm_feature_tensors, self.cmm_feature_range = setup_cuik_molmaker_features(args)

    def atom_random_mask(self, smiles_batch):
        """
        Perform the random mask operation on atoms.
        :param smiles_batch:
        :return: The corresponding atom labels.
        """
        # There is a zero padding.
        vocab_label = [0]
        percent = 0.15
        for smi in smiles_batch:
            mol = Chem.MolFromSmiles(smi)
            mlabel = [0] * mol.GetNumAtoms()
            n_mask = math.ceil(mol.GetNumAtoms() * percent)
            perm = np.random.permutation(mol.GetNumAtoms())[:n_mask]
            for p in perm:
                atom = mol.GetAtomWithIdx(int(p))
                mlabel[p] = self.atom_vocab.stoi.get(atom_to_vocab(mol, atom), self.atom_vocab.other_index)

            vocab_label.extend(mlabel)
        return vocab_label

    def bond_random_mask(self, smiles_batch):
        """
        Perform the random mask operaiion on bonds.
        :param smiles_batch:
        :return: The corresponding bond labels.
        """
        # There is a zero padding.
        vocab_label = [0]
        percent = 0.15
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
                        label = self.bond_vocab.stoi.get(bond_to_vocab(mol, bond), self.bond_vocab.other_index)
                        mlabel.extend([label])
                    else:
                        mlabel.extend([0])

                    virtual_bond_id += 1
            # todo: might need to consider bond_drop_rate
            # todo: double check reverse bond
            vocab_label.extend(mlabel)
        return vocab_label

    def __call__(self, batch_idx):
        batch, idx = zip(*batch_idx)
        smiles_batch = [d.smiles for d in batch]
        if self.args.use_cuikmolmaker_featurization:
            batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args,
                                  cmm_feature_range=self.cmm_feature_range, 
                                  cmm_tensors=self.cmm_feature_tensors).get_components()
        else:
            batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args).get_components()

        atom_vocab_label = torch.Tensor(self.atom_random_mask(smiles_batch)).long()
        bond_vocab_label = torch.Tensor(self.bond_random_mask(smiles_batch)).long()
        fgroup_label = torch.Tensor([d.features for d in batch]).float()
        # may be some mask here
        res = {"graph_input": batchgraph,
               "targets": {"av_task": atom_vocab_label,
                           "bv_task": bond_vocab_label,
                           "fg_task": fgroup_label},
               "idx": idx
               }
        return res
