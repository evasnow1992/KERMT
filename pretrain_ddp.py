# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# MIT License

# Copyright (c) 2021 Tencent AI Lab.  All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os


from kermt.data.kermtdataset import get_data, split_data, KermtCollator, KermtDecoderCollator
from kermt.util.utils import create_logger
from kermt.model.models import KERMTEmbedding
from task.kermttrainer import KERMTTrainer, KERMTCMIMTrainer
from kermt.util.scheduler import NoamLR
from kermt.data.torchvocab import MolVocab, SMILESVocab
from kermt.data.kermtdataset import BatchMolDataset
from kermt.util.parsing import parse_args_ddp
from kermt.util.nn_utils import param_count_trainable, param_count_total

def pre_load_data_ddp(dataset: BatchMolDataset, dataset_size: int, samples_per_file: int):
    for i in range(1, dataset_size, samples_per_file):
        dataset.load_data(i)

def configure_nccl_for_topology():
    """
    Auto-configure NCCL settings based on GPU topology.
    This handles cases where P2P (peer-to-peer) GPU communication is not available.
    Must be called BEFORE spawning processes (in main process).
    """
    # Check if user has already set NCCL settings (don't override)
    if "NCCL_P2P_DISABLE" in os.environ:
        print(f"[INFO] Using user-provided NCCL settings: NCCL_P2P_DISABLE={os.environ['NCCL_P2P_DISABLE']}")
        return
    
    # Try to detect GPU topology
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', 'topo', '-m'], 
                              capture_output=True, text=True, timeout=5)
        topo_output = result.stdout
        
        # Check for poor GPU connectivity (SYS or NODE topology)
        # These topologies typically don't support P2P well
        if 'SYS' in topo_output or 'NODE' in topo_output:
            print("[INFO] Detected cross-NUMA or system-level GPU topology (SYS/NODE).")
            print("[INFO] Disabling P2P for stability. This is normal for multi-socket systems.")
            os.environ["NCCL_P2P_DISABLE"] = "1"
            os.environ["NCCL_IB_DISABLE"] = "1"
            os.environ["NCCL_SHM_DISABLE"] = "0"
        else:
            print("[INFO] GPU topology appears to support P2P. Enabling P2P communication.")
    except Exception as e:
        # If detection fails, use safe defaults (disable P2P)
        print(f"[WARNING] Could not detect GPU topology: {e}")
        print("[INFO] Using safe default: P2P disabled. Set NCCL_P2P_DISABLE=0 to enable if your system supports it.")
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_SHM_DISABLE"] = "0"

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)

    # parse args
    args = parse_args_ddp()

    if rank == 0:
        print(f"{args=}")
    logger = create_logger(name='pretrain', save_dir=args.save_dir)

    # Build train, val, and test datasets
    train_data, train_sample_per_file = get_data(data_path=args.train_data_path)
    train_data_size = len(train_data)
    print(f"Training data size: {train_data_size}")
    pre_load_data_ddp(train_data, train_data_size, train_sample_per_file)

    if args.val_data_path is not None:
        val_data, val_sample_per_file = get_data(data_path=args.val_data_path)
        val_data_size = len(val_data)
        print(f"Validation data size: {val_data_size}")
        pre_load_data_ddp(val_data, val_data_size, val_sample_per_file)
    else:
        val_data = None
        val_data_size = 0

    if args.test_data_path is not None:
        raise NotImplementedError("Test data is not implemented")
        test_data, test_sample_per_file = get_data(data_path=args.test_data_path)
        test_data_size = len(test_data)
        print(f"Test data size: {test_data_size}")
        pre_load_data_ddp(test_data, test_data_size, test_sample_per_file)
    else:
        test_data = None
        test_data_size = 0

    train_sampler = DistributedSampler(
            train_data, num_replicas=world_size, rank=rank, shuffle=True)
    
    if args.val_data_path is not None:
        val_sampler = DistributedSampler(
            val_data, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        val_sampler = None

    if args.test_data_path is not None:
        test_sampler = DistributedSampler(
            test_data, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        test_sampler = None

    # Build collator based on training mode
    shared_dict = {}
    
    if args.use_cmim:
        # CMIM mode - only needs SMILES vocabulary
        if args.smiles_vocab_path is None:
            raise ValueError(
                "CMIM training (--use_cmim) requires --smiles_vocab_path\n"
                "Example: --smiles_vocab_path path/to/pretrain_smiles_vocab.pkl"
            )
        if rank == 0:
            print("[INFO] CMIM mode: Loading SMILES vocabulary")
        smiles_vocab = SMILESVocab.load_vocab(args.smiles_vocab_path)
        smiles_vocab_size = len(smiles_vocab)
        if rank == 0:
            print(f"[INFO] SMILES vocabulary size: {smiles_vocab_size}")
        # CMIM training - use KermtDecoderCollator
        mol_collator = KermtDecoderCollator(
            shared_dict=shared_dict,
            smiles_vocab=smiles_vocab,
            args=args
        )
    else:
        # Vocab-based pretraining mode - only needs atom and bond vocabularies
        if args.atom_vocab_path is None or args.bond_vocab_path is None:
            raise ValueError(
                "Vocab-based pretraining requires --atom_vocab_path and --bond_vocab_path\n"
                "Example: --atom_vocab_path path/to/pretrain_atom_vocab.pkl "
                "--bond_vocab_path path/to/pretrain_bond_vocab.pkl"
            )
        if rank == 0:
            print("[INFO] Vocab-based mode: Loading atom and bond vocabularies")
        atom_vocab = MolVocab.load_vocab(args.atom_vocab_path)
        bond_vocab = MolVocab.load_vocab(args.bond_vocab_path)
        atom_vocab_size, bond_vocab_size = len(atom_vocab), len(bond_vocab)
        if rank == 0:
            print(f"[INFO] Atom vocabulary size: {atom_vocab_size}, Bond vocabulary size: {bond_vocab_size}")
        # Vocab-based pretraining - use KermtCollator
        mol_collator = KermtCollator(
            shared_dict=shared_dict,
            atom_vocab=atom_vocab,
            bond_vocab=bond_vocab,
            args=args
        )

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, # batch size per GPU (aka micro batch size)
                                  shuffle=False, # because train_sampler does the shuffling
                                  num_workers=args.num_dataloader_workers,
                                  sampler=train_sampler,
                                  collate_fn=mol_collator,
                                  drop_last=True)
    
    if args.val_data_path is not None:
        val_dataloader = DataLoader(val_data, batch_size=args.batch_size, 
                                  shuffle=False, # because no shuffling needed
                                  num_workers=args.num_dataloader_workers,
                                  sampler=val_sampler,
                                  collate_fn=mol_collator,
                                  drop_last=True)
    else:
        val_dataloader = None
        
    # Build model - create complete task model based on training mode
    # This ensures all model parameters (encoder, decoder, heads) are included
    kermt_embedding = KERMTEmbedding(args)
    
    if args.use_cmim:
        # Build complete CMIM task model (encoder + latent distribution + decoder)
        if rank == 0:
            print("[INFO] Building KermtCMIMTask model")
        from kermt.model.models import KermtCMIMTask
        model = KermtCMIMTask(
            args,
            kermt=kermt_embedding,
            latent_dim=args.latent_dim,
            contrastive_temperature=args.contrastive_temperature,
            smiles_vocab_size=smiles_vocab_size
        )
    else:
        # Build complete vocab task model (encoder + vocab prediction heads)
        if rank == 0:
            print("[INFO] Building KermtTask model")
        from kermt.model.models import KermtTask
        fg_size = 85
        model = KermtTask(
            args,
            kermt=kermt_embedding,
            atom_vocab_size=atom_vocab_size,
            bond_vocab_size=bond_vocab_size,
            fg_size=fg_size
        )
    
    print(f'Number of trainable parameters = {param_count_trainable(model):,}')
    print(f'Number of total parameters = {param_count_total(model):,}')

    # Build optimizer on COMPLETE model (includes all trainable components)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)

    # Build Learning rate scheduler   
    steps_per_epoch = train_data_size // (args.batch_size*world_size)
    scheduler = NoamLR(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        init_lr=args.init_lr,
        max_lr=args.max_lr,
        final_lr=args.final_lr,
        fine_tune_coff=args.fine_tune_coff
    )
 
    # Build trainer - pass the complete model
    if args.use_cmim:
        if rank == 0:
            print("[INFO] Initializing KERMTCMIMTrainer")
        trainer = KERMTCMIMTrainer(
            args=args,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            gpu_id=rank,
            n_steps=0,
            logger=logger
        )
    else:
        if rank == 0:
            print("[INFO] Initializing KERMTTrainer (vocab-based pretraining)")
        trainer = KERMTTrainer(
            args=args,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            gpu_id=rank,
            n_steps=0,
            logger=logger
        )

    if args.save_dir is not None:
        last_ckpt_path = os.path.join(args.save_dir, "last_checkpoint.pt")
        if os.path.exists(last_ckpt_path):
            print(f"Loading checkpoint from {last_ckpt_path}")
            epoch, scheduler_step, prev_batch_idx = trainer.load(last_ckpt_path)
            print(f"Loaded checkpoint from epoch={epoch}, scheduler_step={scheduler_step}, prev_batch_idx={prev_batch_idx}")
        else:
            epoch = 0
            scheduler_step = 0
            prev_batch_idx = 0

    steps_per_epoch = train_data_size // (args.batch_size*world_size)
    print(f"Steps per epoch: {steps_per_epoch}")
    curr_epoch_batch_idx = scheduler_step % steps_per_epoch
    print(f"Current epoch batch index: {curr_epoch_batch_idx}")

    trainer.set_batch_idx(prev_batch_idx)
    # Train model
    trainer.train(start_epoch=epoch, max_epochs=args.epochs)
    destroy_process_group()


if __name__ == "__main__":

    world_size = os.environ.get("WORLD_SIZE", 1)
    world_size = int(world_size)
    print(f"World size: {world_size}")
    
    # Auto-configure NCCL before spawning processes
    # This detects GPU topology and sets appropriate P2P settings
    configure_nccl_for_topology()
    
    mp.spawn(main, args=(world_size, ), nprocs=world_size)
