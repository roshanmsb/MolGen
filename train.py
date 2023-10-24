import argparse
import os

import torch
from dvclive.lightning import DVCLiveLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)

from models.data import MolGenDataModule
from models.gpt2 import GPT2MolGen

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
# Set matmul precision
torch.set_float32_matmul_precision("medium")


def main(args):
    root = args.root
    max_seq_length = args.max_seq_length

    # Initialize DataModule
    datamodule = MolGenDataModule(
        tokenizer_path=os.path.join(root, "tokenizers/BPE_pubchem_500.json"),
        dataset_path=os.path.join(root, "pubchem/part-0000.snappy.parquet"),
        file_type="parquet",
        overwrite_cache=True,
        max_seq_length=max_seq_length,
        batch_size=args.batch_size,
        preprocess_num_workers=args.preprocess_num_workers,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    # Initialize model
    model = GPT2MolGen(
        model_name_or_path="gpt2",
        max_seq_length=max_seq_length,
        vocab_size=30002,
        bos_token_id="<bos>",
        eos_token_id="<eos>",
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
    )

    # Initialize trainer
    logger = DVCLiveLogger(save_dvc_exp=True)
    checkpoint_callback = ModelCheckpoint(monitor="valid_loss", mode="min")
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=args.epochs,
        devices=args.devices,
        strategy=args.strategy,
        logger=logger,
        callbacks=[checkpoint_callback, RichModelSummary(), RichProgressBar()],
        precision="bf16-mixed",
    )
    # Train model
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="./data/processed/training/"
    )
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--strategy", type=str, default="fsdp")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess_num_workers", type=int, default=20)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
