import pytorch_lightning as pl
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
from typing import List


class SMILESDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: GPT2Tokenizer, block_size: int):
        with open(data_path, "r", encoding="utf-8") as f:
            data = f.read()
        self.examples = tokenizer.batch_encode_plus(
            [data], max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class GPT2Trainer(pl.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.hparams.tokenizer_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(
            self.hparams.model_name_or_path)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch, batch
        loss = self.model(inputs, labels=labels).loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(
            self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def train_dataloader(self):
        dataset = SMILESDataset(
            self.hparams.data_path, self.tokenizer, self.hparams.block_size)
        dataloader = DataLoader(
            dataset, batch_size=self.hparams.batch_size, shuffle=True)
        return dataloader