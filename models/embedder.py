from typing import Any

import numpy as np
from datasets import Dataset
from lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from models.retriever import Retriever
from utils.data import PASSAGE_DATASET_FEATURES


class Embedder(LightningModule):
    def __init__(
        self,
        retriever_ckpt_file: str | None = None,
        predict_dataset_file: str | None = None,
        predict_batch_size: int = 512,
        datasets_num_proc: int | None = None,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.biencoder = Retriever.load_from_checkpoint(
            self.hparams.retriever_ckpt_file, map_location="cpu", strict=False
        )

    def prepare_data(self):
        if self.hparams.predict_dataset_file is not None:
            self._load_predict_dataset(self.hparams.predict_dataset_file)

    def setup(self, stage: str):
        if stage == "predict":
            self.predict_dataset = self._load_predict_dataset(self.hparams.predict_dataset_file)

    def _load_predict_dataset(self, dataset_file: str) -> Dataset:
        dataset = Dataset.from_json(
            dataset_file, features=PASSAGE_DATASET_FEATURES, num_proc=self.hparams.datasets_num_proc
        )
        return dataset

    def predict_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.predict_batch_size,
            shuffle=False,
            num_workers=self.hparams.dataloader_num_workers,
            collate_fn=self._predict_collate_fn,
            pin_memory=True,
        )
        return dataloader

    def _predict_collate_fn(self, examples: list[dict[str, Any]]) -> BatchEncoding:
        passage_titles: list[str] = [example["title"] for example in examples]
        passage_texts: list[str] = [example["text"] for example in examples]

        tokenized_passages = self.biencoder.tokenizer.tokenize_passages(passage_titles, passage_texts)

        return tokenized_passages

    def predict_step(self, batch: BatchEncoding, batch_idx: int) -> np.ndarray:
        tokenized_passages = batch

        embedded_passages = self.embed_passages(tokenized_passages)

        return embedded_passages

    def embed_passages(self, tokenized_passages: BatchEncoding) -> np.ndarray:
        binary_encoded_passages = self.biencoder.encode_passages(tokenized_passages)

        embedded_passages = binary_encoded_passages.cpu().numpy()
        embedded_passages = ((embedded_passages + 1) / 2).astype(bool)
        embedded_passages = np.packbits(embedded_passages, axis=1)

        return embedded_passages
