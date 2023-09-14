import math
import random
from typing import Any

import faiss
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from transformers.optimization import get_linear_schedule_with_warmup

from aio4_bpr_baseline.models.retriever.bpr.modeling import EncoderModel
from aio4_bpr_baseline.models.retriever.bpr.tokenization import QuestionEncoderTokenizer, PassageEncoderTokenizer
from aio4_bpr_baseline.utils.data import DATASET_FEATURES, PASSAGE_DATASET_FEATURES


class BiEncoderLightningModule(LightningModule):
    def __init__(
        self,
        train_dataset_file: str | None = None,
        val_dataset_file: str | None = None,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        base_model_name: str = "bert-base-uncased",
        max_question_length: int = 256,
        max_passage_length: int = 256,
        max_negative_passages: int = 1,
        shuffle_positive_passages: bool = False,
        shuffle_negative_passages: bool = True,
        warmup_ratio: float = 0.06,
        lr: float = 2e-5,
        datasets_num_proc: int | None = None,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.question_tokenizer = QuestionEncoderTokenizer(
            self.hparams.base_model_name, max_question_length=self.hparams.max_question_length
        )
        self.passage_tokenizer = PassageEncoderTokenizer(
            self.hparams.base_model_name, max_passage_length=self.hparams.max_passage_length
        )

        self.question_encoder = EncoderModel(self.hparams.base_model_name)
        self.passage_encoder = EncoderModel(self.hparams.base_model_name)

    def prepare_data(self):
        if self.hparams.train_dataset_file is not None:
            self._load_dataset(self.hparams.train_dataset_file)
        if self.hparams.val_dataset_file is not None:
            self._load_dataset(self.hparams.val_dataset_file)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self._load_dataset(self.hparams.train_dataset_file)
            self.val_dataset = self._load_dataset(self.hparams.val_dataset_file)

    def _load_dataset(self, dataset_file: str) -> Dataset:
        dataset = Dataset.from_json(dataset_file, features=DATASET_FEATURES, num_proc=self.hparams.datasets_num_proc)

        def _filter_example(example: dict[str, Any]) -> bool:
            return len(example["positive_passage_idxs"]) > 0

        dataset = dataset.filter(_filter_example, num_proc=self.hparams.datasets_num_proc)
        return dataset

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=self.hparams.dataloader_num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=self.hparams.dataloader_num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
        return dataloader

    def _collate_fn(self, examples: list[dict[str, Any]]) -> tuple[BatchEncoding, BatchEncoding, Tensor]:
        max_passages = 1 + self.hparams.max_negative_passages

        questions: list[str] = []
        passage_titles: list[str] = []
        passage_texts: list[str] = []
        passage_mask: list[int] = []

        for example in examples:
            questions.append(example["question"])

            positive_passage_idxs = example["positive_passage_idxs"]
            if self.trainer.training and self.hparams.shuffle_positive_passages:
                random.shuffle(positive_passage_idxs)

            negative_passage_idxs = example["negative_passage_idxs"]
            if self.trainer.training and self.hparams.shuffle_negative_passages:
                random.shuffle(negative_passage_idxs)

            passage_idxs = [positive_passage_idxs[0]] + negative_passage_idxs[:self.hparams.max_negative_passages]
            passages = [example["passages"][idx] for idx in passage_idxs]

            num_passages = len(passages)
            num_dummy_passages = max_passages - num_passages
            passages += [{"title": "", "text": ""}] * num_dummy_passages

            passage_titles.extend([passage["title"] for passage in passages])
            passage_texts.extend([passage["text"] for passage in passages])
            passage_mask.extend([1] * num_passages + [0] * num_dummy_passages)

        tokenized_questions = self.question_tokenizer(questions)
        tokenized_passages = self.passage_tokenizer(passage_titles, passage_texts)
        passage_mask = torch.tensor(passage_mask).bool()

        return tokenized_questions, tokenized_passages, passage_mask

    def forward(
        self, tokenized_questions: BatchEncoding, tokenized_passages: BatchEncoding, passage_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        encoded_questions = self.question_encoder(tokenized_questions)
        binary_encoded_questions = self.binary_encode_tensor(encoded_questions)

        encoded_passages = self.passage_encoder(tokenized_passages)
        binary_encoded_passages = self.binary_encode_tensor(encoded_passages)

        if dist.is_available() and dist.is_initialized():
            encoded_questions = self._gather_distributed_tensors(encoded_questions)
            binary_encoded_questions = self._gather_distributed_tensors(binary_encoded_questions)
            binary_encoded_passages = self._gather_distributed_tensors(binary_encoded_passages)
            passage_mask = self._gather_distributed_tensors(passage_mask)

        dense_scores = torch.matmul(encoded_questions, binary_encoded_passages.transpose(0, 1))
        binary_scores = torch.matmul(binary_encoded_questions, binary_encoded_passages.transpose(0, 1))

        return dense_scores, binary_scores, passage_mask

    def binary_encode_tensor(self, input_tensor: Tensor) -> Tensor:
        if self.training:
            return torch.tanh(input_tensor * math.pow((1.0 + self.global_step * 0.1), 0.5))
        else:
            return torch.where(input_tensor >= 0, 1.0, -1.0).to(input_tensor.device)

    def _gather_distributed_tensors(self, input_tensor: Tensor) -> Tensor:
        gathered_tensor = self.all_gather(input_tensor.detach())
        gathered_tensor[self.global_rank] = input_tensor
        return torch.cat(list(gathered_tensor), 0)

    def training_step(self, batch: tuple[BatchEncoding, BatchEncoding, Tensor], batch_idx: int) -> Tensor:
        tokenized_questions, tokenized_passages, passage_mask = batch

        dense_scores, binary_scores, passage_mask = self.forward(tokenized_questions, tokenized_passages, passage_mask)

        cand_loss = self._compute_cand_loss(binary_scores, passage_mask)
        rerank_loss = self._compute_rerank_loss(dense_scores, passage_mask)
        loss = cand_loss + rerank_loss

        metrics = {"train_loss": loss, "train_cand_loss": cand_loss, "train_rerank_loss": rerank_loss}
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch: tuple[BatchEncoding, BatchEncoding, Tensor], batch_idx: int):
        tokenized_questions, tokenized_passages, passage_mask = batch

        dense_scores, binary_scores, passage_mask = self.forward(tokenized_questions, tokenized_passages, passage_mask)

        cand_loss = self._compute_cand_loss(binary_scores, passage_mask)
        rerank_loss = self._compute_rerank_loss(dense_scores, passage_mask)
        loss = cand_loss + rerank_loss

        average_rank = self._compute_average_rank(dense_scores, passage_mask)

        metrics = {
            "val_loss": loss,
            "val_cand_loss": cand_loss,
            "val_rerank_loss": rerank_loss,
            "val_average_rank": average_rank,
        }
        self.log_dict(metrics, sync_dist=True)

    def _compute_cand_loss(self, binary_scores: Tensor, passage_mask: Tensor) -> Tensor:
        num_questions, num_total_passages = binary_scores.size()
        num_passages = num_total_passages // num_questions

        positive_positions = torch.arange(0, num_total_passages, num_passages).long().to(self.device)
        positive_mask = F.one_hot(positive_positions, num_classes=num_total_passages).bool()

        positive_scores = binary_scores.masked_select(positive_mask).repeat_interleave(num_total_passages - 1)
        negative_scores = binary_scores.masked_select(~positive_mask)
        passage_mask = passage_mask[None, :].masked_select(~positive_mask)
        target = torch.ones_like(positive_scores).long()

        cand_losses = F.margin_ranking_loss(positive_scores, negative_scores, target, margin=0.1, reduction="none")
        cand_loss = cand_losses.masked_fill(~passage_mask, 0.0).sum() / passage_mask.long().sum()
        return cand_loss

    def _compute_rerank_loss(self, dense_scores: Tensor, passage_mask: Tensor) -> Tensor:
        num_questions, num_total_passages = dense_scores.size()
        num_passages = num_total_passages // num_questions

        positive_positions = torch.arange(0, num_total_passages, num_passages).long().to(self.device)
        rerank_loss = F.cross_entropy(dense_scores.masked_fill(~passage_mask[None, :], -1e4), positive_positions)
        return rerank_loss

    def _compute_average_rank(self, dense_scores: Tensor, passage_mask: Tensor) -> Tensor:
        num_questions, num_total_passages = dense_scores.size()
        num_passages = num_total_passages // num_questions

        positive_positions = torch.arange(0, num_total_passages, num_passages).long().to(self.device)
        positive_scores = dense_scores.take_along_dim(positive_positions[:, None], dim=1)

        ranks = (dense_scores.masked_fill(~passage_mask[None, :], -1e4) > positive_scores).long().sum(dim=1) + 1
        average_rank = ranks.float().mean()
        return average_rank

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        num_warmup_steps = int(self.hparams.warmup_ratio * self.trainer.estimated_stepping_batches)

        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


class EmbedderLightningModule(LightningModule):
    def __init__(
        self,
        biencoder_ckpt_file: str,
        predict_dataset_file: str | None = None,
        predict_batch_size: int = 512,
        datasets_num_proc: int | None = None,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.biencoder_module = BiEncoderLightningModule.load_from_checkpoint(
            self.hparams.biencoder_ckpt_file, map_location="cpu", strict=False
        )
        self.biencoder_module.freeze()

    def prepare_data(self):
        if self.hparams.predict_dataset_file is not None:
            self._load_dataset(self.hparams.predict_dataset_file)

    def setup(self, stage: str):
        if stage == "predict":
            self.predict_dataset = self._load_dataset(self.hparams.predict_dataset_file)

    def _load_dataset(self, dataset_file: str) -> Dataset:
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
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
        return dataloader

    def _collate_fn(self, examples: list[dict[str, Any]]) -> BatchEncoding:
        passage_titles: list[str] = [example["title"] for example in examples]
        passage_texts: list[str] = [example["text"] for example in examples]

        tokenized_passages = self.biencoder_module.passage_tokenizer(passage_titles, passage_texts)
        return tokenized_passages

    def predict_step(self, batch: BatchEncoding, batch_idx: int) -> np.ndarray:
        tokenized_passages = batch

        encoded_passages = self.biencoder_module.passage_encoder(tokenized_passages).cpu().numpy()
        binary_encoded_passages = np.packbits(np.where(encoded_passages >= 0, 1, 0), axis=-1)

        return binary_encoded_passages


class RetrieverLightningModule(LightningModule):
    def __init__(
        self,
        biencoder_ckpt_file: str,
        passage_faiss_index_file: str,
        predict_dataset_file: str | None = None,
        predict_batch_size: int = 32,
        predict_k: int = 100,
        predict_num_candidates: int = 1000,
        datasets_num_proc: int | None = None,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.biencoder_module = BiEncoderLightningModule.load_from_checkpoint(
            self.hparams.biencoder_ckpt_file, map_location="cpu", strict=False
        )
        self.biencoder_module.freeze()

        self.passage_faiss_index = faiss.read_index_binary(self.hparams.passage_faiss_index_file)

    def prepare_data(self):
        if self.hparams.predict_dataset_file is not None:
            self._load_dataset(self.hparams.predict_dataset_file)

    def setup(self, stage: str):
        if stage == "predict":
            self.predict_dataset = self._load_dataset(self.hparams.predict_dataset_file)

    def _load_dataset(self, dataset_file: str) -> Dataset:
        dataset = Dataset.from_json(dataset_file, features=DATASET_FEATURES, num_proc=self.hparams.datasets_num_proc)
        return dataset

    def predict_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.predict_batch_size,
            shuffle=False,
            num_workers=self.hparams.dataloader_num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
        return dataloader

    def _collate_fn(self, examples: list[dict[str, Any]]) -> BatchEncoding:
        questions: list[str] = [example["question"] for example in examples]

        tokenized_questions = self.biencoder_module.question_tokenizer(questions)
        return tokenized_questions

    def predict_step(self, batch: BatchEncoding, batch_idx: int) -> list[dict[str, list[int] | list[float]]]:
        tokenized_questions = batch
        predictions = self.retrieve_passages_from_tokenized_questions(
            tokenized_questions, k=self.hparams.predict_k, num_candidates=self.hparams.predict_num_candidates
        )
        return predictions

    def retrieve_passages(
        self, questions: list[str], k: int = 100, num_candidates: int = 1000,
    ) -> list[dict[str, list[int] | list[float]]]:
        tokenized_questions = self.biencoder_module.question_tokenizer(questions).to(self.device)
        predictions = self.retrieve_passages_from_tokenized_questions(
            tokenized_questions, k=k, num_candidates=num_candidates
        )
        return predictions

    def retrieve_passages_from_tokenized_questions(
        self, tokenized_questions: BatchEncoding, k: int = 100, num_candidates: int = 1000,
    ) -> list[dict[str, list[int] | list[float]]]:
        assert not self.biencoder_module.training

        if num_candidates > self.passage_faiss_index.ntotal:
            raise ValueError(
                "num_candidates is larger than the number of items in passage_faiss_index "
                f"({num_candidates} > {self.passage_faiss_index.ntotal})."
            )
        if k > num_candidates:
            raise ValueError(f"k is larger than num_candidates ({k} > {num_candidates}).")

        num_questions, _ = tokenized_questions["input_ids"].size()

        encoded_questions = self.biencoder_module.question_encoder(tokenized_questions).detach().cpu().numpy()
        binary_encoded_questions = np.packbits(np.where(encoded_questions >= 0, 1, 0), axis=-1)

        _, candidate_passage_ids = self.passage_faiss_index.search(binary_encoded_questions, num_candidates)

        binary_encoded_candidate_passages = np.vstack(
            [self.passage_faiss_index.reconstruct(i) for i in candidate_passage_ids.flatten().tolist()]
        ).reshape(num_questions, num_candidates, -1)
        binary_encoded_candidate_passages = (
            np.unpackbits(binary_encoded_candidate_passages, axis=2).astype(np.float32) * 2.0 - 1.0
        )
        candidate_passage_scores = np.einsum("ijk,ik->ij", binary_encoded_candidate_passages, encoded_questions)
        topk_positions = np.argsort(-candidate_passage_scores, axis=1)[:, :k]

        topk_passage_ids = np.take_along_axis(candidate_passage_ids, topk_positions, axis=1)
        topk_passage_scores = np.take_along_axis(candidate_passage_scores, topk_positions, axis=1)

        batch_predictions = []
        for passage_ids, scores in zip(topk_passage_ids.tolist(), topk_passage_scores.tolist()):
            prediction = {"passage_ids": passage_ids, "scores": scores}
            batch_predictions.append(prediction)

        return batch_predictions
