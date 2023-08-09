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
from transformers import AutoModel, AutoTokenizer, BatchEncoding, PreTrainedModel, PreTrainedTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from utils.data import DATASET_FEATURES


class RetrieverTokenizer:
    def __init__(self, base_model_name: str, max_question_length: int = 256, max_passage_length: int = 256) -> None:
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.max_question_length = max_question_length
        self.max_passage_length = max_passage_length

    def tokenize_questions(self, questions: list[str]) -> BatchEncoding:
        tokenized_questions = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt",
        )
        return tokenized_questions

    def tokenize_passages(self, passage_titles: list[str], passage_texts: list[str]) -> BatchEncoding:
        tokenized_passages = self.tokenizer(
            passage_titles,
            passage_texts,
            padding=True,
            truncation="only_second",
            max_length=self.max_passage_length,
            return_tensors="pt",
        )
        return tokenized_passages


class Retriever(LightningModule):
    def __init__(
        self,
        train_dataset_file: str | None = None,
        val_dataset_file: str | None = None,
        predict_dataset_file: str | None = None,
        passage_faiss_index_file: str | None = None,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        predict_batch_size: int = 32,
        base_model_name: str = "bert-base-uncased",
        max_question_length: int = 256,
        max_passage_length: int = 256,
        max_negative_passages: int = 1,
        shuffle_positive_passages: bool = False,
        shuffle_negative_passages: bool = True,
        predict_k: int = 100,
        predict_num_candidates: int = 1000,
        warmup_ratio: float = 0.06,
        lr: float = 2e-5,
        datasets_num_proc: int | None = None,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.question_encoder: PreTrainedModel = AutoModel.from_pretrained(
            self.hparams.base_model_name, add_pooling_layer=False
        )
        self.passage_encoder: PreTrainedModel = AutoModel.from_pretrained(
            self.hparams.base_model_name, add_pooling_layer=False
        )

        self.tokenizer = RetrieverTokenizer(
            self.hparams.base_model_name,
            max_question_length=self.hparams.max_question_length,
            max_passage_length=self.hparams.max_passage_length
        )

        if self.hparams.passage_faiss_index_file is not None:
            self.passage_faiss_index = self._load_passage_faiss_index(self.hparams.passage_faiss_index_file)

    def prepare_data(self):
        if self.hparams.train_dataset_file is not None:
            self._load_fit_dataset(self.hparams.train_dataset_file)
        if self.hparams.val_dataset_file is not None:
            self._load_fit_dataset(self.hparams.val_dataset_file)
        if self.hparams.predict_dataset_file is not None:
            self._load_predict_dataset(self.hparams.predict_dataset_file)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self._load_fit_dataset(self.hparams.train_dataset_file)
            self.val_dataset = self._load_fit_dataset(self.hparams.val_dataset_file)
        elif stage == "predict":
            self.predict_dataset = self._load_predict_dataset(self.hparams.predict_dataset_file)

    def _load_fit_dataset(self, dataset_file: str) -> Dataset:
        dataset = Dataset.from_json(dataset_file, features=DATASET_FEATURES, num_proc=self.hparams.datasets_num_proc)

        def _filter_example(example: dict[str, Any]) -> bool:
            return len(example["positive_passage_idxs"]) > 0

        dataset = dataset.filter(_filter_example, num_proc=self.hparams.datasets_num_proc)
        return dataset

    def _load_predict_dataset(self, dataset_file: str) -> Dataset:
        dataset = Dataset.from_json(dataset_file, features=DATASET_FEATURES, num_proc=self.hparams.datasets_num_proc)
        return dataset

    def _load_passage_faiss_index(self, passage_faiss_index_file: str) -> faiss.IndexBinaryFlat:
        passage_faiss_index = faiss.read_index_binary(self.hparams.passage_faiss_index_file)
        return passage_faiss_index

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=self.hparams.dataloader_num_workers,
            collate_fn=self._fit_collate_fn,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=self.hparams.dataloader_num_workers,
            collate_fn=self._fit_collate_fn,
            pin_memory=True,
        )
        return dataloader

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

    def _fit_collate_fn(self, examples: list[dict[str, Any]]) -> tuple[BatchEncoding, BatchEncoding, Tensor]:
        num_passages = 1 + self.hparams.max_negative_passages

        questions: list[str] = []
        passage_titles: list[str] = []
        passage_texts: list[str] = []
        passage_mask: list[int] = []

        for example in examples:
            positive_passage_idxs = example["positive_passage_idxs"]
            if self.trainer.training and self.hparams.shuffle_positive_passages:
                random.shuffle(positive_passage_idxs)

            positive_passage_idx = positive_passage_idxs[0]

            negative_passage_idxs = example["negative_passage_idxs"]
            if self.trainer.training and self.hparams.shuffle_negative_passages:
                random.shuffle(negative_passage_idxs)

            negative_passage_idxs = negative_passage_idxs[:self.hparams.max_negative_passages]

            passage_idxs = [positive_passage_idx] + negative_passage_idxs
            passages = [example["passages"][idx] for idx in passage_idxs]
            dummy_passages = [{"title": "", "text": ""}] * (num_passages - len(passages))
            assert len(passages + dummy_passages) == num_passages

            questions.append(example["question"])
            passage_titles.extend([passage["title"] for passage in passages + dummy_passages])
            passage_texts.extend([passage["text"] for passage in passages + dummy_passages])
            passage_mask.extend([1] * len(passages) + [0] * len(dummy_passages))

        tokenized_questions = self.tokenizer.tokenize_questions(questions)
        tokenized_passages = self.tokenizer.tokenize_passages(passage_titles, passage_texts)
        passage_mask = torch.tensor(passage_mask).bool()

        return tokenized_questions, tokenized_passages, passage_mask

    def _predict_collate_fn(self, examples: list[dict[str, Any]]) -> BatchEncoding:
        questions = [example["question"] for example in examples]

        tokenized_questions = self.tokenizer.tokenize_questions(questions)

        return tokenized_questions

    def training_step(self, batch: tuple[BatchEncoding, BatchEncoding], batch_idx: int) -> Tensor:
        tokenized_questions, tokenized_passages, passage_mask = batch

        dense_scores, binary_scores, positive_positions, passage_mask = self.forward(
            tokenized_questions, tokenized_passages, passage_mask
        )

        cand_loss = self._compute_cand_loss(binary_scores, positive_positions, passage_mask)
        rerank_loss = self._compute_rerank_loss(dense_scores, positive_positions, passage_mask)
        loss = cand_loss + rerank_loss

        metrics = {"train_loss": loss, "train_cand_loss": cand_loss, "train_rerank_loss": rerank_loss}
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch: tuple[BatchEncoding, BatchEncoding], batch_idx: int):
        tokenized_questions, tokenized_passages, passage_mask = batch

        dense_scores, binary_scores, positive_positions, passage_mask = self.forward(
            tokenized_questions, tokenized_passages, passage_mask
        )

        cand_loss = self._compute_cand_loss(binary_scores, positive_positions, passage_mask)
        rerank_loss = self._compute_rerank_loss(dense_scores, positive_positions, passage_mask)
        loss = cand_loss + rerank_loss

        average_rank = self._compute_average_rank(dense_scores, positive_positions, passage_mask)

        metrics = {
            "val_loss": loss,
            "val_cand_loss": cand_loss,
            "val_rerank_loss": rerank_loss,
            "val_average_rank": average_rank,
        }
        self.log_dict(metrics, sync_dist=True)

    def predict_step(self, batch: BatchEncoding, batch_idx: int) -> list[dict[str, list[int] | list[float]]]:
        tokenized_questions = batch

        batch_predictions = self.retrieve_topk_passages(
            tokenized_questions, k=self.hparams.predict_k, num_candidates=self.hparams.predict_num_candidates
        )
        return batch_predictions

    def forward(
        self, tokenized_questions: BatchEncoding, tokenized_passages: BatchEncoding, passage_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dense_questions, binary_questions = self.encode_questions(tokenized_questions)
        binary_passages = self.encode_passages(tokenized_passages)

        if dist.is_available() and dist.is_initialized():
            dense_questions = self._gather_distributed_tensors(dense_questions)
            binary_questions = self._gather_distributed_tensors(binary_questions)
            binary_passages = self._gather_distributed_tensors(binary_passages)
            passage_mask = self._gather_distributed_tensors(passage_mask)

        num_questions = dense_questions.size(0)
        num_passages = binary_passages.size(0) // num_questions

        dense_scores = torch.matmul(dense_questions, binary_passages.transpose(0, 1))
        binary_scores = torch.matmul(binary_questions, binary_passages.transpose(0, 1))

        positive_positions = torch.arange(0, num_questions * num_passages, num_passages).long().to(self.device)

        return dense_scores, binary_scores, positive_positions, passage_mask

    def encode_questions(self, tokenized_questions: BatchEncoding) -> tuple[Tensor, Tensor]:
        dense_questions = self.question_encoder(**tokenized_questions).last_hidden_state[:, 0]
        binary_questions = self._binary_encode_tensor(dense_questions)
        return dense_questions, binary_questions

    def encode_passages(self, tokenized_passages: BatchEncoding) -> Tensor:
        dense_passages = self.passage_encoder(**tokenized_passages).last_hidden_state[:, 0]
        binary_passages = self._binary_encode_tensor(dense_passages)
        return binary_passages

    def _binary_encode_tensor(self, input_tensor: Tensor)-> Tensor:
        if self.training:
            return torch.tanh(input_tensor * math.pow((1.0 + self.global_step * 0.1), 0.5))
        else:
            return torch.where(input_tensor >= 0, 1.0, -1.0).to(input_tensor.device)

    def _gather_distributed_tensors(self, input_tensor: Tensor) -> Tensor:
        gathered_tensor = self.all_gather(input_tensor.detach())
        gathered_tensor[self.global_rank] = input_tensor
        return torch.cat(list(gathered_tensor), 0)

    def _compute_cand_loss(self, binary_scores: Tensor, positive_positions: Tensor, passage_mask: Tensor) -> Tensor:
        num_questions = binary_scores.size(0)
        num_passages = binary_scores.size(1) // num_questions

        positive_mask = F.one_hot(positive_positions, num_classes=num_questions * num_passages).bool()

        positive_scores = binary_scores.masked_select(positive_mask).repeat_interleave(num_questions * num_passages - 1)
        negative_scores = binary_scores.masked_select(~positive_mask)
        passage_mask = passage_mask[None, :].masked_select(~positive_mask)
        target = torch.ones_like(positive_scores).long()

        cand_losses = F.margin_ranking_loss(positive_scores, negative_scores, target, margin=0.1, reduction="none")
        cand_loss = cand_losses.masked_fill(~passage_mask, 0.0).sum() / passage_mask.long().sum()
        return cand_loss

    def _compute_rerank_loss(self, dense_scores: Tensor, positive_positions: Tensor, passage_mask: Tensor) -> Tensor:
        rerank_loss = F.cross_entropy(dense_scores.masked_fill(~passage_mask[None, :], -1e4), positive_positions)
        return rerank_loss

    def _compute_average_rank(
        self, dense_scores: Tensor, positive_positions: Tensor, passage_mask: Tensor
    ) -> Tensor:
        positive_scores = dense_scores.take_along_dim(positive_positions[:, None], dim=1)

        ranks = (dense_scores.masked_fill(~passage_mask[None, :], -1e4) > positive_scores).sum(dim=1).float() + 1.0
        average_rank = ranks.mean()
        return average_rank

    def retrieve_topk_passages(
        self,
        tokenized_questions: BatchEncoding,
        k: int = 100,
        num_candidates: int = 1000,
    ) -> list[dict[str, list[int] | list[float]]]:
        if not hasattr(self, "passage_faiss_index"):
            self.passage_faiss_index = self._load_passage_faiss_index(self.hparams.passage_faiss_index_file)

        if num_candidates > self.passage_faiss_index.ntotal:
            raise ValueError(
                "num_candidates is larger than the number of items in passage_faiss_index "
                f"({num_candidates} > {self.passage_faiss_index.ntotal})."
            )
        if k > num_candidates:
            raise ValueError(f"k is larger than num_candidates ({k} > {num_candidates}).")

        num_questions = tokenized_questions["input_ids"].size(0)

        encoded_questions, binary_encoded_questions = self.encode_questions(tokenized_questions)

        embedded_questions = encoded_questions.detach().cpu().numpy()
        binary_embedded_questions = binary_encoded_questions.detach().cpu().numpy()
        binary_embedded_questions = ((binary_embedded_questions + 1) / 2).astype(bool)
        binary_embedded_questions = np.packbits(binary_embedded_questions, axis=1)

        _, candidate_passage_idxs = self.passage_faiss_index.search(binary_embedded_questions, num_candidates)

        binary_embedded_candidate_passages = np.vstack(
            [self.passage_faiss_index.reconstruct(idx) for idx in candidate_passage_idxs.flatten().tolist()]
        ).reshape(num_questions, num_candidates, -1)
        binary_embedded_candidate_passages = np.unpackbits(binary_embedded_candidate_passages, axis=2)
        binary_embedded_candidate_passages = binary_embedded_candidate_passages.astype(np.float32) * 2 - 1

        candidate_passage_scores = np.einsum("ijk,ik->ij", binary_embedded_candidate_passages, embedded_questions)
        topk_idxs = np.argsort(-candidate_passage_scores, axis=1)[:, :k]

        topk_passage_idxs = np.take_along_axis(candidate_passage_idxs, topk_idxs, axis=1)
        topk_passage_scores = np.take_along_axis(candidate_passage_scores, topk_idxs, axis=1)

        batch_predictions = []
        for passage_idxs, scores in zip(topk_passage_idxs.tolist(), topk_passage_scores.tolist()):
            prediction = {"passage_idxs": passage_idxs, "scores": scores}
            batch_predictions.append(prediction)

        return batch_predictions

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        num_warmup_steps = int(self.hparams.warmup_ratio * self.trainer.estimated_stepping_batches)

        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
