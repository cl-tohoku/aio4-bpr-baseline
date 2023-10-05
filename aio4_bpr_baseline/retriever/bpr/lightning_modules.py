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

from aio4_bpr_baseline.retriever.bpr.modeling import BPREncoderModel
from aio4_bpr_baseline.retriever.bpr.tokenization import BPRPassageTokenizer, BPRQuestionTokenizer
from aio4_bpr_baseline.utils.data import DATASET_FEATURES, PASSAGES_FEATURES


class BPRBiencoderLightningModule(LightningModule):
    def __init__(
        self,
        train_dataset_file: str,
        val_dataset_file: str,
        passages_file: str,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        base_model_name: str = "bert-base-uncased",
        share_encoders: bool = False,
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

        self.question_tokenizer = BPRQuestionTokenizer(
            self.hparams.base_model_name, max_question_length=self.hparams.max_question_length
        )
        self.passage_tokenizer = BPRPassageTokenizer(
            self.hparams.base_model_name, max_passage_length=self.hparams.max_passage_length
        )

        self.question_encoder = BPREncoderModel(self.hparams.base_model_name)
        if self.hparams.share_encoders:
            self.passage_encoder = self.question_encoder
        else:
            self.passage_encoder = BPREncoderModel(self.hparams.base_model_name)

    def prepare_data(self):
        self._load_passages(self.hparams.passages_file)
        self._load_dataset(self.hparams.train_dataset_file)
        self._load_dataset(self.hparams.val_dataset_file)

    def setup(self, stage: str):
        self.all_passages = self._load_passages(self.hparams.passages_file)
        self.train_dataset = self._load_dataset(self.hparams.train_dataset_file)
        self.val_dataset = self._load_dataset(self.hparams.val_dataset_file)

    def _load_passages(self, passages_file: str) -> Dataset:
        return Dataset.from_json(passages_file, features=PASSAGES_FEATURES)

    def _load_dataset(self, dataset_file: str) -> Dataset:
        def _filter_example(example: dict[str, Any]) -> bool:
            if len(example["positive_passages"]) == 0:
                return False

            return True

        dataset = Dataset.from_json(dataset_file, features=DATASET_FEATURES)
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

        questions = []
        passage_titles = []
        passage_texts = []
        passage_mask = []

        for example in examples:
            questions.append(example["question"])

            positive_passages = example["positive_passages"]
            if self.trainer.training and self.hparams.shuffle_positive_passages:
                random.shuffle(positive_passages)

            negative_passages = example["negative_passages"]
            if self.trainer.training and self.hparams.shuffle_negative_passages:
                random.shuffle(negative_passages)

            passages = [positive_passages[0]] + negative_passages[: self.hparams.max_negative_passages]

            num_passages = len(passages)
            passage_titles.extend([self.all_passages[passage["idx"]]["title"] for passage in passages])
            passage_texts.extend([self.all_passages[passage["idx"]]["text"] for passage in passages])
            passage_mask.extend([1] * num_passages)

            num_dummy_passages = max_passages - num_passages
            passage_titles.extend([""] * num_dummy_passages)
            passage_texts.extend([""] * num_dummy_passages)
            passage_mask.extend([0] * num_dummy_passages)

        tokenized_questions = self.question_tokenizer(questions)
        tokenized_passages = self.passage_tokenizer(passage_titles, passage_texts)
        passage_mask = torch.tensor(passage_mask).bool()

        return tokenized_questions, tokenized_passages, passage_mask

    def forward(
        self, tokenized_questions: BatchEncoding, tokenized_passages: BatchEncoding, passage_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        encoded_questions = self.question_encoder(tokenized_questions)
        binary_encoded_questions = self._binary_encode_tensor(encoded_questions)

        encoded_passages = self.passage_encoder(tokenized_passages)
        binary_encoded_passages = self._binary_encode_tensor(encoded_passages)

        if dist.is_available() and dist.is_initialized():
            encoded_questions = self._gather_distributed_tensors(encoded_questions)
            binary_encoded_questions = self._gather_distributed_tensors(binary_encoded_questions)
            binary_encoded_passages = self._gather_distributed_tensors(binary_encoded_passages)
            passage_mask = self._gather_distributed_tensors(passage_mask)

        return encoded_questions, binary_encoded_questions, binary_encoded_passages, passage_mask

    def _binary_encode_tensor(self, input_tensor: Tensor) -> Tensor:
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

        encoded_questions, binary_encoded_questions, binary_encoded_passages, passage_mask = self.forward(
            tokenized_questions, tokenized_passages, passage_mask
        )
        dense_scores = torch.matmul(encoded_questions, binary_encoded_passages.transpose(0, 1))
        binary_scores = torch.matmul(binary_encoded_questions, binary_encoded_passages.transpose(0, 1))

        cand_loss = self._compute_cand_loss(binary_scores, passage_mask)
        self.log("train_cand_loss", cand_loss)

        rerank_loss = self._compute_rerank_loss(dense_scores, passage_mask)
        self.log("train_rerank_loss", rerank_loss)

        loss = cand_loss + rerank_loss
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: tuple[BatchEncoding, BatchEncoding, Tensor], batch_idx: int):
        tokenized_questions, tokenized_passages, passage_mask = batch

        encoded_questions, binary_encoded_questions, binary_encoded_passages, passage_mask = self.forward(
            tokenized_questions, tokenized_passages, passage_mask
        )
        dense_scores = torch.matmul(encoded_questions, binary_encoded_passages.transpose(0, 1))
        binary_scores = torch.matmul(binary_encoded_questions, binary_encoded_passages.transpose(0, 1))

        cand_loss = self._compute_cand_loss(binary_scores, passage_mask)
        self.log("val_cand_loss", cand_loss, sync_dist=True)

        rerank_loss = self._compute_rerank_loss(dense_scores, passage_mask)
        self.log("val_rerank_loss", rerank_loss, sync_dist=True)

        loss = cand_loss + rerank_loss
        self.log("val_loss", loss, sync_dist=True)

        average_rank = self._compute_average_rank(dense_scores, passage_mask)
        self.log("val_average_rank", average_rank, sync_dist=True)

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


class BPREmbedderLightningModule(LightningModule):
    def __init__(
        self,
        biencoder_ckpt_file: str,
        passages_file: str | None = None,
        batch_size: int = 512,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.biencoder_module = BPRBiencoderLightningModule.load_from_checkpoint(
            self.hparams.biencoder_ckpt_file, map_location="cpu", strict=False
        )
        self.biencoder_module.freeze()

    def prepare_data(self):
        self._load_passages(self.hparams.passages_file)

    def setup(self, stage: str):
        self.all_passages = self._load_passages(self.hparams.passages_file)

    def _load_passages(self, passages_file: str) -> Dataset:
        return Dataset.from_json(passages_file, features=PASSAGES_FEATURES)

    def predict_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.all_passages,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.dataloader_num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
        return dataloader

    def _collate_fn(self, examples: list[dict[str, Any]]) -> BatchEncoding:
        passage_titles = [example["title"] for example in examples]
        passage_texts = [example["text"] for example in examples]

        tokenized_passages = self.biencoder_module.passage_tokenizer(passage_titles, passage_texts)
        return tokenized_passages

    def predict_step(self, batch: BatchEncoding, batch_idx: int) -> np.ndarray:
        tokenized_passages = batch
        return self.embed_passages_tokenized(tokenized_passages)

    def embed_passages(self, passage_titles: list[str], passage_texts: list[str]) -> np.ndarray:
        tokenized_passages = self.biencoder_module.passage_tokenizer(passage_titles, passage_texts).to(self.device)
        return self.embed_passages_tokenized(tokenized_passages)

    def embed_passages_tokenized(self, tokenized_passages: BatchEncoding) -> np.ndarray:
        assert not self.biencoder_module.passage_encoder.training

        embedded_passages = self.biencoder_module.passage_encoder(tokenized_passages).cpu().numpy()
        binary_embedded_passages = np.packbits(np.where(embedded_passages >= 0, 1, 0), axis=-1)
        return binary_embedded_passages


class BPRRetrieverLightningModule(LightningModule):
    def __init__(
        self,
        biencoder_ckpt_file: str,
        passage_embeddings_file: str,
        predict_dataset_file: str | None = None,
        predict_batch_size: int = 32,
        predict_num_passages: int = 100,
        predict_num_candidates: int = 1000,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.biencoder_module = BPRBiencoderLightningModule.load_from_checkpoint(
            self.hparams.biencoder_ckpt_file, map_location="cpu", strict=False
        )
        self.biencoder_module.freeze()

        self.passage_faiss_index = self._build_passage_faiss_index(self.hparams.passage_embeddings_file)

    def _build_passage_faiss_index(
        self,
        passage_embeddings_file: str,
        batch_size: int = 1000,
    ) -> faiss.IndexBinaryFlat:
        embeddings = np.load(passage_embeddings_file)
        num_passages, binary_embedding_size = embeddings.shape

        faiss_index = faiss.IndexBinaryFlat(binary_embedding_size * 8)
        for start in range(0, num_passages, batch_size):
            faiss_index.add(embeddings[start : start + batch_size])

        return faiss_index

    def prepare_data(self):
        self._load_dataset(self.hparams.predict_dataset_file)

    def setup(self, stage: str):
        self.predict_dataset = self._load_dataset(self.hparams.predict_dataset_file)

    def _load_dataset(self, dataset_file: str) -> Dataset:
        return Dataset.from_json(dataset_file, features=DATASET_FEATURES)

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
        questions = [example["question"] for example in examples]

        tokenized_questions = self.biencoder_module.question_tokenizer(questions)
        return tokenized_questions

    def predict_step(self, batch: BatchEncoding, batch_idx: int) -> list[list[dict[str, int | float]]]:
        tokenized_questions = batch
        return self.retrieve_passages_tokenized(
            tokenized_questions,
            num_passages=self.hparams.predict_num_passages,
            num_candidates=self.hparams.predict_num_candidates,
        )

    def retrieve_passages(
        self, questions: list[str], num_passages: int = 100, num_candidates: int = 1000
    ) -> list[list[dict[str, int | float]]]:
        tokenized_questions = self.biencoder_module.question_tokenizer(questions).to(self.device)
        return self.retrieve_passages_tokenized(
            tokenized_questions, num_passages=num_passages, num_candidates=num_candidates
        )

    def retrieve_passages_tokenized(
        self, tokenized_questions: BatchEncoding, num_passages: int = 100, num_candidates: int = 1000
    ) -> list[list[dict[str, int | float]]]:
        assert not self.biencoder_module.question_encoder.training

        num_candidates = min(num_candidates, self.passage_faiss_index.ntotal)
        num_passages = min(num_passages, num_candidates)

        num_questions, _ = tokenized_questions["input_ids"].size()

        embedded_questions = self.biencoder_module.question_encoder(tokenized_questions).cpu().numpy()
        binary_embedded_questions = np.packbits(np.where(embedded_questions >= 0, 1, 0), axis=-1)

        _, candidate_passage_idxs = self.passage_faiss_index.search(binary_embedded_questions, num_candidates)

        binary_embedded_candidate_passages = np.vstack(
            [self.passage_faiss_index.reconstruct(i) for i in candidate_passage_idxs.flatten().tolist()]
        ).reshape(num_questions, num_candidates, -1)
        binary_embedded_candidate_passages = (
            np.unpackbits(binary_embedded_candidate_passages, axis=2).astype(np.float32) * 2.0 - 1.0
        )
        candidate_passage_scores = np.einsum("ijk,ik->ij", binary_embedded_candidate_passages, embedded_questions)
        topk_passage_positions = np.argsort(-candidate_passage_scores, axis=1)[:, :num_passages]

        topk_passage_idxs = np.take_along_axis(candidate_passage_idxs, topk_passage_positions, axis=1)
        topk_passage_scores = np.take_along_axis(candidate_passage_scores, topk_passage_positions, axis=1)

        batch_predictions = []
        for passage_idxs, scores in zip(topk_passage_idxs.tolist(), topk_passage_scores.tolist()):
            prediction = [{"idx": idx, "score": score} for idx, score in zip(passage_idxs, scores)]
            batch_predictions.append(prediction)

        return batch_predictions
