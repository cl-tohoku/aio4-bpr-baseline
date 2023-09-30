import json
import random
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from transformers.optimization import get_linear_schedule_with_warmup

from aio4_bpr_baseline.models.reader.extractive_reader.modeling import ReaderModel
from aio4_bpr_baseline.models.reader.extractive_reader.tokenization import ReaderTokenizer
from aio4_bpr_baseline.utils.data import DATASET_FEATURES


class ReaderLightningModule(LightningModule):
    def __init__(
        self,
        train_dataset_file: str | None = None,
        val_dataset_file: str | None = None,
        train_gold_passages_info_file: str | None = None,
        val_gold_passages_info_file: str | None = None,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        base_model_name: str = "bert-base-uncased",
        max_input_length: int = 350,
        max_negative_passages: int = 23,
        max_answer_length: int = 10,
        max_answer_spans: int = 10,
        shuffle_positive_passages: bool = True,
        shuffle_negative_passages: bool = True,
        warmup_ratio: float = 0.06,
        lr: float = 2e-5,
        datasets_num_proc: int | None = None,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = ReaderTokenizer(self.hparams.base_model_name, max_input_length=self.hparams.max_input_length)
        self.reader = ReaderModel(self.hparams.base_model_name)

    def prepare_data(self):
        if self.hparams.train_dataset_file is not None:
            self._load_dataset(
                self.hparams.train_dataset_file, gold_passages_info_file=self.hparams.train_gold_passages_info_file
            )
        if self.hparams.val_dataset_file is not None:
            self._load_dataset(
                self.hparams.val_dataset_file, gold_passages_info_file=self.hparams.val_gold_passages_info_file
            )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self._load_dataset(
                self.hparams.train_dataset_file, gold_passages_info_file=self.hparams.train_gold_passages_info_file
            )
            self.val_dataset = self._load_dataset(
                self.hparams.val_dataset_file, gold_passages_info_file=self.hparams.val_gold_passages_info_file
            )

    def _load_dataset(self, dataset_file: str, gold_passages_info_file: str | None = None) -> Dataset:
        gold_passages_info: dict[str, dict[str, str]] = {}

        if gold_passages_info_file is not None:
            for item in json.load(open(gold_passages_info_file))["data"]:
                question = item["question"]
                question_tokens = item.get("question_tokens")

                passage = {"title": item["title"], "text": item["context"]}

                gold_passages_info[question] = passage
                if question_tokens is not None:
                    gold_passages_info[question_tokens] = passage

        tokenizer = ReaderTokenizer(self.hparams.base_model_name, max_input_length=self.hparams.max_input_length)
        max_answer_length = self.hparams.max_answer_length

        def _preprocess_example(example: dict[str, Any]) -> dict[str, Any]:
            def _filter_passage_idxs(passage_idxs: list[int]) -> list[int]:
                num_passages = len(passage_idxs)
                if num_passages == 0:
                    return []

                question = example["question"]

                passages = [example["passages"][idx] for idx in passage_idxs]
                passage_titles = [passage["title"] for passage in passages]
                passage_texts = [passage["text"] for passage in passages]

                tokenized_inputs, _, span_mask = tokenizer(
                    [question], [passage_titles], [passage_texts], return_tensors="np"
                )
                input_ids = tokenized_inputs["input_ids"]

                filtered_passage_idxs: list[int] = []
                for j in range(num_passages):
                    answer_spans = []
                    for answer in example["answers"]:
                        answer_spans += tokenizer.find_answer_spans(
                            answer,
                            input_ids[0, j, :].tolist(),
                            span_mask=span_mask[0, j, :].tolist(),
                            max_answer_length=max_answer_length,
                        )

                    if len(answer_spans) > 0:
                        filtered_passage_idxs.append(passage_idxs[j])

                return filtered_passage_idxs

            filtered_positive_passage_idxs: list[int] = []
            if example["question"] in gold_passages_info:
                gold_passage_title = gold_passages_info[example["question"]]["title"]
                gold_positive_passage_idxs = [
                    idx
                    for idx in example["positive_passage_idxs"]
                    if example["passages"][idx]["title"].lower() == gold_passage_title.lower()
                ]
                filtered_positive_passage_idxs = _filter_passage_idxs(gold_positive_passage_idxs)

            if len(filtered_positive_passage_idxs) == 0:
                filtered_positive_passage_idxs = _filter_passage_idxs(example["positive_passage_idxs"])

            example["positive_passage_idxs"] = filtered_positive_passage_idxs
            return example

        def _filter_example(example: dict[str, Any]) -> bool:
            return len(example["positive_passage_idxs"]) > 0

        dataset = Dataset.from_json(dataset_file, features=DATASET_FEATURES, num_proc=self.hparams.datasets_num_proc)
        dataset = dataset.map(_preprocess_example, num_proc=self.hparams.datasets_num_proc)
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

    def _collate_fn(
        self, examples: list[dict[str, Any]]
    ) -> tuple[BatchEncoding, Tensor, Tensor, Tensor, Tensor, Tensor]:
        num_questions = len(examples)

        questions: list[str] = []
        passage_titles: list[list[str]] = []
        passage_texts: list[list[str]] = []

        for example in examples:
            questions.append(example["question"])

            positive_passage_idxs = example["positive_passage_idxs"]
            if self.trainer.training and self.hparams.shuffle_positive_passages:
                random.shuffle(positive_passage_idxs)

            negative_passage_idxs = example["negative_passage_idxs"]
            if self.trainer.training and self.hparams.shuffle_negative_passages:
                random.shuffle(negative_passage_idxs)

            passage_idxs = [positive_passage_idxs[0]] + negative_passage_idxs[: self.hparams.max_negative_passages]
            passages = [example["passages"][idx] for idx in passage_idxs]

            passage_titles.append([passage["title"] for passage in passages])
            passage_texts.append([passage["text"] for passage in passages])

        tokenized_inputs, passage_mask, span_mask = self.tokenizer(questions, passage_titles, passage_texts)

        positive_input_ids = tokenized_inputs["input_ids"][:, 0, :]
        positive_span_mask = span_mask[:, 0, :]

        answer_starts: list[list[int]] = []
        answer_ends: list[list[int]] = []
        answer_mask: list[list[int]] = []

        for i in range(num_questions):
            answer_spans = []
            for answer in examples[i]["answers"]:
                answer_spans += self.tokenizer.find_answer_spans(
                    answer,
                    positive_input_ids[i].tolist(),
                    span_mask=positive_span_mask[i].tolist(),
                    max_answer_length=self.hparams.max_answer_length,
                )

            answer_spans = answer_spans[: self.hparams.max_answer_spans]
            num_answer_spans = len(answer_spans)
            assert num_answer_spans > 0

            num_dummy_answer_spans = self.hparams.max_answer_spans - num_answer_spans
            answer_spans += [(0, 0)] * num_dummy_answer_spans

            answer_starts.append([span[0] for span in answer_spans])
            answer_ends.append([span[1] for span in answer_spans])
            answer_mask.append([1] * num_answer_spans + [0] * num_dummy_answer_spans)

        answer_starts = torch.tensor(answer_starts)
        answer_ends = torch.tensor(answer_ends)
        answer_mask = torch.tensor(answer_mask).bool()

        return tokenized_inputs, passage_mask, span_mask, answer_starts, answer_ends, answer_mask

    def training_step(
        self, batch: tuple[BatchEncoding, Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        tokenized_inputs, passage_mask, span_mask, answer_starts, answer_ends, answer_mask = batch

        classifier_logits, start_logits, end_logits = self.reader(tokenized_inputs)

        positive_start_logits = start_logits[:, 0, :]
        positive_end_logits = end_logits[:, 0, :]
        positive_span_mask = span_mask[:, 0, :]

        classifier_loss = self._compute_classifier_loss(classifier_logits, passage_mask)
        span_loss = self._compute_span_loss(
            positive_start_logits, positive_end_logits, positive_span_mask, answer_starts, answer_ends, answer_mask
        )
        loss = classifier_loss + span_loss

        metrics = {"train_loss": loss, "train_classifier_loss": classifier_loss, "train_span_loss": span_loss}
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch: tuple[BatchEncoding, Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int):
        tokenized_inputs, passage_mask, span_mask, answer_starts, answer_ends, answer_mask = batch

        classifier_logits, start_logits, end_logits = self.reader(tokenized_inputs)

        positive_start_logits = start_logits[:, 0, :]
        positive_end_logits = end_logits[:, 0, :]
        positive_span_mask = span_mask[:, 0, :]

        classifier_loss = self._compute_classifier_loss(classifier_logits, passage_mask)
        span_loss = self._compute_span_loss(
            positive_start_logits, positive_end_logits, positive_span_mask, answer_starts, answer_ends, answer_mask
        )
        loss = classifier_loss + span_loss

        classifier_accuracy = self._compute_classifier_accuracy(classifier_logits, passage_mask)
        span_accuracy = self._compute_span_accuracy(
            positive_start_logits, positive_end_logits, positive_span_mask, answer_starts, answer_ends, answer_mask
        )
        joint_accuracy = classifier_accuracy * span_accuracy

        metrics = {
            "val_loss": loss,
            "val_classifier_loss": classifier_loss,
            "val_span_loss": span_loss,
            "val_classifier_accuracy": classifier_accuracy,
            "val_span_accuracy": span_accuracy,
            "val_joint_accuracy": joint_accuracy,
        }
        self.log_dict(metrics)

    def _compute_classifier_loss(self, classifier_logits: Tensor, passage_mask: Tensor) -> Tensor:
        num_questions, _ = classifier_logits.size()

        labels = classifier_logits.new_zeros(num_questions, dtype=torch.long)
        classifier_loss = F.cross_entropy(classifier_logits.masked_fill(~passage_mask, -1e4), labels, reduction="sum")

        return classifier_loss

    def _compute_span_loss(
        self,
        start_logits: Tensor,
        end_logits: Tensor,
        span_mask: Tensor,
        answer_starts: Tensor,
        answer_ends: Tensor,
        answer_mask: Tensor,
    ) -> Tensor:
        start_log_probs = F.log_softmax(start_logits.masked_fill(~span_mask, -1e4), dim=1)
        end_log_probs = F.log_softmax(end_logits.masked_fill(~span_mask, -1e4), dim=1)

        answer_start_log_probs = start_log_probs.take_along_dim(answer_starts, dim=1)
        answer_end_log_probs = end_log_probs.take_along_dim(answer_ends, dim=1)
        answer_span_log_probs = answer_start_log_probs + answer_end_log_probs

        span_losses = -answer_span_log_probs.masked_fill(~answer_mask, -1e4).logsumexp(dim=1)
        span_loss = span_losses.sum()
        return span_loss

    def _compute_classifier_accuracy(self, classifier_logits: Tensor, passage_mask: Tensor) -> Tensor:
        num_questions, _ = classifier_logits.size()

        selected_passage_idxs = classifier_logits.masked_fill(~passage_mask, -1e4).argmax(dim=1)
        labels = classifier_logits.new_zeros(num_questions, dtype=torch.long)
        accuracy = (selected_passage_idxs == labels).float().mean()

        return accuracy

    def _compute_span_accuracy(
        self,
        start_logits: Tensor,
        end_logits: Tensor,
        span_mask: Tensor,
        answer_starts: Tensor,
        answer_ends: Tensor,
        answer_mask: Tensor,
    ) -> Tensor:
        num_questions, _ = start_logits.size()
        _, max_answer_spans = answer_starts.size()

        pred_answer_starts, pred_answer_ends, _ = self.get_pred_answer_spans(start_logits, end_logits, span_mask)

        num_correct = 0

        for i in range(num_questions):
            pred_start = pred_answer_starts[i]
            pred_end = pred_answer_ends[i]

            for j in range(max_answer_spans):
                start = answer_starts[i, j]
                end = answer_ends[i, j]
                mask = answer_mask[i, j]

                if mask is False:
                    continue

                if (pred_start == start) and (pred_end == end):
                    num_correct += 1
                    break

        accuracy = num_correct / num_questions

        return torch.tensor(accuracy).to(start_logits.device)

    def get_pred_answer_spans(
        self, start_logits: Tensor, end_logits: Tensor, span_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        num_inputs, max_input_length = start_logits.size()

        start_log_probs = F.log_softmax(start_logits.masked_fill(~span_mask, -1e4), dim=1)
        end_log_probs = F.log_softmax(end_logits.masked_fill(~span_mask, -1e4), dim=1)

        span_log_probs = start_log_probs[:, :, None] + end_log_probs[:, None, :]
        span_matrix_mask = span_log_probs.new_ones((max_input_length, max_input_length)).bool()
        span_matrix_mask = span_matrix_mask.triu()
        span_matrix_mask = span_matrix_mask.tril(self.hparams.max_answer_length - 1)

        span_log_probs = span_log_probs.view(num_inputs, max_input_length * max_input_length)
        span_matrix_mask = span_matrix_mask.view(1, max_input_length * max_input_length)

        pred_span_log_probs, selected_span_idxs = span_log_probs.masked_fill(~span_matrix_mask, -1e4).max(dim=1)
        pred_answer_starts = selected_span_idxs // max_input_length
        pred_answer_ends = selected_span_idxs % max_input_length
        assert (pred_answer_starts <= pred_answer_ends).all()
        assert (pred_answer_ends - pred_answer_starts + 1 > 0).all()
        assert (pred_answer_ends - pred_answer_starts + 1 <= self.hparams.max_answer_length).all()

        return pred_answer_starts, pred_answer_ends, pred_span_log_probs

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        num_warmup_steps = int(self.hparams.warmup_ratio * self.trainer.estimated_stepping_batches)

        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


class ReaderPredictionLightningModule(LightningModule):
    def __init__(
        self,
        reader_ckpt_file: str,
        predict_dataset_file: str | None = None,
        predict_batch_size: int = 1,
        predict_max_passages: int = 100,
        datasets_num_proc: int | None = None,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.reader_module = ReaderLightningModule.load_from_checkpoint(
            self.hparams.reader_ckpt_file, map_location="cpu", strict=False
        )
        self.reader_module.freeze()

    def prepare_data(self):
        if self.hparams.predict_dataset_file is not None:
            self._load_dataset(self.hparams.predict_dataset_file)

    def setup(self, stage: str):
        if stage == "predict":
            self.predict_dataset = self._load_dataset(self.hparams.predict_dataset_file)

    def _load_dataset(self, dataset_file: str) -> Dataset:
        dataset = Dataset.from_json(dataset_file, num_proc=self.hparams.datasets_num_proc)
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

    def _collate_fn(self, examples: list[dict[str, Any]]) -> tuple[BatchEncoding, Tensor, Tensor]:
        questions: list[str] = []
        passage_titles: list[list[str]] = []
        passage_texts: list[list[str]] = []

        for example in examples:
            questions.append(example["question"])

            passages = example["passages"][: self.hparams.predict_max_passages]
            passage_titles.append([passage["title"] for passage in passages])
            passage_texts.append([passage["text"] for passage in passages])

        tokenized_inputs, passage_mask, span_mask = self.reader_module.tokenizer(
            questions, passage_titles, passage_texts
        )
        return tokenized_inputs, passage_mask, span_mask

    def predict_step(
        self, batch: tuple[BatchEncoding, Tensor, Tensor], batch_idx: int
    ) -> list[dict[str, str | float]]:
        tokenized_inputs, passage_mask, span_mask = batch
        predictions = self.predict_answers_from_tokenized_inputs(tokenized_inputs, passage_mask, span_mask)
        return predictions

    def predict_answers(
        self, questions: list[str], passage_titles: list[list[str]], passage_texts: list[list[str]]
    ) -> list[dict[str, str | float]]:
        tokenized_inputs, passage_mask, span_mask = self.reader_module.tokenizer(
            questions, passage_titles, passage_texts
        )
        tokenized_inputs = tokenized_inputs.to(self.device)
        passage_mask = passage_mask.to(self.device)
        span_mask = span_mask.to(self.device)

        predictions = self.predict_answers_from_tokenized_inputs(tokenized_inputs, passage_mask, span_mask)
        return predictions

    def predict_answers_from_tokenized_inputs(
        self, tokenized_inputs: BatchEncoding, passage_mask: Tensor, span_mask: Tensor
    ) -> list[dict[str, str | float]]:
        assert not self.reader_module.training

        input_ids = tokenized_inputs["input_ids"]
        num_questions, _, _ = input_ids.size()

        classifier_logits, start_logits, end_logits = self.reader_module.reader(tokenized_inputs)

        classifier_log_probs = F.log_softmax(classifier_logits.masked_fill(~passage_mask, -1e4), dim=1)
        pred_classifier_log_probs, selected_input_idxs = classifier_log_probs.max(dim=1)

        selected_input_ids = input_ids.take_along_dim(selected_input_idxs[:, None, None], dim=1)[:, 0, :]
        selected_span_mask = span_mask.take_along_dim(selected_input_idxs[:, None, None], dim=1)[:, 0, :]
        selected_start_logits = start_logits.take_along_dim(selected_input_idxs[:, None, None], dim=1)[:, 0, :]
        selected_end_logits = end_logits.take_along_dim(selected_input_idxs[:, None, None], dim=1)[:, 0, :]

        pred_answer_starts, pred_answer_ends, pred_span_log_probs = self.reader_module.get_pred_answer_spans(
            selected_start_logits, selected_end_logits, selected_span_mask
        )

        predictions: list[dict[str, str | float]] = []

        for i in range(num_questions):
            start = pred_answer_starts[i]
            end = pred_answer_ends[i]

            pred_answer = self.reader_module.tokenizer.decode(
                selected_input_ids[i].tolist(),
                selected_span_mask[i].tolist(),
                start=start,
                end=end,
                extend_subwords=True,
            )
            score = float(torch.exp(pred_classifier_log_probs[i] + pred_span_log_probs[i]))

            predictions.append({"pred_answer": pred_answer, "score": score})

        return predictions
