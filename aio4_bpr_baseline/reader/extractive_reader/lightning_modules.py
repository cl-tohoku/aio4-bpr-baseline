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

from aio4_bpr_baseline.reader.extractive_reader.modeling import ExtractiveReaderModel
from aio4_bpr_baseline.reader.extractive_reader.tokenization import ExtractiveReaderTokenizer
from aio4_bpr_baseline.utils.data import DATASET_FEATURES, PASSAGES_FEATURES


class ExtractiveReaderLightningModule(LightningModule):
    def __init__(
        self,
        train_dataset_file: str,
        val_dataset_file: str,
        passages_file: str,
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

        self.tokenizer = ExtractiveReaderTokenizer(
            self.hparams.base_model_name, max_input_length=self.hparams.max_input_length
        )

        self.reader = ExtractiveReaderModel(self.hparams.base_model_name)

    def prepare_data(self):
        self._load_passages(self.hparams.passages_file)
        self._load_dataset(
            self.hparams.train_dataset_file, gold_passages_info_file=self.hparams.train_gold_passages_info_file
        )
        self._load_dataset(
            self.hparams.val_dataset_file, gold_passages_info_file=self.hparams.val_gold_passages_info_file
        )

    def setup(self, stage: str):
        self.all_passages = self._load_passages(self.hparams.passages_file)
        self.train_dataset = self._load_dataset(
            self.hparams.train_dataset_file, gold_passages_info_file=self.hparams.train_gold_passages_info_file
        )
        self.val_dataset = self._load_dataset(
            self.hparams.val_dataset_file, gold_passages_info_file=self.hparams.val_gold_passages_info_file
        )

    def _load_passages(self, passages_file: str) -> Dataset:
        return Dataset.from_json(passages_file, features=PASSAGES_FEATURES)

    def _load_dataset(self, dataset_file: str, gold_passages_info_file: str | None = None) -> Dataset:
        gold_passages = {}

        if gold_passages_info_file is not None:
            for gold_info in json.load(open(gold_passages_info_file))["data"]:
                gold_passage = {"title": gold_info["title"], "text": gold_info["context"]}

                gold_passages[gold_info["question"]] = gold_passage
                if "question_tokens" in gold_info:
                    gold_passages[gold_info["question_tokens"]] = gold_passage

        tokenizer = ExtractiveReaderTokenizer(
            self.hparams.base_model_name, max_input_length=self.hparams.max_input_length
        )
        max_answer_length = self.hparams.max_answer_length
        all_passages = self._load_passages(self.hparams.passages_file)

        def _map_example(example: dict[str, Any]) -> dict[str, Any]:
            question = example["question"]
            answers = example["answers"]
            positive_passages = example["positive_passages"]

            def _filter_passages(passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
                num_passages = len(passages)
                if num_passages == 0:
                    return []

                questions = [question] * num_passages
                passage_titles = [all_passages[passage["idx"]]["title"] for passage in passages]
                passage_texts = [all_passages[passage["idx"]]["text"] for passage in passages]
                tokenized_inputs, span_mask = tokenizer(questions, passage_titles, passage_texts, return_tensors="np")
                input_ids = tokenized_inputs["input_ids"]

                filtered_passages = []
                for i, passage in enumerate(passages):
                    for answer in answers:
                        answer_spans = tokenizer.find_answer_spans(
                            answer,
                            input_ids[i, :].tolist(),
                            span_mask=span_mask[i, :].tolist(),
                            max_answer_length=max_answer_length,
                        )
                        if len(answer_spans) > 0:
                            filtered_passages.append(passage)
                            break

                return filtered_passages

            filtered_positive_passages = []
            if question in gold_passages:
                gold_passage_title = gold_passages[question]["title"]
                gold_positive_passages = [
                    passage for passage in positive_passages if passage["title"].lower() == gold_passage_title.lower()
                ]
                filtered_positive_passages = _filter_passages(gold_positive_passages)

            if len(filtered_positive_passages) == 0:
                filtered_positive_passages = _filter_passages(positive_passages)

            example["positive_passages"] = filtered_positive_passages
            return example

        def _filter_example(example: dict[str, Any]) -> bool:
            if len(example["positive_passages"]) == 0:
                return False

            return True

        dataset = Dataset.from_json(dataset_file, features=DATASET_FEATURES)
        dataset = dataset.map(_map_example, num_proc=self.hparams.datasets_num_proc)
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
        self, examples: list[dict[str, Any]], stage: str = "fit",
    ) -> tuple[BatchEncoding, Tensor, Tensor, Tensor, Tensor, Tensor]:
        num_questions = len(examples)
        max_passages = 1 + self.hparams.max_negative_passages
        max_answer_spans = self.hparams.max_answer_spans

        questions = []
        passage_titles = []
        passage_texts = []
        passage_mask = []

        for example in examples:
            questions.extend([example["question"]] * max_passages)

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

        tokenized_inputs, span_mask = self.tokenizer(questions, passage_titles, passage_texts)
        tokenized_inputs = BatchEncoding(
            {k: v.view(num_questions, max_passages, -1) for k, v in tokenized_inputs.items()}
        )
        span_mask = span_mask.view(num_questions, max_passages, -1)
        passage_mask = torch.tensor(passage_mask).view(num_questions, max_passages).bool()

        positive_input_ids = tokenized_inputs["input_ids"][:, 0, :]
        positive_span_mask = span_mask[:, 0, :]

        answer_starts = []
        answer_ends = []
        answer_mask = []

        for i, example in enumerate(examples):
            answers = example["answers"]

            answer_spans = []
            for answer in answers:
                answer_spans.extend(self.tokenizer.find_answer_spans(
                    answer,
                    positive_input_ids[i].tolist(),
                    span_mask=positive_span_mask[i].tolist(),
                    max_answer_length=self.hparams.max_answer_length,
                ))

            answer_spans = answer_spans[: max_answer_spans]

            num_answer_spans = len(answer_spans)
            assert num_answer_spans > 0
            answer_starts.extend([span[0] for span in answer_spans])
            answer_ends.extend([span[1] for span in answer_spans])
            answer_mask.extend([1] * num_answer_spans)

            num_dummy_answer_spans = max_answer_spans - num_answer_spans
            answer_starts.extend([0] * num_dummy_answer_spans)
            answer_ends.extend([0] * num_dummy_answer_spans)
            answer_mask.extend([0] * num_dummy_answer_spans)

        answer_starts = torch.tensor(answer_starts).view(num_questions, max_answer_spans)
        answer_ends = torch.tensor(answer_ends).view(num_questions, max_answer_spans)
        answer_mask = torch.tensor(answer_mask).view(num_questions, max_answer_spans).bool()

        return tokenized_inputs, passage_mask, span_mask, answer_starts, answer_ends, answer_mask

    def training_step(
        self, batch: tuple[BatchEncoding, Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        tokenized_inputs, passage_mask, span_mask, answer_starts, answer_ends, answer_mask = batch
        num_questions, max_passages, _ = tokenized_inputs["input_ids"].size()

        classifier_logits, start_logits, end_logits = self.reader(
            BatchEncoding({k: v.view(num_questions * max_passages, -1) for k, v in tokenized_inputs.items()})
        )
        classifier_logits = classifier_logits.view(num_questions, max_passages)
        start_logits = start_logits.view(num_questions, max_passages, -1)
        end_logits = end_logits.view(num_questions, max_passages, -1)

        positive_start_logits = start_logits[:, 0, :]
        positive_end_logits = end_logits[:, 0, :]
        positive_span_mask = span_mask[:, 0, :]

        classifier_loss = self._compute_classifier_loss(classifier_logits, passage_mask)
        self.log("train_classifier_loss", classifier_loss)

        span_loss = self._compute_span_loss(
            positive_start_logits, positive_end_logits, positive_span_mask, answer_starts, answer_ends, answer_mask
        )
        self.log("train_span_loss", span_loss)

        loss = classifier_loss + span_loss
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: tuple[BatchEncoding, Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int):
        tokenized_inputs, passage_mask, span_mask, answer_starts, answer_ends, answer_mask = batch
        num_questions, max_passages, _ = tokenized_inputs["input_ids"].size()

        classifier_logits, start_logits, end_logits = self.reader(
            BatchEncoding({k: v.view(num_questions * max_passages, -1) for k, v in tokenized_inputs.items()})
        )
        classifier_logits = classifier_logits.view(num_questions, max_passages)
        start_logits = start_logits.view(num_questions, max_passages, -1)
        end_logits = end_logits.view(num_questions, max_passages, -1)

        positive_start_logits = start_logits[:, 0, :]
        positive_end_logits = end_logits[:, 0, :]
        positive_span_mask = span_mask[:, 0, :]

        classifier_loss = self._compute_classifier_loss(classifier_logits, passage_mask)
        self.log("val_classifier_loss", classifier_loss, sync_dist=True)

        span_loss = self._compute_span_loss(
            positive_start_logits, positive_end_logits, positive_span_mask, answer_starts, answer_ends, answer_mask
        )
        self.log("val_span_loss", span_loss, sync_dist=True)

        loss = classifier_loss + span_loss
        self.log("val_loss", loss, sync_dist=True)

        classifier_accuracy = self._compute_classifier_accuracy(classifier_logits, passage_mask)
        self.log("val_classifier_accuracy", classifier_accuracy)

        span_accuracy = self._compute_span_accuracy(
            positive_start_logits, positive_end_logits, positive_span_mask, answer_starts, answer_ends, answer_mask
        )
        self.log("val_span_accuracy", span_accuracy, sync_dist=True)

        joint_accuracy = classifier_accuracy * span_accuracy
        self.log("val_joint_accuracy", joint_accuracy, sync_dist=True)

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


class ExtractiveReaderPredictLightningModule(LightningModule):
    def __init__(
        self,
        reader_ckpt_file: str,
        predict_dataset_file: str | None = None,
        passages_file: str | None = None,
        predict_batch_size: int = 1,
        predict_max_passages: int = 100,
        datasets_num_proc: int | None = None,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.reader_module = ExtractiveReaderLightningModule.load_from_checkpoint(
            self.hparams.reader_ckpt_file, map_location="cpu", strict=False
        )
        self.reader_module.freeze()

    def prepare_data(self):
        self._load_passages(self.hparams.passages_file)
        self._load_dataset(self.hparams.predict_dataset_file)

    def setup(self, stage: str):
        self.all_passages = self._load_passages(self.hparams.passages_file)
        self.predict_dataset = self._load_dataset(self.hparams.predict_dataset_file)

    def _load_passages(self, passages_file: str) -> Dataset:
        return Dataset.from_json(passages_file, features=PASSAGES_FEATURES)

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

    def _collate_fn(self, examples: list[dict[str, Any]]) -> tuple[BatchEncoding, Tensor, Tensor]:
        num_questions = len(examples)
        max_passages = self.hparams.predict_max_passages

        questions = []
        passage_titles = []
        passage_texts = []
        passage_mask = []

        for example in examples:
            questions.extend([example["question"]] * max_passages)

            passages = example["positive_passages"] + example["negative_passages"]
            passages = sorted(passages, key=lambda x: x["score"], reverse=True)[:max_passages]

            num_passages = len(passages)
            passage_titles.extend([self.all_passages[passage["idx"]]["title"] for passage in passages])
            passage_texts.extend([self.all_passages[passage["idx"]]["text"] for passage in passages])
            passage_mask.extend([1] * num_passages)

            num_dummy_passages = max_passages - num_passages
            passage_titles.extend([""] * num_dummy_passages)
            passage_texts.extend([""] * num_dummy_passages)
            passage_mask.extend([0] * num_dummy_passages)

        tokenized_inputs, span_mask = self.reader_module.tokenizer(questions, passage_titles, passage_texts)
        tokenized_inputs = BatchEncoding(
            {k: v.view(num_questions, max_passages, -1) for k, v in tokenized_inputs.items()}
        )
        span_mask = span_mask.view(num_questions, max_passages, -1)
        passage_mask = torch.tensor(passage_mask).view(num_questions, max_passages).bool()

        return tokenized_inputs, passage_mask, span_mask

    def predict_step(
        self, batch: tuple[BatchEncoding, Tensor, Tensor], batch_idx: int
    ) -> list[dict[str, str | float]]:
        tokenized_inputs, passage_mask, span_mask = batch
        predictions = self.predict_answers_tokenized(tokenized_inputs, passage_mask, span_mask)
        return predictions

    def predict_answer(
        self,
        question: str,
        passage_titles: list[str],
        passage_texts: list[str],
    ) -> dict[str, str | float]:
        num_passages = len(passage_titles)
        if len(passage_texts) != num_passages:
            raise ValueError(
                f"len(passage_titles) != len(passage_texts) ({len(passage_titles)} != {len(passage_texts)}"
            )

        questions = [question] * num_passages

        tokenized_inputs, span_mask = self.reader_module.tokenizer(questions, passage_titles, passage_texts)
        tokenized_inputs = BatchEncoding({k: v.view(1, num_passages, -1) for k, v in tokenized_inputs.items()})
        span_mask = span_mask.view(1, num_passages, -1)
        passage_mask = torch.ones(1, num_passages).bool()

        tokenized_inputs = tokenized_inputs.to(self.device)
        span_mask = span_mask.to(self.device)
        passage_mask = passage_mask.to(self.device)

        predictions = self.predict_answers_tokenized(tokenized_inputs, passage_mask, span_mask)
        assert len(predictions) == 1
        return predictions[0]

    def predict_answers_tokenized(
        self,
        tokenized_inputs: BatchEncoding,
        passage_mask: Tensor,
        span_mask: Tensor,
    ) -> list[dict[str, str | float]]:
        assert not self.reader_module.reader.training

        input_ids = tokenized_inputs["input_ids"]
        num_questions, max_passages, _ = input_ids.size()

        classifier_logits, start_logits, end_logits = self.reader_module.reader(
            BatchEncoding({k: v.view(num_questions * max_passages, -1) for k, v in tokenized_inputs.items()})
        )
        classifier_logits = classifier_logits.view(num_questions, max_passages)
        start_logits = start_logits.view(num_questions, max_passages, -1)
        end_logits = end_logits.view(num_questions, max_passages, -1)

        classifier_log_probs = F.log_softmax(classifier_logits.masked_fill(~passage_mask, -1e4), dim=1)
        pred_classifier_log_probs, selected_input_idxs = classifier_log_probs.max(dim=1)

        selected_input_ids = input_ids.take_along_dim(selected_input_idxs[:, None, None], dim=1).squeeze(1)
        selected_span_mask = span_mask.take_along_dim(selected_input_idxs[:, None, None], dim=1).squeeze(1)
        selected_start_logits = start_logits.take_along_dim(selected_input_idxs[:, None, None], dim=1).squeeze(1)
        selected_end_logits = end_logits.take_along_dim(selected_input_idxs[:, None, None], dim=1).squeeze(1)

        pred_answer_starts, pred_answer_ends, pred_span_log_probs = self.reader_module.get_pred_answer_spans(
            selected_start_logits, selected_end_logits, selected_span_mask
        )

        predictions = []
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
