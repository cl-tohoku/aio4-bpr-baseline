import json
import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BatchEncoding, PreTrainedModel, PreTrainedTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from utils.data import DATASET_FEATURES, find_spans, resize_batch_encoding


class ReaderTokenizer:
    def __init__(self, base_model_name: str, max_input_length: int = 350) -> None:
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.max_input_length = max_input_length

    def tokenize_inputs(
        self,
        questions: list[str],
        passage_titles: list[str],
        passage_texts: list[str],
        return_numpy_arrays: bool = False,
    ) -> tuple[BatchEncoding, Tensor]:
        assert len(questions) == len(passage_titles) == len(passage_texts)

        passages = [title + self.tokenizer.sep_token + text for title, text in zip(passage_titles, passage_texts)]

        if return_numpy_arrays:
            return_tensors = "np"
        else:
            return_tensors = "pt"

        tokenized_inputs = self.tokenizer(
            questions,
            passages,
            padding=True,
            truncation="only_second",
            max_length=self.max_input_length,
            return_tensors=return_tensors,
        )

        tokenized_short_inputs = self.tokenizer(
            questions,
            passage_titles,
            padding="max_length",
            truncation="only_second",
            max_length=len(tokenized_inputs["input_ids"][0]),
            return_tensors=return_tensors,
        )
        span_mask_1 = tokenized_inputs["input_ids"] != self.tokenizer.pad_token_id
        span_mask_2 = tokenized_short_inputs["input_ids"] == self.tokenizer.pad_token_id
        span_mask = span_mask_1 * span_mask_2

        return tokenized_inputs, span_mask

    def find_answer_spans(self, token_ids: list[int], mask: list[int], answers: list[str]):
        answers_token_ids = self.tokenizer(answers, add_special_tokens=False)["input_ids"]
        answer_spans = find_spans(token_ids, answers_token_ids, mask=mask)
        return answer_spans

    def is_subword_id(self, token_id: int) -> bool:
        return self.tokenizer.convert_ids_to_tokens([token_id])[0].startswith("##")

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)


class Reader(LightningModule):
    def __init__(
        self,
        train_dataset_file: str | None = None,
        val_dataset_file: str | None = None,
        predict_dataset_file: str | None = None,
        train_gold_passages_info_file: str | None = None,
        val_gold_passages_info_file: str | None = None,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        predict_batch_size: int = 1,
        base_model_name: str = "bert-base-uncased",
        max_input_length: int = 350,
        max_negative_passages: int = 23,
        max_answer_spans: int = 10,
        shuffle_positive_passages: bool = True,
        shuffle_negative_passages: bool = True,
        predict_max_passages: int = 100,
        predict_max_predictions: int = 1,
        predict_max_passages_to_read: int = 1,
        predict_max_answer_spans: int = 1,
        predict_max_answer_length: int = 10,
        warmup_ratio: float = 0.06,
        lr: float = 2e-5,
        datasets_num_proc: int | None = None,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder: PreTrainedModel = AutoModel.from_pretrained(self.hparams.base_model_name, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(self.encoder.config.hidden_size, 2)
        self.qa_classifier = nn.Linear(self.encoder.config.hidden_size, 1)

        self.tokenizer = ReaderTokenizer(
            self.hparams.base_model_name, max_input_length=self.hparams.max_input_length
        )

    def prepare_data(self):
        if self.hparams.train_dataset_file is not None:
            self._load_fit_dataset(
                self.hparams.train_dataset_file, gold_passages_info_file=self.hparams.train_gold_passages_info_file
            )
        if self.hparams.val_dataset_file is not None:
            self._load_fit_dataset(
                self.hparams.val_dataset_file, gold_passages_info_file=self.hparams.val_gold_passages_info_file
            )
        if self.hparams.predict_dataset_file is not None:
            self._load_predict_dataset(self.hparams.predict_dataset_file)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self._load_fit_dataset(
                self.hparams.train_dataset_file, gold_passages_info_file=self.hparams.train_gold_passages_info_file
            )
            self.val_dataset = self._load_fit_dataset(
                self.hparams.val_dataset_file, gold_passages_info_file=self.hparams.val_gold_passages_info_file
            )
        elif stage == "predict":
            self.predict_dataset = self._load_predict_dataset(self.hparams.predict_dataset_file)

    def _load_fit_dataset(self, dataset_file: str, gold_passages_info_file: str | None = None) -> Dataset:
        gold_passages_info: dict[str, dict[str, str]] = {}

        if gold_passages_info_file is not None:
            for item in json.load(open(gold_passages_info_file))["data"]:
                question = item["question"]
                question_tokens = item.get("question_tokens")

                passage = {"title": item["title"], "text": item["context"]}

                gold_passages_info[question] = passage
                if question_tokens is not None:
                    gold_passages_info[question_tokens] = passage

        tokenizer = ReaderTokenizer(
            self.hparams.base_model_name, max_input_length=self.hparams.max_input_length
        )

        def _preprocess_example(example: dict[str, Any]) -> dict[str, BatchEncoding | Tensor | list[int]]:
            def _filter_passage_idxs(passage_idxs: list[int]) -> list[int]:
                if len(passage_idxs) == 0:
                    return []

                passages = [example["passages"][idx] for idx in passage_idxs]

                questions = [example["question"]] * len(passages)
                passage_titles = [passage["title"] for passage in passages]
                passage_texts = [passage["text"] for passage in passages]

                tokenized_inputs, span_mask = tokenizer.tokenize_inputs(
                    questions, passage_titles, passage_texts, return_numpy_arrays=True
                )
                input_token_ids = tokenized_inputs["input_ids"]

                filtered_passage_idxs: list[int] = []

                for idx, token_ids, mask in zip(passage_idxs, input_token_ids.tolist(), span_mask.tolist()):
                    answer_spans = tokenizer.find_answer_spans(token_ids, mask, example["answers"])
                    if len(answer_spans) > 0:
                        filtered_passage_idxs.append(idx)

                return filtered_passage_idxs

            filtered_positive_passage_idxs: list[int] = []

            if example["question"] in gold_passages_info:
                gold_passage_title = gold_passages_info[example["question"]]["title"]

                gold_positive_passage_idxs = [
                    idx for idx in example["positive_passage_idxs"]
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

    def _load_predict_dataset(self, dataset_file: str) -> Dataset:
        dataset = Dataset.from_json(dataset_file, features=DATASET_FEATURES, num_proc=self.hparams.datasets_num_proc)
        return dataset

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

    def _fit_collate_fn(
        self, examples: list[dict[str, Any]]
    ) -> tuple[BatchEncoding, Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
        num_questions = len(examples)
        num_passages = self.hparams.max_negative_passages + 1
        num_answer_spans = self.hparams.max_answer_spans

        questions: list[str] = []
        passage_titles: list[str] = []
        passage_texts: list[str] = []
        passage_mask: list[list[int]] = []

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

            questions.extend([example["question"]] * len(passages + dummy_passages))
            passage_titles.extend([passage["title"] for passage in passages + dummy_passages])
            passage_texts.extend([passage["text"] for passage in passages + dummy_passages])
            passage_mask.append([1] * len(passages) + [0] * len(dummy_passages))

        tokenized_inputs, span_mask = self.tokenizer.tokenize_inputs(questions, passage_titles, passage_texts)
        tokenized_inputs = resize_batch_encoding(tokenized_inputs, (num_questions, num_passages, -1))
        span_mask = span_mask.view(num_questions, num_passages, -1)
        passage_mask = torch.tensor(passage_mask).bool()

        positive_token_ids = tokenized_inputs["input_ids"][:, 0]
        positive_span_mask = span_mask[:, 0]

        answer_start_positions: list[list[int]] = []
        answer_end_positions: list[list[int]]  = []
        answer_mask: list[list[int]]  = []

        for example, token_ids, mask, in zip(examples, positive_token_ids.tolist(), positive_span_mask.tolist()):
            answers = example["answers"]

            answer_spans = self.tokenizer.find_answer_spans(token_ids, mask, answers)[:num_answer_spans]
            assert len(answer_spans) > 0
            dummy_answer_spans = [(0, 0)] * (num_answer_spans - len(answer_spans))
            assert len(answer_spans + dummy_answer_spans) == num_answer_spans

            answer_start_positions.append([span[0] for span in answer_spans + dummy_answer_spans])
            answer_end_positions.append([span[1] for span in answer_spans + dummy_answer_spans])
            answer_mask.append([1] * len(answer_spans) + [0] * len(dummy_answer_spans))

        answer_start_positions = torch.tensor(answer_start_positions)
        answer_end_positions = torch.tensor(answer_end_positions)
        answer_mask = torch.tensor(answer_mask).bool()

        return tokenized_inputs, passage_mask, span_mask, answer_start_positions, answer_end_positions, answer_mask

    def _predict_collate_fn(self, examples: list[dict[str, Any]]) -> tuple[BatchEncoding, Tensor, Tensor]:
        num_questions = len(examples)
        num_passages = self.hparams.predict_max_passages

        questions: list[str] = []
        passage_titles: list[str] = []
        passage_texts: list[str] = []
        passage_mask: list[list[int]] = []

        for example in examples:
            passages = example["passages"][:num_passages]
            dummy_passages = [{"title": "", "text": ""}] * (num_passages - len(passages))
            assert len(passages + dummy_passages) == num_passages

            questions.extend([example["question"]] * len(passages + dummy_passages))
            passage_titles.extend([passage["title"] for passage in passages + dummy_passages])
            passage_texts.extend([passage["text"] for passage in passages + dummy_passages])
            passage_mask.append([1] * len(passages) + [0] * len(dummy_passages))

        tokenized_inputs, span_mask = self.tokenizer.tokenize_inputs(questions, passage_titles, passage_texts)
        tokenized_inputs = resize_batch_encoding(tokenized_inputs, (num_questions, num_passages, -1))
        span_mask = span_mask.view(num_questions, num_passages, -1)
        passage_mask = torch.tensor(passage_mask).bool()

        return tokenized_inputs, passage_mask, span_mask

    def forward(self, tokenized_inputs: BatchEncoding) -> tuple[Tensor, Tensor, Tensor]:
        num_questions, num_passages, max_input_length = tokenized_inputs["input_ids"].size()

        tokenized_inputs = resize_batch_encoding(tokenized_inputs, (num_questions * num_passages, max_input_length))
        encoded_inputs = self.encoder(**tokenized_inputs).last_hidden_state

        classifier_logits = self.qa_classifier(encoded_inputs[:, 0]).view(num_questions, num_passages)

        start_end_logits = self.qa_outputs(encoded_inputs)
        start_logits = start_end_logits[:, :, 0].view(num_questions, num_passages, max_input_length)
        end_logits = start_end_logits[:, :, 1].view(num_questions, num_passages, max_input_length)

        return classifier_logits, start_logits, end_logits

    def training_step(
        self, batch: tuple[BatchEncoding, Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        tokenized_inputs, passage_mask, span_mask, answer_start_positions, answer_end_positions, answer_mask = batch

        classifier_logits, start_logits, end_logits = self.forward(tokenized_inputs)

        classifier_loss = self._compute_classifier_loss(classifier_logits, passage_mask)
        span_loss = self._compute_span_loss(
            start_logits, end_logits, span_mask, answer_start_positions, answer_end_positions, answer_mask
        )
        loss = classifier_loss + span_loss

        metrics = {"train_loss": loss, "train_classifier_loss": classifier_loss, "train_span_loss": span_loss}
        self.log_dict(metrics)

        return loss

    def validation_step(
        self, batch: tuple[BatchEncoding, Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ):
        tokenized_inputs, passage_mask, span_mask, answer_start_positions, answer_end_positions, answer_mask = batch

        classifier_logits, start_logits, end_logits = self.forward(tokenized_inputs)

        classifier_loss = self._compute_classifier_loss(classifier_logits, passage_mask)
        span_loss = self._compute_span_loss(
            start_logits, end_logits, span_mask, answer_start_positions, answer_end_positions, answer_mask
        )
        loss = classifier_loss + span_loss

        classifier_accuracy = self._compute_classifier_accuracy(classifier_logits, passage_mask)
        span_accuracy = self._compute_span_accuracy(
            start_logits, end_logits, span_mask, answer_start_positions, answer_end_positions, answer_mask
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

    def predict_step(
        self, batch: tuple[BatchEncoding, Tensor, Tensor], batch_idx: int
    ) -> list[dict[str, list[str] | list[float]]]:
        tokenized_inputs, passage_mask, span_mask = batch

        batch_best_predictions = self.get_best_predictions(
            tokenized_inputs,
            passage_mask,
            span_mask,
            max_predictions=self.hparams.predict_max_predictions,
            max_passages_to_read=self.hparams.predict_max_passages_to_read,
            max_answer_spans=self.hparams.predict_max_answer_spans,
            max_answer_length=self.hparams.predict_max_answer_length,
        )
        return batch_best_predictions

    def _compute_classifier_loss(self, classifier_logits: Tensor, passage_mask: Tensor) -> Tensor:
        num_questions = classifier_logits.size(0)

        label = classifier_logits.new_zeros(num_questions, dtype=torch.long)
        classifier_loss = F.cross_entropy(classifier_logits.masked_fill(~passage_mask, -1e4), label, reduction="sum")

        return classifier_loss

    def _compute_span_loss(
        self,
        start_logits: Tensor,
        end_logits: Tensor,
        span_mask: Tensor,
        answer_start_positions: Tensor,
        answer_end_positions: Tensor,
        answer_mask: Tensor,
    ) -> Tensor:
        start_logits = start_logits[:, 0, :]
        end_logits = end_logits[:, 0, :]
        span_mask = span_mask[:, 0, :]

        start_log_probs = F.log_softmax(start_logits.masked_fill(~span_mask, -1e4), dim=1)
        end_log_probs = F.log_softmax(end_logits.masked_fill(~span_mask, -1e4), dim=1)

        answer_start_log_probs = start_log_probs.gather(dim=1, index=answer_start_positions)
        answer_end_log_probs = end_log_probs.gather(dim=1, index=answer_end_positions)
        answer_span_log_probs = answer_start_log_probs + answer_end_log_probs

        span_losses = -answer_span_log_probs.masked_fill(~answer_mask, -1e4).logsumexp(dim=1)
        span_loss = span_losses.sum()

        return span_loss

    def _compute_classifier_accuracy(self, classifier_logits: Tensor, passage_mask: Tensor) -> Tensor:
        num_questions = classifier_logits.size(0)

        label = classifier_logits.new_zeros(num_questions, dtype=torch.long)
        accuracy = (classifier_logits.masked_fill(~passage_mask, -1e4).argmax(dim=1) == label).float().mean()

        return accuracy

    def _compute_span_accuracy(
        self,
        start_logits: Tensor,
        end_logits: Tensor,
        span_mask: Tensor,
        answer_start_positions: Tensor,
        answer_end_positions: Tensor,
        answer_mask: Tensor,
    ) -> Tensor:
        num_questions = start_logits.size(0)

        start_logits = start_logits[:, 0, :].unsqueeze(1)
        end_logits = end_logits[:, 0, :].unsqueeze(1)
        span_mask = span_mask[:, 0, :].unsqueeze(1)

        pred_answer_start_positions, pred_answer_end_positions, pred_answer_mask, pred_answer_span_scores = (
            self._predict_answer_spans(start_logits, end_logits, span_mask)
        )

        num_correct = 0

        for i in range(num_questions):
            pred_start: int = pred_answer_start_positions[i, 0, 0].item()
            pred_end: int = pred_answer_end_positions[i, 0, 0].item()

            correct_spans = [
                (start, end) for start, end, mask
                in zip(answer_start_positions[i].tolist(), answer_end_positions[i].tolist(), answer_mask[i].tolist())
                if mask is True
            ]

            if (pred_start, pred_end) in correct_spans:
                num_correct += 1

        accuracy = num_correct / num_questions

        return torch.tensor(accuracy).to(start_logits.device)

    def _predict_answer_spans(
        self,
        start_logits: Tensor,
        end_logits: Tensor,
        span_mask: Tensor,
        max_answer_spans: int = 1,
        max_answer_length: int = 10,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        num_questions, num_passages, max_input_length = start_logits.size()

        start_log_probs = F.log_softmax(start_logits.masked_fill(~span_mask, -1e4), dim=2)
        end_log_probs = F.log_softmax(end_logits.masked_fill(~span_mask, -1e4), dim=2)

        span_log_probs = start_log_probs.unsqueeze(3) + end_log_probs.unsqueeze(2)

        span_log_probs = span_log_probs.view(num_questions, num_passages, max_input_length * max_input_length)
        sorted_span_log_probs, sorted_span_positions = span_log_probs.sort(dim=2, descending=True)

        pred_answer_start_positions = start_logits.new_zeros((num_questions, num_passages, max_answer_spans)).long()
        pred_answer_end_positions = start_logits.new_zeros((num_questions, num_passages, max_answer_spans)).long()
        pred_answer_mask = start_logits.new_zeros((num_questions, num_passages, max_answer_spans)).bool()
        pred_answer_span_scores = start_logits.new_zeros((num_questions, num_passages, max_answer_spans))

        for i in range(num_questions):
            for j in range(num_passages):
                mask: list[bool] = span_mask[i, j, :].tolist()
                log_probs: list[float] = sorted_span_log_probs[i, j, :].tolist()
                positions: list[int] = sorted_span_positions[i, j, :].tolist()

                k = 0
                for log_prob, position in zip(log_probs, positions):
                    start = position // max_input_length
                    end = position % max_input_length

                    if not (0 <= end - start < max_answer_length):
                        continue

                    if not all(mask[start : end + 1]):
                        continue

                    prev_starts = pred_answer_start_positions[i, j, :k].tolist()
                    prev_ends = pred_answer_end_positions[i, j, :k].tolist()

                    if any([
                        (start <= prev_start <= prev_end <= end) or (prev_start <= start <= end <= prev_end)
                        for prev_start, prev_end in zip(prev_starts, prev_ends)
                    ]):
                        continue

                    pred_answer_start_positions[i, j, k] = start
                    pred_answer_end_positions[i, j, k] = end
                    pred_answer_mask[i, j, k] = True
                    pred_answer_span_scores[i, j, k] = log_prob

                    k += 1
                    if k >= max_answer_spans:
                        break

        return pred_answer_start_positions, pred_answer_end_positions, pred_answer_mask, pred_answer_span_scores

    def get_best_predictions(
        self,
        tokenized_inputs: BatchEncoding,
        passage_mask: Tensor,
        span_mask: Tensor,
        max_predictions: int = 1,
        max_passages_to_read: int = 1,
        max_answer_spans: int = 1,
        max_answer_length: int = 10,
    ) -> list[list[tuple[str, float]]]:
        token_ids = tokenized_inputs["input_ids"]
        num_questions, num_passages, max_input_length = token_ids.size()

        classifier_logits, start_logits, end_logits = self.forward(tokenized_inputs)

        classifier_log_probs = F.log_softmax(classifier_logits.masked_fill(~passage_mask, -1e4), dim=1)

        top_classifier_log_probs, top_idxs = classifier_log_probs.topk(max_passages_to_read, dim=1)
        num_top_passages = top_classifier_log_probs.size(1)

        top_token_ids = token_ids.take_along_dim(top_idxs[:, :, None], dim=1)
        top_span_mask = span_mask.take_along_dim(top_idxs[:, :, None], dim=1)
        top_start_logits = start_logits.take_along_dim(top_idxs[:, :, None], dim=1)
        top_end_logits = end_logits.take_along_dim(top_idxs[:, :, None], dim=1)

        pred_answer_start_positions, pred_answer_end_positions, pred_answer_mask, pred_answer_span_scores = (
            self._predict_answer_spans(
                top_start_logits,
                top_end_logits,
                top_span_mask,
                max_answer_spans=max_answer_spans,
                max_answer_length=max_answer_length,
            )
        )

        def is_subword_id(token_id: int):
            return self.tokenizer.is_subword_id(token_id)

        batch_best_predictions: list[list[tuple[str, float]]] = []

        for i in range(num_questions):
            answer_probs: dict[str, float] = {}

            for j in range(num_top_passages):
                token_ids: list[int] = top_token_ids[i, j, :].tolist()
                span_mask: list[bool] = top_span_mask[i, j, :].tolist()
                classifier_log_prob = top_classifier_log_probs[i, j]

                for k in range(max_answer_spans):
                    start: int = pred_answer_start_positions[i, j, k].item()
                    end: int = pred_answer_end_positions[i, j, k].item()
                    answer_mask: bool = pred_answer_mask[i, j, k].item()
                    span_log_prob = pred_answer_span_scores[i, j, k]

                    if not answer_mask:
                        continue

                    while start - 1 >= 0 and span_mask[start - 1] and is_subword_id(token_ids[start]):
                        start -= 1
                    while end + 1 < max_input_length and span_mask[end + 1] and is_subword_id(token_ids[end + 1]):
                        end += 1

                    answer_text = self.tokenizer.decode(token_ids[start : end + 1])
                    answer_prob: float = torch.exp(classifier_log_prob + span_log_prob).item()

                    if answer_text in answer_probs:
                        answer_probs[answer_text] += answer_prob
                    else:
                        answer_probs[answer_text] = answer_prob

            best_predictions = sorted(list(answer_probs.items()), key=lambda x: x[1], reverse=True)[:max_predictions]
            answer_texts = [prediction[0] for prediction in best_predictions]
            answer_probs = [prediction[1] for prediction in best_predictions]

            batch_best_predictions.append({"answers": answer_texts, "scores": answer_probs})

        return batch_best_predictions

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        num_warmup_steps = int(self.hparams.warmup_ratio * self.trainer.estimated_stepping_batches)

        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
