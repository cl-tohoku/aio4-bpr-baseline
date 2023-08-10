import torch
from typing import Any

from datasets import Dataset
from lightning import LightningModule
from torch.utils.data import DataLoader

from models.reader import Reader
from models.retriever import Retriever
from utils.data import DATASET_FEATURES, PASSAGE_DATASET_FEATURES, resize_batch_encoding


class BPRPipeline(LightningModule):
    def __init__(
        self,
        retriever_ckpt_file: str,
        reader_ckpt_file: str,
        passage_faiss_index_file: str,
        passage_dataset_file: str,
        predict_dataset_file: str | None = None,
        predict_batch_size: int = 2,
        predict_retriever_k: int = 100,
        predict_retriever_num_candidates: int = 1000,
        predict_reader_max_predictions: int = 1,
        predict_reader_max_passages_to_read: int = 1,
        predict_reader_max_answer_spans: int = 1,
        predict_reader_max_answer_length: int = 10,
        datasets_num_proc: int | None = None,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.retriever = Retriever.load_from_checkpoint(
            self.hparams.retriever_ckpt_file,
            map_location="cpu",
            strict=False,
            passage_faiss_index_file=self.hparams.passage_faiss_index_file,
        )
        self.reader = Reader.load_from_checkpoint(self.hparams.reader_ckpt_file, map_location="cpu", strict=False)

    def prepare_data(self):
        if self.hparams.passage_dataset_file is not None:
            self._load_passage_dataset(self.hparams.passage_dataset_file)
        if self.hparams.predict_dataset_file is not None:
            self._load_predict_dataset(self.hparams.predict_dataset_file)

    def setup(self, stage: str):
        self.passage_dataset = self._load_passage_dataset(self.hparams.passage_dataset_file)

        if stage == "predict":
            self.predict_dataset = self._load_predict_dataset(self.hparams.predict_dataset_file)

    def _load_passage_dataset(self, passage_dataset_file: str) -> Dataset:
        passage_dataset = Dataset.from_json(
            passage_dataset_file, features=PASSAGE_DATASET_FEATURES, num_proc=self.hparams.datasets_num_proc
        )
        return passage_dataset

    def _load_predict_dataset(self, dataset_file: str) -> Dataset:
        dataset = Dataset.from_json(dataset_file, features=DATASET_FEATURES, num_proc=self.hparams.datasets_num_proc)
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

    def _predict_collate_fn(self, examples: list[dict[str, Any]]) -> dict[str, list[str] | list[int]]:
        qids: list[str] = [example["qid"] for example in examples]
        positions: list[int] = [example["position"] for example in examples]
        questions: list[str] = [example["question"] for example in examples]
        return {"qids": qids, "positions": positions, "questions": questions}

    def predict_step(self, batch: dict[str, list[str]], batch_idx: int) -> list[dict[str, str | int | float]]:
        qids: list[str] = batch["qids"]
        positions: list[int] = batch["positions"]
        questions: list[str] = batch["questions"]

        predictions = self.predict_answers(
            questions,
            retriever_k=self.hparams.predict_retriever_k,
            retriever_num_candidates=self.hparams.predict_retriever_num_candidates,
            reader_max_predictions=self.hparams.predict_reader_max_predictions,
            reader_max_passages_to_read=self.hparams.predict_reader_max_passages_to_read,
            reader_max_answer_spans=self.hparams.predict_reader_max_answer_spans,
            reader_max_answer_length=self.hparams.predict_reader_max_answer_length,
        )

        outputs = [
            {"qid": qid, "position": position, "prediction": prediction["answers"][0]}
            for qid, position, prediction in zip(qids, positions, predictions)
        ]
        return outputs

    def predict_answers(
        self,
        questions: list[str],
        retriever_k: int = 100,
        retriever_num_candidates: int = 1000,
        reader_max_predictions: int = 1,
        reader_max_passages_to_read: int = 1,
        reader_max_answer_spans: int = 1,
        reader_max_answer_length: int = 10,
    ) -> list[list[dict[str, str | int | float]]]:
        if not hasattr(self, "passage_dataset"):
            self.passage_dataset = self._load_passage_dataset(self.hparams.passage_dataset_file)

        num_questions = len(questions)
        num_passages = retriever_k

        tokenized_questions = self.retriever.tokenizer.tokenize_questions(questions).to(self.device)
        max_question_length = tokenized_questions["input_ids"].size(1)
        assert tokenized_questions["input_ids"].size() == (num_questions, max_question_length)

        retriever_predictions = self.retriever.retrieve_topk_passages(
            tokenized_questions, k=retriever_k, num_candidates=retriever_num_candidates
        )

        repeated_questions: list[str] = []
        passage_titles: list[str] = []
        passage_texts: list[str] = []
        passage_mask: list[int] = []

        for question, retriever_prediction in zip(questions, retriever_predictions):
            passage_idxs = retriever_prediction["passage_idxs"]
            passages = [self.passage_dataset[passage_idx] for passage_idx in passage_idxs]
            dummy_passages = [{"title": "", "text": ""}] * (num_passages - len(passages))
            assert len(passages + dummy_passages) == num_passages

            repeated_questions.extend([question] * len(passages + dummy_passages))
            passage_titles.extend([passage["title"] for passage in passages + dummy_passages])
            passage_texts.extend([passage["text"] for passage in passages + dummy_passages])
            passage_mask.append([1] * len(passages) + [0] * len(dummy_passages))

        tokenized_inputs, span_mask = self.reader.tokenizer.tokenize_inputs(
            repeated_questions, passage_titles, passage_texts
        )
        tokenized_inputs = tokenized_inputs.to(self.device)
        span_mask = span_mask.to(self.device)
        max_input_length = tokenized_inputs["input_ids"].size(1)
        assert tokenized_inputs["input_ids"].size() == (num_questions * num_passages, max_input_length)
        assert span_mask.size() == (num_questions * num_passages, max_input_length)

        tokenized_inputs = resize_batch_encoding(tokenized_inputs, (num_questions, num_passages, -1))
        span_mask = span_mask.view(num_questions, num_passages, -1)
        assert tokenized_inputs["input_ids"].size() == (num_questions, num_passages, max_input_length)
        assert span_mask.size() == (num_questions, num_passages, max_input_length)

        passage_mask = torch.tensor(passage_mask).bool().to(self.device)
        assert passage_mask.size() == (num_questions, num_passages)

        reader_predictions = self.reader.get_best_predictions(
            tokenized_inputs,
            passage_mask,
            span_mask,
            max_predictions=reader_max_predictions,
            max_passages_to_read=reader_max_passages_to_read,
            max_answer_spans=reader_max_answer_spans,
            max_answer_length=reader_max_answer_length,
        )

        return reader_predictions
