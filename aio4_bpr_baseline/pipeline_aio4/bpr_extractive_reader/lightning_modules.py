from typing import Any

from datasets import Dataset
from lightning import LightningModule
from torch.utils.data import DataLoader

from aio4_bpr_baseline.reader.extractive_reader.lightning_modules import ExtractiveReaderPredictLightningModule
from aio4_bpr_baseline.retriever.bpr.lightning_modules import BPRRetrieverLightningModule
from aio4_bpr_baseline.utils.data import PASSAGES_FEATURES


class BPRExtractiveReaderPipelineLightningModule(LightningModule):
    def __init__(
        self,
        biencoder_ckpt_file: str,
        reader_ckpt_file: str,
        passage_embeddings_file: str,
        passages_file: str,
        predict_dataset_file: str | None = None,
        predict_batch_size: int = 2,
        predict_num_passages: int = 100,
        predict_num_candidates: int = 1000,
        predict_answer_score_threshold: float = 0.0,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.retriever_module = BPRRetrieverLightningModule(
            self.hparams.biencoder_ckpt_file, self.hparams.passage_embeddings_file
        )
        self.reader_predict_module = ExtractiveReaderPredictLightningModule(self.hparams.reader_ckpt_file)

    def prepare_data(self):
        self._load_passages(self.hparams.passages_file)
        self._load_dataset(self.hparams.predict_dataset_file)

    def setup(self, stage: str):
        self.all_passages = self._load_passages(self.hparams.passages_file)
        self.predict_dataset = self._load_dataset(self.hparams.predict_dataset_file)

    def _load_passages(self, passages_file: str) -> Dataset:
        return Dataset.from_json(passages_file, features=PASSAGES_FEATURES)

    def _load_dataset(self, dataset_file: str) -> Dataset:
        return Dataset.from_json(dataset_file)

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

    def _collate_fn(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        qids = [example["qid"] for example in examples]
        positions = [example["position"] for example in examples]
        questions = [example["question"] for example in examples]

        return {"qids": qids, "positions": positions, "questions": questions}

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> list[dict[str, Any]]:
        qids = batch["qids"]
        positions = batch["positions"]
        questions = batch["questions"]

        predictions = self.predict_answers(
            questions,
            num_passages=self.hparams.predict_num_passages,
            num_candidates=self.hparams.predict_num_candidates,
            answer_score_threshold=self.hparams.predict_answer_score_threshold,
        )

        outputs = []
        for qid, position, prediction in zip(qids, positions, predictions):
            pred_answer = prediction["pred_answer"]
            outputs.append({"qid": qid, "position": position, "prediction": pred_answer})

        return outputs

    def predict_answers(
        self,
        questions: list[str],
        num_passages: int = 100,
        num_candidates: int = 1000,
        answer_score_threshold: float = 0.0,
    ) -> list[dict[str, str | float]]:
        if not hasattr(self, "all_passages"):
            self.all_passages = self._load_passages(self.hparams.passages_file)

        retriever_predictions = self.retriever_module.retrieve_passages(
            questions, num_passages=num_passages, num_candidates=num_candidates
        )

        predictions = []
        for question, retriever_prediction in zip(questions, retriever_predictions):
            passages = [self.all_passages[passage_info["idx"]] for passage_info in retriever_prediction]
            passage_titles = [passage["title"] for passage in passages]
            passage_texts = [passage["text"] for passage in passages]

            reader_prediction = self.reader_predict_module.predict_answer(question, passage_titles, passage_texts)
            score = reader_prediction["score"]
            if score >= answer_score_threshold:
                pred_answer = reader_prediction["pred_answer"]
            else:
                pred_answer = None

            predictions.append({"pred_answer": pred_answer, "score": score})

        return predictions
