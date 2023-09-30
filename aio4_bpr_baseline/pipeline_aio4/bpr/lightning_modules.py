from typing import Any

from datasets import Dataset
from lightning import LightningModule
from torch.utils.data import DataLoader

from aio4_bpr_baseline.models.reader.extractive_reader.lightning_modules import ReaderPredictionLightningModule
from aio4_bpr_baseline.models.retriever.bpr.lightning_modules import RetrieverLightningModule
from aio4_bpr_baseline.utils.data import PASSAGE_DATASET_FEATURES


class PipelineLightningModule(LightningModule):
    def __init__(
        self,
        biencoder_ckpt_file: str,
        reader_ckpt_file: str,
        passage_faiss_index_file: str,
        passage_dataset_file: str,
        predict_dataset_file: str | None = None,
        predict_batch_size: int = 2,
        predict_retriever_k: int = 100,
        predict_retriever_num_candidates: int = 1000,
        predict_answer_score_threshold: float = 0.0,
        datasets_num_proc: int | None = None,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.retriever_module = RetrieverLightningModule(
            self.hparams.biencoder_ckpt_file, self.hparams.passage_faiss_index_file
        )
        self.reader_prediction_module = ReaderPredictionLightningModule(self.hparams.reader_ckpt_file)

        self.passage_dataset = Dataset.from_json(
            self.hparams.passage_dataset_file,
            features=PASSAGE_DATASET_FEATURES,
            num_proc=self.hparams.datasets_num_proc,
        )

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
            retriever_k=self.hparams.predict_retriever_k,
            retriever_num_candidates=self.hparams.predict_retriever_num_candidates,
            answer_score_threshold=self.hparams.predict_answer_score_threshold,
        )

        outputs = []
        for qid, position, prediction in zip(qids, positions, predictions):
            pred_answer = prediction["pred_answer"]
            score = prediction["score"]
            outputs.append({"qid": qid, "position": position, "prediction": pred_answer, "score": score})

        return outputs

    def predict_answers(
        self,
        questions: list[str],
        retriever_k: int = 100,
        retriever_num_candidates: int = 1000,
        answer_score_threshold: float = 0.0,
    ) -> list[dict[str, str | float]]:
        retriever_predictions = self.retriever_module.retrieve_passages(
            questions, k=retriever_k, num_candidates=retriever_num_candidates
        )

        passage_titles = []
        passage_texts = []
        for retriever_prediction in retriever_predictions:
            passage_ids = retriever_prediction["passage_ids"]
            passages = [self.passage_dataset[passage_id] for passage_id in passage_ids]

            passage_titles.append([passage["title"] for passage in passages])
            passage_texts.append([passage["text"] for passage in passages])

        reader_predictions = self.reader_prediction_module.predict_answers(questions, passage_titles, passage_texts)

        output_predictions = []
        for reader_prediction in reader_predictions:
            score = reader_prediction["score"]
            if score >= answer_score_threshold:
                pred_answer = reader_prediction["pred_answer"]
            else:
                pred_answer = None

            output_predictions.append({"pred_answer": pred_answer, "score": score})

        return output_predictions
