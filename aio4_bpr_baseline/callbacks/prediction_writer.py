import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter


class JsonPredictionWriter(BasePredictionWriter):
    def __init__(self):
        super().__init__(write_interval="batch")

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: list[Any],
        batch_indices: list[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        predictions_dir = Path(trainer.log_dir, "predictions")
        predictions_dir.mkdir(exist_ok=True)
        prediction_file = predictions_dir / f"rank={trainer.global_rank}-batch={batch_idx}.jsonl.gz"

        assert len(prediction) == len(batch_indices)
        with gzip.open(prediction_file, "wt") as fo:
            json.dump({"idxs": batch_indices, "prediction": prediction}, fo, ensure_ascii=False)


class NumpyPredictionWriter(BasePredictionWriter):
    def __init__(self):
        super().__init__(write_interval="batch")

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: np.ndarray,
        batch_indices: list[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        predictions_dir = Path(trainer.log_dir, "predictions")
        predictions_dir.mkdir(exist_ok=True)
        prediction_file = predictions_dir / f"rank={trainer.global_rank}-batch={batch_idx}.npz"

        assert prediction.shape[0] == len(batch_indices)
        np.savez(prediction_file, idxs=np.array(batch_indices), prediction=prediction)
