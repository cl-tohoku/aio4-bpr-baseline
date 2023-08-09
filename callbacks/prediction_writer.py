import json
import gzip
from pathlib import Path
from typing import Any, Optional, Sequence
import lightning.pytorch as pl

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
        prediction: list[dict[str, Any]],
        batch_indices: list[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        predictions_dir = Path(trainer.log_dir, "predictions")
        predictions_dir.mkdir(exist_ok=True)
        prediction_file = predictions_dir / f"rank={trainer.global_rank}-batch={batch_idx}.jsonl.gz"

        with gzip.open(prediction_file, "wt") as fo:
            for _idx, prediction_item in zip(batch_indices, prediction):
                prediction_item["_idx"] = _idx
                print(json.dumps(prediction_item, ensure_ascii=False), file=fo)


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

        np.savez(prediction_file, _idx=np.array(batch_indices), prediction=prediction)
