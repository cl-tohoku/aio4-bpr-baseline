import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main(args: argparse.Namespace):
    predictions_dir = Path(args.predictions_dir)

    _idx: list[np.ndarray] = []
    prediction: list[np.ndarray] = []
    for prediction_file in tqdm(predictions_dir.iterdir()):
        with np.load(prediction_file) as npz:
            _idx.append(npz["_idx"])
            prediction.append(npz["prediction"])

    _idx = np.concatenate(_idx, axis=0)
    prediction = np.concatenate(prediction, axis=0)

    prediction = prediction[np.argsort(_idx)]

    np.save(args.output_file, prediction)

    if args.remove_predictions_dir:
        for prediction_file in predictions_dir.iterdir():
            prediction_file.unlink()

        predictions_dir.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--remove_predictions_dir", action="store_true")

    args = parser.parse_args()
    main(args)
