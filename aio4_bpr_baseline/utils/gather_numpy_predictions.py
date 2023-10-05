import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_prediction(predictions_dir: str) -> np.ndarray:
    idxs_arrays = []
    prediction_arrays = []
    for prediction_file in tqdm(Path(predictions_dir).iterdir()):
        with np.load(prediction_file) as npz:
            idxs_arrays.append(npz["idxs"])
            prediction_arrays.append(npz["prediction"])

    idxs = np.concatenate(idxs_arrays, axis=0)
    prediction = np.concatenate(prediction_arrays, axis=0)
    assert idxs.shape[0] == prediction.shape[0]

    prediction = prediction[np.argsort(idxs)]
    return prediction


def write_prediction(prediction: np.ndarray, output_file: str):
    np.save(output_file, prediction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    prediction = load_prediction(args.predictions_dir)
    write_prediction(prediction, args.output_file)


if __name__ == "__main__":
    main()
