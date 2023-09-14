import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_prediction(predictions_dir: str) -> np.ndarray:
    _idx_arrays = []
    prediction_arrays = []
    for prediction_file in tqdm(Path(predictions_dir).iterdir()):
        with np.load(prediction_file) as npz:
            _idx_arrays.append(npz["_idx"])
            prediction_arrays.append(npz["prediction"])

    _idx_array = np.concatenate(_idx_arrays, axis=0)
    prediction_array = np.concatenate(prediction_arrays, axis=0)
    prediction_array = prediction_array[np.argsort(_idx_array)]
    return prediction_array


def write_prediction(prediction_array: np.ndarray, output_file: str):
    np.save(output_file, prediction_array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    prediction_array = load_prediction(args.predictions_dir)
    write_prediction(prediction_array, args.output_file)


if __name__ == "__main__":
    main()
