import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from aio4_bpr_baseline.utils.data import open_file


def load_prediction(predictions_dir: str) -> list[Any]:
    idxs = []
    prediction = []

    for prediction_file in tqdm(Path(predictions_dir).iterdir()):
        with gzip.open(prediction_file, "rt") as f:
            dumped_dict = json.load(f)
            idxs.extend(dumped_dict["idxs"])
            prediction.extend(dumped_dict["prediction"])

    assert len(idxs) == len(prediction)

    prediction = [prediction_item for idx, prediction_item in sorted(zip(idxs, prediction), key=lambda x: x[0])]
    return prediction


def write_prediction(prediction: list[Any], output_file: str):
    with open_file(output_file, "wt") as fo:
        for prediction_item in tqdm(prediction):
            print(json.dumps(prediction_item, ensure_ascii=False), file=fo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    prediction = load_prediction(args.predictions_dir)
    write_prediction(prediction, args.output_file)


if __name__ == "__main__":
    main()
