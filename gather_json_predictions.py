import argparse
import json
import gzip
from pathlib import Path
from typing import Any

from tqdm import tqdm
from utils.data import open_file


def main(args: argparse.Namespace):
    predictions_dir = Path(args.predictions_dir)

    prediction: list[dict[str, Any]] = []
    for prediction_file in tqdm(predictions_dir.iterdir()):
        prediction += [json.loads(line) for line in gzip.open(prediction_file, "rt")]

    prediction = sorted(prediction, key=lambda x: x["_idx"])

    with open_file(args.output_file, "wt") as fo:
        for prediction_item in tqdm(prediction):
            prediction_item.pop("_idx")
            print(json.dumps(prediction_item, ensure_ascii=False), file=fo)

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
