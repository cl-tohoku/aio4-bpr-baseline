import argparse
import json
import gzip
from pathlib import Path
from typing import Any

from tqdm import tqdm

from aio4_bpr_baseline.utils.data import open_file


def load_prediction(predictions_dir: str) -> list[dict[str, Any]]:
    prediction_items = []
    for prediction_file in tqdm(Path(predictions_dir).iterdir()):
        prediction_items += [json.loads(line) for line in gzip.open(prediction_file, "rt")]

    prediction_items = sorted(prediction_items, key=lambda x: x.pop("_idx"))
    return prediction_items


def write_prediction(prediction_items: list[dict[str, Any]], output_file: str):
    with open_file(output_file, "wt") as fo:
        for prediction_item in tqdm(prediction_items):
            print(json.dumps(prediction_item, ensure_ascii=False), file=fo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    prediction_items = load_prediction(args.predictions_dir)
    write_prediction(prediction_items, args.output_file)


if __name__ == "__main__":
    main()
