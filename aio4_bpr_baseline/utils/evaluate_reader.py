import argparse
import json
import re
import string
import unicodedata
from typing import Any

from datasets import Dataset
from tqdm import tqdm

from aio4_bpr_baseline.utils.data import PASSAGES_FEATURES, open_file


def normalize_answer(answer_text: str, mode: str = "default") -> str:
    if mode == "default":
        answer_text = answer_text.lower()
        answer_text = "".join(ch for ch in answer_text if ch not in set(string.punctuation))
        answer_text = re.sub(r"\b(a|an|the)\b", " ", answer_text)
        answer_text = " ".join(answer_text.split())
    elif mode == "nfkc":
        answer_text = unicodedata.normalize("NFKC", answer_text)
        answer_text = answer_text.lower()
        answer_text = "".join(answer_text.split())

    return answer_text


def load_dataset_file(dataset_file: str) -> list[str, Any]:
    examples = []
    with open_file(dataset_file, "rt") as f:
        for line in tqdm(f):
            example = json.loads(line)
            examples.append(example)

    return examples


def load_prediction_file(prediction_file: str) -> list[str, Any]:
    prediction_items = []
    with open_file(prediction_file, "rt") as f:
        for line in tqdm(f):
            prediction_item = json.loads(line)
            prediction_items.append(prediction_item)

    return prediction_items


def load_passages(passages_file: str) -> Dataset:
    return Dataset.from_json(passages_file, features=PASSAGES_FEATURES)


def compute_metrics(
    examples: list[dict[str, Any]], prediction_items: list[str, Any], answer_normalization_mode: str = "default"
) -> dict[str, float]:
    assert len(examples) == len(prediction_items)

    num_correct = 0
    for example, prediction_item in tqdm(zip(examples, prediction_items)):
        gold_answers = [normalize_answer(answer, mode=answer_normalization_mode) for answer in example["answers"]]
        pred_answer = normalize_answer(prediction_item["pred_answer"], mode=answer_normalization_mode)

        is_correct = pred_answer in gold_answers

        if is_correct:
            num_correct += 1

    metrics = {}
    em = num_correct / len(examples)
    metrics["Exact Match"] = em

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--passages_file", type=str, required=True)
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--answer_normalization_mode", choices=("default", "nfkc"), default="default")
    args = parser.parse_args()

    examples = load_dataset_file(args.dataset_file)
    prediction_items = load_prediction_file(args.prediction_file)

    num_examples = len(examples)
    num_prediction_items = len(prediction_items)

    if num_examples != num_prediction_items:
        raise ValueError(
            "The number of items in dataset_file and prediction_file are not the same.",
            f"({num_examples}) != ({num_prediction_items})",
        )

    metrics = compute_metrics(examples, prediction_items, answer_normalization_mode=args.answer_normalization_mode)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
