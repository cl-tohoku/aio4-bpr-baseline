import argparse
import json
import re
import unicodedata
from typing import Any

import regex
from datasets import Dataset
from tqdm import tqdm

from aio4_bpr_baseline.utils.data import PASSAGE_DATASET_FEATURES, open_file


SIMPLE_TOKENIZER_REGEXP = regex.compile(
    r"([\p{L}\p{N}\p{M}]+|[^\p{Z}\p{C}])", flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
)


def simple_tokenize(text: str) -> list[str]:
    tokens = [match.group() for match in SIMPLE_TOKENIZER_REGEXP.finditer(text)]
    return tokens


def has_answer(passage_text: str, answers: list[str], answer_match_type: str = "string") -> bool:
    if answer_match_type == "string":
        passage_text = unicodedata.normalize("NFD", passage_text)
        passage_tokens = simple_tokenize(passage_text)
        for answer in answers:
            answer_text = unicodedata.normalize("NFD", answer)
            answer_tokens = simple_tokenize(answer_text)
            for i in range(len(passage_tokens) - len(answer_tokens) + 1):
                if passage_tokens[i : i + len(answer_tokens)] == answer_tokens:
                    return True

    elif answer_match_type == "regex":
        passage_text = unicodedata.normalize("NFD", passage_text)
        for answer in answers:
            answer_text = unicodedata.normalize("NFD", answer)
            try:
                answer_regexp = re.compile(answer_text, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
            except BaseException:
                return False
            if answer_regexp.search(passage_text) is not None:
                return True

    elif answer_match_type == "nfkc":
        passage_text = unicodedata.normalize("NFKC", passage_text).lower()
        passage_text = " ".join(passage_text.split())
        for answer in answers:
            answer_text = unicodedata.normalize("NFKC", answer).lower()
            answer_text = " ".join(answer_text.split())
            if answer_text in passage_text:
                return True

    return False


def load_input_file(input_file: str) -> list[str, Any]:
    examples = []
    with open_file(input_file, "rt") as f:
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


def load_passage_dataset(passage_dataset_file: str, datasets_num_proc: int | None = None) -> Dataset:
    passage_dataset = Dataset.from_json(
        passage_dataset_file, features=PASSAGE_DATASET_FEATURES, num_proc=datasets_num_proc
    )
    return passage_dataset


def update_example_with_prediction(
    example: dict[str, Any], prediction_item: dict[str, Any], passage_dataset: Dataset, answer_match_type: str
) -> dict[str, Any]:
    answers = example["answers"]
    passage_ids = prediction_item["passage_ids"]
    scores = prediction_item["scores"]

    passages = []
    positive_passage_idxs = []
    negative_passage_idxs = []

    for i, (passage_id, score) in enumerate(zip(passage_ids, scores)):
        passage = passage_dataset[passage_id]
        passage["score"] = score

        passages.append(passage)
        if has_answer(passage["text"], answers, answer_match_type=answer_match_type):
            positive_passage_idxs.append(i)
        else:
            negative_passage_idxs.append(i)

    updated_example = dict(**example)
    updated_example["passages"] = passages
    updated_example["positive_passage_idxs"] = positive_passage_idxs
    updated_example["negative_passage_idxs"] = negative_passage_idxs

    return updated_example


def compute_metrics(examples: list[dict[str, Any]], recall_ks: list[int], mrr_ks: list[int]) -> dict[str, float]:
    ranks = []
    for example in examples:
        positive_passage_idxs = example["positive_passage_idxs"]
        if len(positive_passage_idxs) > 0:
            rank = min(positive_passage_idxs) + 1
        else:
            rank = None

        ranks.append(rank)

    metrics = {}
    for k in recall_ks:
        recall = sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)
        metrics[f"Recall@{k}"] = recall

    for k in mrr_ks:
        mrr = sum(1 / rank for rank in ranks if rank is not None and rank <= k) / len(ranks)
        metrics[f"MRR@{k}"] = mrr

    return metrics


def write_dataset(examples: list[dict[str, Any]], output_file: str):
    with open_file(output_file, "wt") as fo:
        for example in tqdm(examples):
            print(json.dumps(example, ensure_ascii=False), file=fo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--passage_dataset_file", type=str, required=True)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--answer_match_type", choices=("string", "regex", "nfkc"), default="string")
    parser.add_argument("--recall_k", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100])
    parser.add_argument("--mrr_k", type=int, nargs="+", default=[10])
    parser.add_argument("--datasets_num_proc", type=int)
    args = parser.parse_args()

    examples = load_input_file(args.input_file)
    prediction_items = load_prediction_file(args.prediction_file)

    num_examples = len(examples)
    num_prediction_items = len(prediction_items)

    if num_examples != num_prediction_items:
        raise ValueError(
            "The number of items in input_file and prediction_file are not the same.",
            f"({num_examples}) != ({num_prediction_items})"
        )

    passage_dataset = load_passage_dataset(args.passage_dataset_file, args.datasets_num_proc)

    updated_examples = [
        update_example_with_prediction(example, prediction_item, passage_dataset, args.answer_match_type)
        for example, prediction_item in tqdm(zip(examples, prediction_items), total=num_examples)
    ]

    metrics = compute_metrics(updated_examples, args.recall_k, args.mrr_k)

    if args.output_file is not None:
        write_dataset(updated_examples, args.output_file)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
