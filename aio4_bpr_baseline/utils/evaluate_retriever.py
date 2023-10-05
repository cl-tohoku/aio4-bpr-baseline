import argparse
import json
import re
import unicodedata
from typing import Any

import regex
from datasets import Dataset
from tqdm import tqdm

from aio4_bpr_baseline.utils.data import PASSAGES_FEATURES, open_file


SIMPLE_TOKENIZER_REGEXP = regex.compile(
    r"([\p{L}\p{N}\p{M}]+|[^\p{Z}\p{C}])", flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
)


def simple_tokenize(text: str) -> list[str]:
    tokens = [match.group().lower() for match in SIMPLE_TOKENIZER_REGEXP.finditer(text)]
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
                continue
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


def load_dataset_file(dataset_file: str) -> list[str, Any]:
    examples = []
    with open_file(dataset_file, "rt") as f:
        for line in tqdm(f):
            example = json.loads(line)
            examples.append(example)

    return examples


def load_prediction_file(prediction_file: str) -> list[str, Any]:
    prediction = []
    with open_file(prediction_file, "rt") as f:
        for line in tqdm(f):
            prediction_item = json.loads(line)
            prediction.append(prediction_item)

    return prediction


def load_passages(passages_file: str) -> Dataset:
    return Dataset.from_json(passages_file, features=PASSAGES_FEATURES)


def update_example_with_prediction(
    example: dict[str, Any],
    prediction_item: list[dict[str, int | float]],
    all_passages: Dataset,
    answer_match_type: str,
) -> dict[str, Any]:
    answers = example["answers"]

    positive_passages = []
    negative_passages = []
    for passage_info in prediction_item:
        idx = passage_info["idx"]
        score = passage_info["score"]

        passage = all_passages[idx]
        assert idx == passage["idx"]
        pid = passage["pid"]
        text = passage["text"]

        if has_answer(text, answers, answer_match_type=answer_match_type):
            positive_passages.append({"idx": idx, "pid": pid, "title": None, "text": None, "score": score})
        else:
            negative_passages.append({"idx": idx, "pid": pid, "title": None, "text": None, "score": score})

    updated_example = dict(**example)
    updated_example["positive_passages"] = positive_passages
    updated_example["negative_passages"] = negative_passages

    return updated_example


def compute_metrics(examples: list[dict[str, Any]], recall_ks: list[int], mrr_ks: list[int]) -> dict[str, float]:
    ranks = []
    for example in examples:
        positive_passages = example["positive_passages"]
        negative_passages = example["negative_passages"]
        if len(positive_passages) > 0:
            top_positive_score = positive_passages[0]["score"]
            rank = sum(1 for passage_info in negative_passages if passage_info["score"] > top_positive_score) + 1
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
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--passages_file", type=str, required=True)
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--answer_match_type", choices=("string", "regex", "nfkc"), default="string")
    parser.add_argument("--recall_k", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100])
    parser.add_argument("--mrr_k", type=int, nargs="+", default=[10])
    args = parser.parse_args()

    examples = load_dataset_file(args.dataset_file)
    prediction = load_prediction_file(args.prediction_file)

    num_examples = len(examples)
    num_prediction = len(prediction)

    if num_examples != num_prediction:
        raise ValueError(
            "The number of items in dataset_file and prediction_file are not the same.",
            f"({num_examples}) != ({num_prediction})",
        )

    all_passages = load_passages(args.passages_file)

    updated_examples = [
        update_example_with_prediction(example, prediction_item, all_passages, args.answer_match_type)
        for example, prediction_item in tqdm(zip(examples, prediction), total=num_examples)
    ]

    metrics = compute_metrics(updated_examples, args.recall_k, args.mrr_k)

    if args.output_file is not None:
        write_dataset(updated_examples, args.output_file)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
