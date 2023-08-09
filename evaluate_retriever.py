import argparse
import json
import re
import unicodedata

import regex
from datasets import Dataset
from tqdm import tqdm

from utils.data import PASSAGE_DATASET_FEATURES, open_file


SIMPLE_TOKENIZER_REGEXP = regex.compile(
    r"([\p{L}\p{N}\p{M}]+|[^\p{Z}\p{C}])", flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
)


def simple_tokenize(text: str) -> list[str]:
    tokens = [match.group() for match in SIMPLE_TOKENIZER_REGEXP.finditer(text)]
    return tokens


def has_answer(passage: dict[str, str | float], answers: list[str], match_type: str = "string") -> bool:
    if match_type == "string":
        passage_text = unicodedata.normalize("NFD", passage["text"])
        passage_tokens = simple_tokenize(passage_text)
        for answer in answers:
            answer_text = unicodedata.normalize("NFD", answer)
            answer_tokens = simple_tokenize(answer_text)
            for i in range(len(passage_tokens) - len(answer_tokens) + 1):
                if passage_tokens[i : i + len(answer_tokens)] == answer_tokens:
                    return True
    elif match_type == "regex":
        passage_text = unicodedata.normalize("NFD", passage["text"])
        for answer in answers:
            answer_text = unicodedata.normalize("NFD", answer)
            try:
                answer_regexp = re.compile(answer_text, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
            except BaseException:
                return False
            if answer_regexp.search(passage_text) is not None:
                return True
    elif match_type == "nfkc":
        passage_text = unicodedata.normalize("NFKC", passage["text"]).lower()
        passage_text = " ".join(passage_text.split())
        for answer in answers:
            answer_text = unicodedata.normalize("NFKC", answer).lower()
            answer_text = " ".join(answer_text.split())
            if answer_text in passage_text:
                return True

    return False


def main(args: argparse.Namespace):
    num_examples = sum(1 for _ in open_file(args.retriever_input_file))
    num_prediction_items = sum(1 for _ in open_file(args.retriever_prediction_file))

    if num_examples != num_prediction_items:
        raise RuntimeError(
            "retriever_input_file and retriever_prediction_file have different number of lines "
            f"({num_examples} != {num_prediction_items})."
        )

    passage_dataset = Dataset.from_json(
        args.passage_dataset_file, features=PASSAGE_DATASET_FEATURES, num_proc=args.datasets_num_proc
    )
    ranks: list[int | None] = []

    if args.output_file is not None:
        fo = open_file(args.output_file, "wt")
    else:
        fo = None

    with open_file(args.retriever_input_file) as fi, open_file(args.retriever_prediction_file) as fp:
        for fi_line, fp_line in tqdm(zip(fi, fp), total=num_examples):
            example = json.loads(fi_line)
            prediction_item = json.loads(fp_line)

            passages: list[dict[str, str | float]] = []
            positive_passage_idxs: list[int] = []
            negative_passage_idxs: list[int] = []
            rank: int | None = None

            for i, (passage_idx, score) in enumerate(zip(prediction_item["passage_idxs"], prediction_item["scores"])):
                passage = passage_dataset[passage_idx]
                passage["score"] = score

                if has_answer(passage, example["answers"], match_type=args.match_type):
                    positive_passage_idxs.append(i)
                    if rank is None:
                        rank = i + 1
                else:
                    negative_passage_idxs.append(i)

                passages.append(passage)

            ranks.append(rank)

            if fo is not None:
                output_example = {
                    "qid": example["qid"],
                    "question": example["question"],
                    "answers": example["answers"],
                    "passages": passages,
                    "positive_passage_idxs": positive_passage_idxs,
                    "negative_passage_idxs": negative_passage_idxs,
                }
                print(json.dumps(output_example, ensure_ascii=False), file=fo)

    if fo is not None:
        fo.close()

    for k in args.recall_k:
        num_correct = sum(1 for rank in ranks if rank is not None and rank <= k)
        recall = num_correct / len(ranks)
        print(f"Recall@{k}: {recall:.4f} ({num_correct}/{len(ranks)})")

    for k in args.mrr_k:
        mrr = sum(1 / rank if rank is not None else 0 for rank in ranks) / len(ranks)
        print(f"MRR@{k}: {mrr:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--retriever_input_file", type=str, required=True)
    parser.add_argument("--retriever_prediction_file", type=str, required=True)
    parser.add_argument("--passage_dataset_file", type=str, required=True)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--match_type", choices=("string", "regex", "nfkc"), default="string")
    parser.add_argument("--recall_k", type=int, default=[1, 2, 5, 10, 20, 50, 100])
    parser.add_argument("--mrr_k", type=int, default=[10])
    parser.add_argument("--datasets_num_proc", type=int)

    args = parser.parse_args()
    main(args)
