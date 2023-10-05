import argparse
import json
import logging
from collections.abc import Iterator
from typing import Any

from tqdm import tqdm

from aio4_bpr_baseline.utils.data import open_file


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger()


def load_dataset_examples(dataset_file: str) -> list[dict[str, Any]]:
    dataset_examples = []
    with open_file(dataset_file) as f:
        for line in tqdm(f):
            example = json.loads(line)
            dataset_examples.append(example)

    return dataset_examples


def load_pid_idx_map(pid_idx_map_file: str) -> dict[str, int]:
    with open_file(pid_idx_map_file) as f:
        pid_idx_map = json.load(f)

    return pid_idx_map


def generate_dataset_examples(
    dataset_examples: list[dict[str, Any]], pid_idx_map: dict[str, int]
) -> Iterator[dict[str, Any]]:
    for example in tqdm(dataset_examples):
        qid = example["qid"]
        question = example["question"]
        answers = example["answers"]

        passages = example["passages"]

        positive_passages = []
        for i in example["positive_passage_indices"]:
            pid = str(passages[i]["passage_id"])
            idx = pid_idx_map[pid]
            positive_passages.append({"idx": idx, "pid": pid, "title": None, "text": None, "score": None})

        negative_passages = []
        for i in example["negative_passage_indices"]:
            pid = str(passages[i]["passage_id"])
            idx = pid_idx_map[pid]
            negative_passages.append({"idx": idx, "pid": pid, "title": None, "text": None, "score": None})

        output_example = {
            "qid": qid,
            "question": question,
            "answers": answers,
            "positive_passages": positive_passages,
            "negative_passages": negative_passages,
        }
        yield output_example


def make_dataset(dataset_file: str, pid_idx_map_file: str, output_dataset_file: str):
    logger.info(f"Loading {dataset_file}")
    dataset_examples = load_dataset_examples(dataset_file)

    logger.info(f"Loading {pid_idx_map_file}")
    pid_idx_map = load_pid_idx_map(pid_idx_map_file)

    logger.info(f"Writing to {output_dataset_file}")
    with open_file(output_dataset_file, "wt") as fo:
        for dataset_example in generate_dataset_examples(dataset_examples, pid_idx_map):
            print(json.dumps(dataset_example, ensure_ascii=False), file=fo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--pid_idx_map_file", type=str, required=True)
    parser.add_argument("--output_dataset_file", type=str, required=True)
    args = parser.parse_args()

    make_dataset(args.dataset_file, args.pid_idx_map_file, args.output_dataset_file)


if __name__ == "__main__":
    main()
