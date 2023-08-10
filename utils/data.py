import gzip
from typing import Sequence

from datasets import Features, Value
from transformers import BatchEncoding


DATASET_FEATURES = Features({
    "qid": Value(dtype="string"),
    "position": Value(dtype="int64"),
    "question": Value(dtype="string"),
    "answers": [Value(dtype="string")],
    "passages": [
        {
            "pid": Value(dtype="string"),
            "title": Value(dtype="string"),
            "text": Value(dtype="string"),
            "score": Value(dtype="float32"),
        }
    ],
    "positive_passage_idxs": [Value(dtype="int64")],
    "negative_passage_idxs": [Value(dtype="int64")],
})

PASSAGE_DATASET_FEATURES = Features({
    "pid": Value(dtype="string"),
    "title": Value(dtype="string"),
    "text": Value(dtype="string"),
})

BUZZER_FEATURES = Features({
    "qid": Value(dtype="string"),
    "question": Value(dtype="string"),
    "answers": [Value(dtype="string")],
    "pred_answers": [Value(dtype="string")],
    "pred_scores": [Value(dtype="float32")],
    "is_correct": Value(dtype="bool"),
})


def find_spans(
    source: Sequence[int], targets: Sequence[Sequence[int]], mask: Sequence[bool] | None = None,
) -> list[tuple[int, int]]:
    if mask is None:
        mask = [True] * len(source)

    assert len(source) == len(mask)

    spans: list[tuple[int, int]] = []
    for target in targets:
        for i in range(len(source) - len(target) + 1):
            if source[i : i + len(target)] == target and all(mask[i : i + len(target)]):
                spans.append((i, i + len(target) - 1))

    return spans


def resize_batch_encoding(batch_encoding: BatchEncoding, size: Sequence[int]) -> BatchEncoding:
    return BatchEncoding({key: tensor.view(*size) for key, tensor in batch_encoding.items()})


def open_file(filepath: str, mode: str = "rt"):
    if filepath.endswith(".gz"):
        return gzip.open(filepath, mode=mode)
    else:
        return open(filepath, mode=mode)
