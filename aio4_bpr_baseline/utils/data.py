import gzip

from datasets import Features, Value


DATASET_FEATURES = Features({
    "qid": Value(dtype="string"),
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


def open_file(filepath: str, mode: str = "rt"):
    if filepath.endswith(".gz"):
        return gzip.open(filepath, mode=mode)
    else:
        return open(filepath, mode=mode)
