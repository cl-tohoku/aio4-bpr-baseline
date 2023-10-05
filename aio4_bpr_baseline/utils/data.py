import gzip

from datasets import Features, Value


DATASET_FEATURES = Features(
    {
        "qid": Value(dtype="string"),
        "question": Value(dtype="string"),
        "answers": [Value(dtype="string")],
        "positive_passages": [
            {
                "idx": Value(dtype="int64"),
                "pid": Value(dtype="string"),
                "title": Value(dtype="string"),
                "text": Value(dtype="string"),
                "score": Value(dtype="float32"),
            }
        ],
        "negative_passages": [
            {
                "idx": Value(dtype="int64"),
                "pid": Value(dtype="string"),
                "title": Value(dtype="string"),
                "text": Value(dtype="string"),
                "score": Value(dtype="float32"),
            }
        ],
    }
)

PASSAGES_FEATURES = Features(
    {
        "idx": Value(dtype="int64"),
        "pid": Value(dtype="string"),
        "title": Value(dtype="string"),
        "text": Value(dtype="string"),
    }
)


def open_file(filepath: str, mode: str = "rt"):
    if filepath.endswith(".gz"):
        return gzip.open(filepath, mode=mode)
    else:
        return open(filepath, mode=mode)
