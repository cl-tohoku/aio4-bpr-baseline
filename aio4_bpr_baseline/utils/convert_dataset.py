import argparse
import json

from tqdm import tqdm

from aio4_bpr_baseline.utils.data import open_file


def process_files(input_file: str, output_file: str):
    with open_file(input_file) as f, open_file(output_file, "wt") as fo:
        for line in tqdm(f):
            example = json.loads(line)

            qid = example["qid"]
            question = example["question"]
            answers = example["answers"]
            passages = [
                {
                    "pid": passage["passage_id"],
                    "title": passage["title"],
                    "text": passage["text"],
                    "score": None,
                }
                for passage in example["passages"]
            ]
            positive_passage_idxs = example["positive_passage_indices"]
            negative_passage_idxs = example["negative_passage_indices"]

            output_example = {
                "qid": qid,
                "question": question,
                "answers": answers,
                "passages": passages,
                "positive_passage_idxs": positive_passage_idxs,
                "negative_passage_idxs": negative_passage_idxs,
            }
            print(json.dumps(output_example, ensure_ascii=False), file=fo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    process_files(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
