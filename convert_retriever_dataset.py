import argparse
import json

from tqdm import tqdm

from utils.data import open_file


def main(args: argparse.Namespace):
    with open_file(args.input_dataset_file) as f, open_file(args.output_dataset_file, "wt") as fo:
        for line in tqdm(f):
            example = json.loads(line)

            qid = example["qid"]

            questions: list[str] = []
            for i in range(args.num_question_splits):
                question_length = int(len(example["question"]) * (i + 1) / args.num_question_splits)
                question = example["question"][:question_length]
                questions.append(question)

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

            for question in questions:
                output_example = {
                    "qid": qid,
                    "position": len(question),
                    "question": question,
                    "answers": answers,
                    "passages": passages,
                    "positive_passage_idxs": positive_passage_idxs,
                    "negative_passage_idxs": negative_passage_idxs,
                }
                print(json.dumps(output_example, ensure_ascii=False), file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dataset_file", type=str, required=True)
    parser.add_argument("--output_dataset_file", type=str, required=True)
    parser.add_argument("--num_question_splits", type=int, default=1)

    args = parser.parse_args()
    main(args)
