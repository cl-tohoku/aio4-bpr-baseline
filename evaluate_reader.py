import argparse
import json
import re
import string
import unicodedata

from tqdm import tqdm

from utils.data import open_file


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


def main(args: argparse.Namespace):
    num_examples = sum(1 for _ in open_file(args.reader_input_file))
    num_predictions = sum(1 for _ in open_file(args.reader_prediction_file))

    if num_examples != num_predictions:
        raise RuntimeError("The number of lines in reader_input_file and reader_prediction_file must be the same.")

    num_correct = 0

    if args.output_file is not None:
        fo = open_file(args.output_file, "wt")
    else:
        fo = None

    with open_file(args.reader_input_file) as fi, open_file(args.reader_prediction_file) as fp:
        for fi_line, fp_line in tqdm(zip(fi, fp)):
            example = json.loads(fi_line)
            prediction = json.loads(fp_line)

            gold_answers = [normalize_answer(answer, mode=args.normalization_mode) for answer in example["answers"]]
            top_predicted_answer = normalize_answer(prediction["answers"][0], mode=args.normalization_mode)

            is_correct = top_predicted_answer in gold_answers

            if is_correct:
                num_correct += 1

            if fo is not None:
                output_example = {
                    "qid": example["qid"],
                    "position": example["position"],
                    "question": example["question"],
                    "answers": example["answers"],
                    "pred_answers": prediction["answers"],
                    "pred_scores": prediction["scores"],
                    "is_correct": is_correct,
                }
                print(json.dumps(output_example, ensure_ascii=False), file=fo)

    em = num_correct / num_examples

    print(f"Exact Match: {em:.4f} ({num_correct}/{num_examples})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--reader_input_file", type=str, required=True)
    parser.add_argument("--reader_prediction_file", type=str, required=True)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--normalization_mode", choices=("default", "nfkc"), default="default")

    args = parser.parse_args()
    main(args)
