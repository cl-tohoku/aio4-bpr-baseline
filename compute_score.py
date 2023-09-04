import argparse
import json
import re
import unicodedata


def normalize_answer(text: str) -> str:
    # substitute some symbols that will not be replaced by unicode normalization
    text = text.replace("～", "〜")

    # unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # lowercase alphabetical characters
    text = text.lower()

    # remove kagi-kakkos
    text = re.sub(r"「(.*?)」", r"\1", text)
    text = re.sub(r"『(.*?)』", r"\1", text)

    # remove some punctuation marks
    text = text.replace("・", "")
    text = text.replace("=", "")
    text = text.replace("-", "")

    # compress whitespaces
    text = re.sub(r"\s+", "", text).strip()

    return text


def main(args: argparse.Namespace):
    all_questions = {}  # qid -> question
    all_gold_answers = {}  # qid -> answers
    with open(args.gold_file) as f:
        for line in f:
            gold_item = json.loads(line)
            qid = gold_item["qid"]
            question = gold_item["question"]
            gold_answers = gold_item["answers"]

            all_questions[qid] = question
            all_gold_answers[qid] = [normalize_answer(answer) for answer in gold_answers]

    num_questions = len(all_questions)
    print("num_questions:", num_questions)
    assert len(all_gold_answers) == num_questions

    all_pred_answers = {}  # qid -> position -> answer
    with open(args.prediction_file) as f:
        for line in f:
            pred_item = json.loads(line)
            qid = pred_item["qid"]
            position = pred_item["position"]
            pred_answer = pred_item["prediction"]

            if qid not in all_pred_answers:
                all_pred_answers[qid] = {}

            all_pred_answers[qid][position] = normalize_answer(pred_answer)

    assert len(all_pred_answers) == num_questions

    all_correct_positions = {}  # qid -> positions
    for qid, gold_answers in all_gold_answers.items():
        all_correct_positions[qid] = []

        pred_answers = all_pred_answers[qid]  # position -> pred_answer
        for position, pred_answer in sorted(pred_answers.items(), key=lambda x: x[0]):
            if pred_answer in gold_answers:
                all_correct_positions[qid].append(position)

    assert len(all_correct_positions) == num_questions

    accuracy_score = 0.0
    position_score = 0.0

    for qid, positions in all_correct_positions.items():
        if len(positions) == 0:
            continue

        best_position = positions[0]
        question_length = len(all_questions[qid])

        accuracy_score += 1.0
        position_score += (1.0 - best_position / question_length)

    accuracy = accuracy_score / num_questions
    total_score = accuracy_score + position_score

    print(f"accuracy: {accuracy:.1%}")
    print(f"accuracy_score: {accuracy_score:.3f}")
    print(f"position_score: {position_score:.3f}")
    print(f"total_score: {total_score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--gold_file", type=str, required=True)
    args = parser.parse_args()

    main(args)
