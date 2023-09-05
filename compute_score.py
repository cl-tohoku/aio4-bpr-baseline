import argparse
import json
import re
import unicodedata


def normalize_answer(text: str | None) -> str | None:
    if text is None:
        return None

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


def compute_scores(
    gold_file: str,
    prediction_file: str,
    limit_num_wrong_answers: int | None = None,
) -> dict[str, float]:
    # load the gold file
    all_questions: dict[str, str] = {}  # qid -> question
    all_gold_answers: dict[str, list[str]] = {}  # qid -> answers
    with open(gold_file) as f:
        for line in f:
            gold_item = json.loads(line)
            qid = gold_item["qid"]
            question = gold_item["question"]
            gold_answers = gold_item["answers"]

            all_questions[qid] = question
            all_gold_answers[qid] = [normalize_answer(answer) for answer in gold_answers]

    num_questions = len(all_questions)
    assert len(all_gold_answers) == num_questions

    # load the prediction file
    all_pred_answers: dict[str, dict[int, str]] = {}  # qid -> position -> answer
    with open(prediction_file) as f:
        for line in f:
            pred_item = json.loads(line)
            qid = pred_item["qid"]
            position = pred_item["position"]
            pred_answer = pred_item["prediction"]

            if qid not in all_pred_answers:
                all_pred_answers[qid]: dict[int, str] = {}

            all_pred_answers[qid][position] = normalize_answer(pred_answer)

    assert len(all_pred_answers) == num_questions

    # calculate scores
    accuracy_score = 0.0
    position_score = 0.0
    num_correct = 0
    num_missed = 0
    num_failed = 0
    for qid, question in all_questions.items():
        pred_answers = all_pred_answers[qid]  # position -> pred_answer
        gold_answers = all_gold_answers[qid]

        correct_position: int | None = None  # the earliest position of the correct predictions
        wrong_answers = set()

        for position, pred_answer in sorted(pred_answers.items(), key=lambda x: x[0]):
            if pred_answer in gold_answers:
                correct_position = position
                break
            elif pred_answer is not None:
                wrong_answers.add(pred_answer)

        if correct_position is None:
            num_missed += 1
            continue

        if limit_num_wrong_answers is not None and len(wrong_answers) > limit_num_wrong_answers:
            num_failed += 1
            continue

        num_correct += 1
        accuracy_score += 1.0
        position_score += (1.0 - correct_position / len(question))

    accuracy = num_correct / num_questions
    total_score = accuracy_score + position_score

    scores = {
        "num_questions": num_questions,
        "num_correct": num_correct,
        "num_missed": num_missed,
        "num_failed": num_failed,
        "accuracy": accuracy,
        "accuracy_score": accuracy_score,
        "position_score": position_score,
        "total_score": total_score,
    }
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--limit_num_wrong_answers", type=int)
    args = parser.parse_args()

    scores = compute_scores(args.gold_file, args.prediction_file, limit_num_wrong_answers=args.limit_num_wrong_answers)

    print("num_questions: {}".format(scores["num_questions"]))
    print("num_correct: {}".format(scores["num_correct"]))
    print("num_missed: {}".format(scores["num_missed"]))
    print("num_failed: {}".format(scores["num_failed"]))
    print("accuracy: {:.1%}".format(scores["accuracy"]))
    print("accuracy_score: {:.3f}".format(scores["accuracy_score"]))
    print("position_score: {:.3f}".format(scores["position_score"]))
    print("total_score: {:.3f}".format(scores["total_score"]))
