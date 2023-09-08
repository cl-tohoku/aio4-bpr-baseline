import argparse
import json
import re
import unicodedata
import pandas as pd
import time


def normalize_answer(text: str) -> str:
    if text is None or isinstance(text, float):
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


def get_all_gold_answers(gold_file: str) -> (dict[str, str], dict[str, str]):
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

    assert len(all_gold_answers) == len(all_questions)
    return all_gold_answers, all_questions


def get_all_pred_answers(*, prediction_file: str = None, df: pd.DataFrame = None) -> dict[str, dict[int, str]]:
    if prediction_file is not None and df is not None:
        raise ValueError("Only one of prediction_file or df should be specified.")
    if prediction_file is None and df is None:
        raise ValueError("Either prediction_file or df should be specified.")
    if prediction_file:
        df = pd.read_json(prediction_file, lines=True)
    # df['prediction'] = df['prediction'].astype(str)
    all_pred_answers: dict[str, dict[int, str]] = {}  # qid -> position -> answer
    df["prediction"] = df["prediction"].apply(normalize_answer)
    all_pred_answers = df.groupby("qid").apply(lambda x: x.set_index("position").to_dict()["prediction"]).to_dict()
    return all_pred_answers


def compute_scores(
    all_gold_answers, all_questions, all_pred_answers, limit_num_wrong_answers: int = None
) -> dict[str, float]:
    num_questions = len(all_questions)

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
        position_score += 1.0 - correct_position / len(question)

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


def main(
    gold_file: str,
    prediction_file: str,
    limit_num_wrong_answers: int = None,
) -> dict[str, float]:
    # load the gold file
    all_gold_answers, all_questions = get_all_gold_answers(gold_file)
    num_questions = len(all_questions)

    # load the prediction file
    all_pred_answers = get_all_pred_answers(prediction_file=prediction_file)
    assert len(all_pred_answers) == num_questions

    scores = compute_scores(
        all_gold_answers,
        all_questions,
        all_pred_answers,
        limit_num_wrong_answers=limit_num_wrong_answers,
    )
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--limit_num_wrong_answers", type=int)
    args = parser.parse_args()

    # start = time.time()
    scores = main(args.gold_file, args.prediction_file, limit_num_wrong_answers=args.limit_num_wrong_answers)
    # end = time.time()
    # print(end - start)

    print("num_questions: {}".format(scores["num_questions"]))
    print("num_correct: {}".format(scores["num_correct"]))
    print("num_missed: {}".format(scores["num_missed"]))
    print("num_failed: {}".format(scores["num_failed"]))
    print("accuracy: {:.1%}".format(scores["accuracy"]))
    print("accuracy_score: {:.3f}".format(scores["accuracy_score"]))
    print("position_score: {:.3f}".format(scores["position_score"]))
    print("total_score: {:.3f}".format(scores["total_score"]))
