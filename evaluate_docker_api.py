import argparse
import json
import logging
import urllib.request


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_unlabelded_file", type=str, required=True)
    parser.add_argument("--output_prediction_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.test_unlabelded_file) as f:
        test_items = [json.loads(line) for line in f]

    with open(args.output_prediction_file, "w") as fo:
        for i, test_item in enumerate(test_items, start=1):
            qid: str = test_item["qid"]
            position: int = test_item["position"]
            question: str = test_item["question"]

            logger.info(f"Processing qid = {qid}, position = {position} ({i}/{len(test_items)})")

            try:
                query = {"qid": qid, "position": position, "question": question}
                req = urllib.request.Request(f"http://localhost:8000/answer?{urllib.parse.urlencode(query)}")
                with urllib.request.urlopen(req) as res:
                    output_item = json.load(res)

                if output_item.get("qid") != qid:
                    raise ValueError("Invalid qid")

                if output_item.get("position") != position:
                    raise ValueError("Invalid position")

                pred_answer = output_item.get("prediction")
                if pred_answer is not None and not isinstance(pred_answer, str):
                    raise ValueError("Invalid prediction")

            except Exception as e:
                logger.error(e)
                pred_answer = None

            prediction_item = {"qid": qid, "position": position, "prediction": pred_answer}
            print(json.dumps(prediction_item, ensure_ascii=False), file=fo)


if __name__ == "__main__":
    main()
