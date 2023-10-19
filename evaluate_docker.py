import argparse
import json
import logging
from pathlib import Path
from time import sleep


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_unlabelded_file", type=str, required=True)
    parser.add_argument("--output_prediction_file", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.test_unlabelded_file) as f:
        test_items = [json.loads(line) for line in f]

    with open(args.output_prediction_file, "w") as fo:
        for i, item in enumerate(test_items, start=1):
            qid: str = item["qid"]
            position: int = item["position"]

            logger.info(f"Processing qid = {qid}, position = {position} ({i}/{len(test_items)})")

            filename = f"{qid}-{position:04d}.json"
            input_path = Path(args.input_dir, filename)
            output_path = Path(args.output_dir, filename)

            with open(input_path, "wt") as fi:
                json.dump(item, fi, ensure_ascii=False)

            while input_path.exists():
                sleep(1)

            try:
                with open(output_path) as f:
                    output_item = json.load(f)

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

            output_path.unlink()


if __name__ == "__main__":
    main()
