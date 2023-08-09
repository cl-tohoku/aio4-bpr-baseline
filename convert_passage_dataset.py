import argparse
import json

from tqdm import tqdm

from utils.data import open_file


def main(args: argparse.Namespace):
    with open_file(args.passage_file) as f, open_file(args.output_dataset_file, "wt") as fo:
        for line in tqdm(f):
            example = json.loads(line)

            pid = str(example["id"])
            text = example["text"]
            title = example["title"]

            output_example = {
                "pid": pid,
                "text": text,
                "title": title,
            }
            print(json.dumps(output_example, ensure_ascii=False), file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--passage_file", type=str, required=True)
    parser.add_argument("--output_dataset_file", type=str, required=True)

    args = parser.parse_args()
    main(args)
