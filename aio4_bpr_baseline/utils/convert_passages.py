import argparse
import json
import logging

from tqdm import tqdm

from aio4_bpr_baseline.utils.data import open_file


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger()


def make_passages(passages_file: str, output_passages_file: str, output_pid_idx_map_file: str):
    pid_idx_map = {}
    idx = 0

    logger.info(f"Reading {passages_file} and writing to {output_passages_file}")
    with open_file(passages_file) as f, open_file(output_passages_file, "wt") as fo:
        for line in tqdm(f):
            example = json.loads(line)

            pid = str(example["id"])
            title = example["title"]
            text = example["text"]

            output_example = {"idx": idx, "pid": pid, "title": title, "text": text}
            print(json.dumps(output_example, ensure_ascii=False), file=fo)

            pid_idx_map[pid] = idx
            idx += 1

    logger.info(f"Writing pid_idx_map to {output_pid_idx_map_file}")
    with open_file(output_pid_idx_map_file, "wt") as fo:
        json.dump(pid_idx_map, fo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages_file", type=str, required=True)
    parser.add_argument("--output_passages_file", type=str, required=True)
    parser.add_argument("--output_pid_idx_map_file", type=str, required=True)
    args = parser.parse_args()

    make_passages(args.passages_file, args.output_passages_file, args.output_pid_idx_map_file)


if __name__ == "__main__":
    main()
