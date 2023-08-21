import json
from pathlib import Path
from time import sleep

import torch

from models.pipeline import BPRPipeline


INPUT_DIR = "/input"
OUTPUT_DIR = "/output"

RETRIEVER_CKPT_FILE = "/work/retriever.ckpt"
READER_CKPT_FILE = "/work/reader.ckpt"
PASSAGE_FAISS_INDEX_FILE = "/work/passages.faiss"
PASSAGE_DATASET_FILE = "/work/passages.json.gz"

ANSWER_SCORE_THRESHOLD = 0.1


@torch.inference_mode()
def predict_answer(pipeline: BPRPipeline, question: str) -> str | None:
    prediction = pipeline.predict_answers([question])[0]

    answer = prediction["answers"][0]
    score = prediction["scores"][0]

    if score > ANSWER_SCORE_THRESHOLD:
        return answer
    else:
        return None


def main():
    print("Loading BPRPipeline")

    pipeline = BPRPipeline(
        retriever_ckpt_file=RETRIEVER_CKPT_FILE,
        reader_ckpt_file=READER_CKPT_FILE,
        passage_faiss_index_file=PASSAGE_FAISS_INDEX_FILE,
        passage_dataset_file=PASSAGE_DATASET_FILE,
    )
    pipeline.eval()

    if torch.cuda.is_available():
        pipeline.to("cuda")
        print("Loaded BPRPipeline to GPU")
    else:
        print("Loaded BPRPipeline to CPU")

    while True:
        for input_file in Path(INPUT_DIR).iterdir():
            print("Processing", input_file.name)

            input_item = json.load(open(input_file))

            qid = input_item["qid"]
            position = input_item["position"]
            question = input_item["question"]

            predicted_answer = predict_answer(pipeline, question)

            output = {"qid": qid, "position": position, "prediction": predicted_answer}

            output_file = Path(OUTPUT_DIR) / input_file.name
            json.dump(output, open(output_file, "w"), ensure_ascii=False)

            print("Removing", input_file.name)
            input_file.unlink()

        sleep(1.0)


if __name__ == "__main__":
    main()
