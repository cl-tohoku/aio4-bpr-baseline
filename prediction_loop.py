import json
import logging
from pathlib import Path
from time import sleep

import torch
from datasets import Dataset

from aio4_bpr_baseline.retriever.bpr.lightning_modules import BPRRetrieverLightningModule
from aio4_bpr_baseline.reader.extractive_reader.lightning_modules import ExtractiveReaderPredictLightningModule
from aio4_bpr_baseline.utils.data import PASSAGES_FEATURES


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger()


BIENCODER_CKPT_FILE = "/work/biencoder.ckpt"
READER_CKPT_FILE = "/work/reader.ckpt"
PASSAGE_EMBEDDINGS_FILE = "/work/passage_embeddings.npy"
PASSAGES_FILE = "/work/passages.json.gz"

INPUT_DIR = "/input"
OUTPUT_DIR = "/output"


class BPRPipeline:
    def __init__(
        self,
        biencoder_ckpt_file: str,
        reader_ckpt_file: str,
        passage_embeddings_file: str,
        passages_file: str,
        device: str = "cuda",
    ):
        self.retriever_module = BPRRetrieverLightningModule(biencoder_ckpt_file, passage_embeddings_file)
        self.retriever_module.to(device)

        self.reader_predict_module = ExtractiveReaderPredictLightningModule(reader_ckpt_file)
        self.reader_predict_module.to(device)

        self.all_passages = Dataset.from_json(passages_file, features=PASSAGES_FEATURES)

    def predict_answer(
        self,
        question: str,
        num_passages: int = 10,
        num_candidates: int = 1000,
        answer_score_threshold: float = 0.5,
    ) -> dict[str, str | float]:
        retriever_prediction = self.retriever_module.retrieve_passages(
            [question], num_passages=num_passages, num_candidates=num_candidates
        )[0]
        passages = [self.all_passages[passage_info["idx"]] for passage_info in retriever_prediction]
        passage_titles = [passage["title"] for passage in passages]
        passage_texts = [passage["text"] for passage in passages]

        reader_prediction = self.reader_predict_module.predict_answer(question, passage_titles, passage_texts)
        score = reader_prediction["score"]
        if score >= answer_score_threshold:
            pred_answer = reader_prediction["pred_answer"]
        else:
            pred_answer = None

        return {"pred_answer": pred_answer, "score": score}


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

logger.info(f"Loading BPRPipeline (device: {device})")
pipeline = BPRPipeline(
    biencoder_ckpt_file=BIENCODER_CKPT_FILE,
    reader_ckpt_file=READER_CKPT_FILE,
    passage_embeddings_file=PASSAGE_EMBEDDINGS_FILE,
    passages_file=PASSAGES_FILE,
    device=device,
)
logger.info("Finished loading BPRPipeline")

while True:
    input_files = list(Path(INPUT_DIR).iterdir())
    logger.info("Input files: %s", [input_file.name for input_file in input_files])

    for input_file in input_files:
        logger.info("Processing %s", input_file.name)

        input_item = json.load(open(input_file))

        qid = input_item["qid"]
        position = input_item["position"]
        question = input_item["question"]

        prediction = pipeline.predict_answer(question)
        output = {"qid": qid, "position": position, "prediction": prediction["pred_answer"]}

        output_file = Path(OUTPUT_DIR) / input_file.name
        json.dump(output, open(output_file, "w"), ensure_ascii=False)

        logger.info("Removing %s", input_file.name)
        input_file.unlink()

    sleep(1.0)
