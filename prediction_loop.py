import json
import logging
from pathlib import Path
from time import sleep

from aio4_bpr_baseline.models.pipeline_aio4.bpr.onnx_modules import PipelineOnnxModule


INPUT_DIR = "/input"
OUTPUT_DIR = "/output"

QUESTION_ENCODER_ONNX_FILE = "/work/question_encoder.onnx"
READER_ONNX_FILE = "/work/reader.onnx"
PASSAGE_FAISS_INDEX_FILE = "/work/passages.faiss"
PASSAGE_DATASET_FILE = "/work/passages.json.gz"

RETRIEVER_K = 10
ANSWER_SCORE_THRESHOLD = 0.3


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger()


def main():
    logger.info("Loading PipelineOnnxModule")

    pipeline = PipelineOnnxModule(
        QUESTION_ENCODER_ONNX_FILE,
        READER_ONNX_FILE,
        PASSAGE_FAISS_INDEX_FILE,
        PASSAGE_DATASET_FILE,
        question_encoder_base_model_name="cl-tohoku/bert-base-japanese-v3",
        reader_base_model_name="cl-tohoku/bert-base-japanese-v3",
    )

    logger.info("Finished loading PipelineOnnxModule")

    while True:
        input_files = list(Path(INPUT_DIR).iterdir())
        logger.info("Input files: %s", [input_file.name for input_file in input_files])

        for input_file in input_files:
            logger.info("Processing %s", input_file.name)

            input_item = json.load(open(input_file))

            qid = input_item["qid"]
            position = input_item["position"]
            question = input_item["question"]

            prediction = pipeline.predict_answers(
                [question], retriever_k=RETRIEVER_K, answer_score_threshold=ANSWER_SCORE_THRESHOLD
            )[0]

            output = {"qid": qid, "position": position, "prediction": prediction["pred_answer"]}

            output_file = Path(OUTPUT_DIR) / input_file.name
            json.dump(output, open(output_file, "w"), ensure_ascii=False)

            logger.info("Removing %s", input_file.name)
            input_file.unlink()

        sleep(5.0)


if __name__ == "__main__":
    main()
