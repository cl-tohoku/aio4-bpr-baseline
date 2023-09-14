from datasets import Dataset

from aio4_bpr_baseline.models.reader.extractive_reader.onnx_modules import ReaderPredictionOnnxModule
from aio4_bpr_baseline.models.retriever.bpr.onnx_modules import RetrieverPredictionOnnxModule
from aio4_bpr_baseline.utils.data import PASSAGE_DATASET_FEATURES


class PipelineOnnxModule:
    def __init__(
        self,
        question_encoder_onnx_file: str,
        reader_onnx_file: str,
        passage_faiss_index_file: str,
        passage_dataset_file: str,
        question_encoder_base_model_name: str = "bert-base-uncased",
        question_encoder_max_question_length: int = 256,
        reader_base_model_name: str = "bert-base-uncased",
        reader_max_input_length: int = 350,
        reader_max_answer_length: int = 10,
        datasets_num_proc: int | None = None,
    ):
        self.retriever_module = RetrieverPredictionOnnxModule(
            question_encoder_onnx_file,
            passage_faiss_index_file,
            base_model_name=question_encoder_base_model_name,
            max_question_length=question_encoder_max_question_length,
        )
        self.reader_prediction_module = ReaderPredictionOnnxModule(
            reader_onnx_file,
            base_model_name=reader_base_model_name,
            max_input_length=reader_max_input_length,
            max_answer_length=reader_max_answer_length,
        )

        self.passage_dataset = Dataset.from_json(
            passage_dataset_file, features=PASSAGE_DATASET_FEATURES, num_proc=datasets_num_proc
        )

    def predict_answers(
        self,
        questions: list[str],
        retriever_k: int = 100,
        retriever_num_candidates: int = 1000,
        answer_score_threshold: float = 0.0,
    ) -> list[dict[str, str | float]]:
        retriever_predictions = self.retriever_module.retrieve_passages(
            questions, k=retriever_k, num_candidates=retriever_num_candidates
        )

        passage_titles = []
        passage_texts = []
        for retriever_prediction in retriever_predictions:
            passage_ids = retriever_prediction["passage_ids"]
            passages = [self.passage_dataset[passage_id] for passage_id in passage_ids]

            passage_titles.append([passage["title"] for passage in passages])
            passage_texts.append([passage["text"] for passage in passages])

        reader_predictions = self.reader_prediction_module.predict_answers(questions, passage_titles, passage_texts)

        output_predictions = []
        for reader_prediction in reader_predictions:
            score = reader_prediction["score"]
            if score >= answer_score_threshold:
                pred_answer = reader_prediction["pred_answer"]
            else:
                pred_answer = None

            output_predictions.append({"pred_answer": pred_answer, "score": score})

        return output_predictions
