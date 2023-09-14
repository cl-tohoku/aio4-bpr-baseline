import faiss
import numpy as np
import onnxruntime as ort

from transformers import BatchEncoding

from aio4_bpr_baseline.models.retriever.bpr.tokenization import QuestionEncoderTokenizer


class RetrieverPredictionOnnxModule:
    def __init__(
        self,
        question_encoder_onnx_file: str,
        passage_faiss_index_file: str,
        base_model_name: str = "bert-base-uncased",
        max_question_length: int = 256,
    ) -> None:
        self.question_encoder_session = ort.InferenceSession(question_encoder_onnx_file)
        self.question_tokenizer = QuestionEncoderTokenizer(base_model_name, max_question_length=max_question_length)

        self.passage_faiss_index = faiss.read_index_binary(passage_faiss_index_file)

    def retrieve_passages(
        self, questions: list[str], k: int = 100, num_candidates: int = 1000,
    ) -> list[dict[str, list[int] | list[float]]]:
        tokenized_questions = self.question_tokenizer(questions, return_tensors="np")
        predictions = self._retrieve_passages_from_tokenized_questions(
            tokenized_questions, k=k, num_candidates=num_candidates
        )
        return predictions

    def _retrieve_passages_from_tokenized_questions(
        self, tokenized_questions: BatchEncoding, k: int = 100, num_candidates: int = 1000,
    ) -> list[dict[str, list[int] | list[float]]]:
        if num_candidates > self.passage_faiss_index.ntotal:
            raise ValueError(
                "num_candidates is larger than the number of items in passage_faiss_index "
                f"({num_candidates} > {self.passage_faiss_index.ntotal})."
            )
        if k > num_candidates:
            raise ValueError(f"k is larger than num_candidates ({k} > {num_candidates}).")

        num_questions, _ = tokenized_questions["input_ids"].shape

        encoded_questions = self.question_encoder_session.run(None, dict(tokenized_questions))[0]
        binary_encoded_questions = np.packbits(np.where(encoded_questions >= 0, 1, 0), axis=-1)

        _, candidate_passage_ids = self.passage_faiss_index.search(binary_encoded_questions, num_candidates)

        binary_encoded_candidate_passages = np.vstack(
            [self.passage_faiss_index.reconstruct(i) for i in candidate_passage_ids.flatten().tolist()]
        ).reshape(num_questions, num_candidates, -1)
        binary_encoded_candidate_passages = (
            np.unpackbits(binary_encoded_candidate_passages, axis=2).astype(np.float32) * 2.0 - 1.0
        )
        candidate_passage_scores = np.einsum("ijk,ik->ij", binary_encoded_candidate_passages, encoded_questions)
        topk_positions = np.argsort(-candidate_passage_scores, axis=1)[:, :k]

        topk_passage_ids = np.take_along_axis(candidate_passage_ids, topk_positions, axis=1)
        topk_passage_scores = np.take_along_axis(candidate_passage_scores, topk_positions, axis=1)

        batch_predictions = []
        for passage_ids, scores in zip(topk_passage_ids.tolist(), topk_passage_scores.tolist()):
            prediction = {"passage_ids": passage_ids, "scores": scores}
            batch_predictions.append(prediction)

        return batch_predictions
