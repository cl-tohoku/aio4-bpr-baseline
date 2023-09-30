import numpy as np
import onnxruntime as ort
import scipy.special as sp

from transformers import BatchEncoding

from aio4_bpr_baseline.models.reader.extractive_reader.tokenization import ReaderTokenizer


class ReaderPredictionOnnxModule:
    def __init__(
        self,
        reader_onnx_file: str,
        base_model_name: str = "bert-base-uncased",
        max_input_length: int = 350,
        max_answer_length: int = 10,
    ) -> None:
        self.reader_session = ort.InferenceSession(reader_onnx_file, providers=["CPUExecutionProvider"])
        self.tokenizer = ReaderTokenizer(base_model_name, max_input_length=max_input_length)
        self.max_answer_length = max_answer_length

    def predict_answers(
        self,
        questions: list[str],
        passage_titles: list[list[str]],
        passage_texts: list[list[str]],
    ) -> list[dict[str, str | float]]:
        tokenized_inputs, passage_mask, span_mask = self.tokenizer(
            questions, passage_titles, passage_texts, return_tensors="np"
        )
        predictions = self._predict_answers_from_tokenized_inputs(tokenized_inputs, passage_mask, span_mask)
        return predictions

    def _predict_answers_from_tokenized_inputs(
        self, tokenized_inputs: BatchEncoding, passage_mask: np.ndarray, span_mask: np.ndarray
    ) -> list[dict[str, str | float]]:
        input_ids = tokenized_inputs["input_ids"]
        num_questions, _, _ = input_ids.shape

        classifier_logits, start_logits, end_logits = self.reader_session.run(None, dict(tokenized_inputs))

        classifier_log_probs = sp.log_softmax(np.where(passage_mask, classifier_logits, -1e4), axis=1)
        selected_input_idxs = classifier_log_probs.argmax(axis=1)
        pred_classifier_log_probs = np.take_along_axis(classifier_log_probs, selected_input_idxs[:, None], axis=1)

        selected_input_ids = np.take_along_axis(input_ids, selected_input_idxs[:, None, None], axis=1)[:, 0, :]
        selected_span_mask = np.take_along_axis(span_mask, selected_input_idxs[:, None, None], axis=1)[:, 0, :]
        selected_start_logits = np.take_along_axis(start_logits, selected_input_idxs[:, None, None], axis=1)[:, 0, :]
        selected_end_logits = np.take_along_axis(end_logits, selected_input_idxs[:, None, None], axis=1)[:, 0, :]

        pred_answer_starts, pred_answer_ends, pred_span_log_probs = self._get_pred_answer_spans(
            selected_start_logits, selected_end_logits, selected_span_mask
        )

        predictions: list[dict[str, str | float]] = []

        for i in range(num_questions):
            start = pred_answer_starts[i]
            end = pred_answer_ends[i]

            pred_answer = self.tokenizer.decode(
                selected_input_ids[i].tolist(),
                selected_span_mask[i].tolist(),
                start=start,
                end=end,
                extend_subwords=True,
            )
            score = float(np.exp(pred_classifier_log_probs[i] + pred_span_log_probs[i]))

            predictions.append({"pred_answer": pred_answer, "score": score})

        return predictions

    def _get_pred_answer_spans(
        self, start_logits: np.ndarray, end_logits: np.ndarray, span_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_inputs, max_input_length = start_logits.shape

        start_log_probs = sp.log_softmax(np.where(span_mask, start_logits, -1e4), axis=1)
        end_log_probs = sp.log_softmax(np.where(span_mask, end_logits, -1e4), axis=1)

        span_log_probs = start_log_probs[:, :, None] + end_log_probs[:, None, :]
        span_matrix_mask = np.ones((max_input_length, max_input_length), dtype=int)
        span_matrix_mask = np.triu(span_matrix_mask)
        span_matrix_mask = np.tril(span_matrix_mask, self.max_answer_length - 1)

        span_log_probs = span_log_probs.reshape(num_inputs, max_input_length * max_input_length)
        span_matrix_mask = span_matrix_mask.reshape(1, max_input_length * max_input_length)

        masked_span_log_probs = np.where(span_matrix_mask, span_log_probs, -1e4)
        selected_span_idxs = masked_span_log_probs.argmax(axis=1)
        pred_span_log_probs = np.take_along_axis(masked_span_log_probs, selected_span_idxs[:, None], axis=1)
        pred_answer_starts = selected_span_idxs // max_input_length
        pred_answer_ends = selected_span_idxs % max_input_length
        assert (pred_answer_starts <= pred_answer_ends).all()
        assert (0 < pred_answer_ends - pred_answer_starts + 1 <= self.max_answer_length).all()

        return pred_answer_starts, pred_answer_ends, pred_span_log_probs
