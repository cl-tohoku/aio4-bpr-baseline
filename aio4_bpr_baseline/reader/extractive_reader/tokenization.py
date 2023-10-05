from typing import Sequence

from transformers import AutoTokenizer, BatchEncoding


class ExtractiveReaderTokenizer:
    def __init__(self, base_model_name: str, max_input_length: int = 350):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.max_input_length = max_input_length

    def __call__(
        self,
        questions: list[str],
        passage_titles: list[str],
        passage_texts: list[str],
        return_tensors: str = "pt",
    ) -> tuple[BatchEncoding, Sequence[Sequence[bool]]]:
        num_questions = len(questions)
        if len(passage_titles) != num_questions:
            raise ValueError(f"len(questions) != len(passage_titles) ({num_questions} != {len(passage_titles)})")
        if len(passage_texts) != num_questions:
            raise ValueError(f"len(questions) != len(passage_texts) ({num_questions} != {len(passage_texts)})")

        passages = [title + self.tokenizer.sep_token + text for title, text in zip(passage_titles, passage_texts)]

        tokenized_inputs = self.tokenizer(
            questions,
            passages,
            padding=True,
            truncation="only_second",
            max_length=self.max_input_length,
            return_tensors=return_tensors,
        )
        input_ids = tokenized_inputs["input_ids"]

        short_input_ids = self.tokenizer(
            questions,
            passage_titles,
            padding="max_length",
            truncation="only_second",
            max_length=input_ids.shape[1],
            return_tensors=return_tensors,
        )["input_ids"]

        span_mask = (input_ids != self.tokenizer.pad_token_id) * (short_input_ids == self.tokenizer.pad_token_id)

        return tokenized_inputs, span_mask

    def find_answer_spans(
        self,
        answer: str,
        input_ids: list[int],
        span_mask: list[bool] | None = None,
        max_answer_length: int | None = None,
    ) -> list[tuple[int, int]]:
        input_length = len(input_ids)

        if span_mask is None:
            span_mask = [True] * input_length

        answer_input_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        answer_length = len(answer_input_ids)

        if max_answer_length is not None and answer_length > max_answer_length:
            return []

        answer_spans: list[tuple[int, int]] = []
        for start in range(input_length - answer_length + 1):
            end = start + answer_length - 1
            if input_ids[start : end + 1] == answer_input_ids and all(span_mask[start : end + 1]):
                answer_spans.append((start, end))

        return answer_spans

    def decode(
        self,
        input_ids: list[int],
        span_mask: list[bool] | None = None,
        start: int | None = None,
        end: int | None = None,
        extend_subwords: bool = False,
    ) -> str:
        if span_mask is None:
            span_mask = [1] * len(input_ids)
        if start is None:
            start = 0
        if end is None:
            end = len(input_ids) - 1

        if extend_subwords:
            while start - 1 >= 0 and span_mask[start - 1] and self._is_subword_id(input_ids[start]):
                start = start - 1
            while end + 1 < len(input_ids) and span_mask[end + 1] and self._is_subword_id(input_ids[end + 1]):
                end = end + 1

        return self.tokenizer.decode(input_ids[start : end + 1])

    def _is_subword_id(self, token_id: int) -> bool:
        return self.tokenizer.convert_ids_to_tokens([token_id])[0].startswith("##")
