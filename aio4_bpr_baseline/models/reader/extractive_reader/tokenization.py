from typing import Any

from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer


class ReaderTokenizer:
    def __init__(self, base_model_name: str, max_input_length: int = 350) -> None:
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.max_input_length = max_input_length

    def __call__(
        self,
        questions: list[str],
        passage_titles: list[list[str]],
        passage_texts: list[list[str]],
        return_tensors: str = "pt",
    ) -> tuple[BatchEncoding, Any, Any]:
        num_questions = len(questions)
        if len(passage_titles) != num_questions:
            raise ValueError(f"len(questions) != len(passage_titles) ({num_questions} != {len(passage_titles)})")
        if len(passage_texts) != num_questions:
            raise ValueError(f"len(questions) != len(passage_texts) ({num_questions} != {len(passage_texts)})")

        max_passages = 0
        for i in range(num_questions):
            if len(passage_titles[i]) != len(passage_texts[i]):
                raise ValueError(
                    f"len(passage_titles[{i}]) != len(passage_texts[{i}]) ({passage_titles[i]} != {passage_texts[i]})"
                )

            max_passages = max(len(passage_titles[i]), max_passages)

        passage_mask: list[list[int]] = []
        for i in range(num_questions):
            num_passages = len(passage_titles[i])
            num_dummy_passages = max_passages - num_passages

            passage_titles[i] += [""] * num_dummy_passages
            passage_texts[i] += [""] * num_dummy_passages
            passage_mask.append([1] * num_passages + [0] * num_dummy_passages)

        questions_repeated = [question for each_question in questions for question in [each_question] * max_passages]
        passages_flattened = [
            title + self.tokenizer.sep_token + text
            for titles, texts in zip(passage_titles, passage_texts) for title, text in zip(titles, texts)
        ]
        passage_titles_flattened = [title for titles in passage_titles for title in titles]
        assert len(questions_repeated) == num_questions * max_passages
        assert len(passages_flattened) == num_questions * max_passages
        assert len(passage_titles_flattened) == num_questions * max_passages

        tokenized_inputs = self.tokenizer(
            questions_repeated,
            passages_flattened,
            padding=True,
            truncation="only_second",
            max_length=self.max_input_length,
            return_tensors=return_tensors,
        )
        input_ids = tokenized_inputs["input_ids"]

        short_input_ids = self.tokenizer(
            questions_repeated,
            passage_titles_flattened,
            padding="max_length",
            truncation="only_second",
            max_length=input_ids.shape[1],
            return_tensors=return_tensors,
        )["input_ids"]

        span_mask = (input_ids != self.tokenizer.pad_token_id) * (short_input_ids == self.tokenizer.pad_token_id)

        if return_tensors == "pt":
            import torch

            tokenized_inputs = BatchEncoding({
                key: tensor.view(num_questions, max_passages, -1) for key, tensor in tokenized_inputs.items()
            })
            span_mask = span_mask.view(num_questions, max_passages, -1)
            passage_mask = torch.tensor(passage_mask).bool()
        else:
            import numpy as np

            tokenized_inputs = BatchEncoding({
                key: array.reshape(num_questions, max_passages, -1) for key, array in tokenized_inputs.items()
            })
            span_mask = span_mask.reshape(num_questions, max_passages, -1)
            passage_mask = np.array(passage_mask, dtype=bool)

        return tokenized_inputs, passage_mask, span_mask

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

        if answer_length > max_answer_length:
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
            while start - 1 >= 0 and span_mask[start - 1] and self.is_subword_id(input_ids[start]):
                start = start - 1
            while end + 1 < len(input_ids) and span_mask[end + 1] and self.is_subword_id(input_ids[end + 1]):
                end = end + 1

        return self.tokenizer.decode(input_ids[start : end + 1])

    def is_subword_id(self, token_id: int) -> bool:
        return self.tokenizer.convert_ids_to_tokens([token_id])[0].startswith("##")
