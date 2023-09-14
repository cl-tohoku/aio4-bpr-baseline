from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer


class QuestionEncoderTokenizer:
    def __init__(self, base_model_name: str, max_question_length: int = 256):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.max_question_length = max_question_length

    def __call__(self, questions: list[str], return_tensors: str = "pt") -> BatchEncoding:
        tokenized_questions = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_question_length,
            return_tensors=return_tensors,
        )
        return tokenized_questions


class PassageEncoderTokenizer:
    def __init__(self, base_model_name: str, max_passage_length: int = 256):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.max_passage_length = max_passage_length

    def __call__(
        self, passage_titles: list[str], passage_texts: list[str], return_tensors: str = "pt"
    ) -> BatchEncoding:
        tokenized_passages = self.tokenizer(
            passage_titles,
            passage_texts,
            padding=True,
            truncation="only_second",
            max_length=self.max_passage_length,
            return_tensors=return_tensors,
        )
        return tokenized_passages
