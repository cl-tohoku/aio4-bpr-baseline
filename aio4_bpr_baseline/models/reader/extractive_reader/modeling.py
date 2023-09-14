import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, BatchEncoding, PreTrainedModel


class ReaderModel(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(base_model_name, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(self.encoder.config.hidden_size, 2)
        self.qa_classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, tokenized_inputs: BatchEncoding) -> tuple[Tensor, Tensor, Tensor]:
        num_questions, max_passages, max_input_length = tokenized_inputs["input_ids"].size()

        tokenized_inputs = BatchEncoding(
            {
                key: tensor.view(num_questions * max_passages, max_input_length)
                for key, tensor in tokenized_inputs.items()
            }
        )
        encoder_outputs = self.encoder(**tokenized_inputs).last_hidden_state

        classifier_logits = self.qa_classifier(encoder_outputs[:, 0]).view(num_questions, max_passages)
        start_logits, end_logits = self.qa_outputs(encoder_outputs).split(1, dim=-1)
        start_logits = start_logits.view(num_questions, max_passages, max_input_length)
        end_logits = end_logits.view(num_questions, max_passages, max_input_length)

        return classifier_logits, start_logits, end_logits
