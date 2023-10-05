import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, BatchEncoding


class ExtractiveReaderModel(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(self.encoder.config.hidden_size, 2)
        self.qa_classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, tokenized_inputs: BatchEncoding) -> tuple[Tensor, Tensor, Tensor]:
        encoder_outputs = self.encoder(**tokenized_inputs).last_hidden_state

        classifier_logits = self.qa_classifier(encoder_outputs[:, 0, :])
        start_logits, end_logits = self.qa_outputs(encoder_outputs).split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        return classifier_logits, start_logits, end_logits
