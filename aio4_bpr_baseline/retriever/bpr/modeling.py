import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, BatchEncoding


class BPREncoderModel(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name, add_pooling_layer=False)

    def forward(self, tokenized_inputs: BatchEncoding) -> Tensor:
        return self.encoder(**tokenized_inputs).last_hidden_state[:, 0, :]
