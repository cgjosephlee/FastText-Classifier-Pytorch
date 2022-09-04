from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class FastTextClassifierConfig:
    vocab_size: int = 0
    min_n: int = 2
    max_n: int = 5
    word_ngrams: int = 1
    dim: int = 100
    bucket: int = 2000000
    lr: float = 0.1
    lrUpdateRate: int = 100  # update lr by n tokens, here we update by batch
    num_classes: int = 1
    epoch: int = 5
    batch_size: int = 256


class FastTextClassifier(nn.Module):
    def __init__(self, config: FastTextClassifierConfig):
        super(FastTextClassifier, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.vocab_size + config.bucket, config.dim, padding_idx=0
        )
        self.fc1 = nn.Linear(config.dim, config.num_classes)

    def forward(self, input_ids):
        ntokens = torch.count_nonzero(input_ids)
        output = self.embedding(input_ids)
        output = torch.sum(output, 1) / ntokens.view([-1, 1])
        output = self.fc1(output)
        return output
