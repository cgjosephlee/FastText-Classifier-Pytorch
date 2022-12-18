from dataclasses import dataclass
import torch
import torch.nn as nn
# import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import numpy as np


@dataclass
class FastTextClassifierConfig:
    vocab_size: int = 0
    min_count: int = 1
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


class FastTextClassifier(pl.LightningModule):
    def __init__(self, config: FastTextClassifierConfig):
        super(FastTextClassifier, self).__init__()
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.embedding = nn.Embedding(
            config.vocab_size + config.bucket, config.dim, padding_idx=0
        )
        self.fc1 = nn.Linear(config.dim, config.num_classes)
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, input_ids):
        ntokens = torch.count_nonzero(input_ids)
        output = self.embedding(input_ids)
        output = torch.sum(output, 1) / ntokens.view([-1, 1])
        output = self.fc1(output)
        return output

    def step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        output = self(input_ids=input_ids)
        loss = self.criterion(output, labels)
        return output, loss

    def training_step(self, batch, batch_idx):
        output, loss = self.step(batch, batch_idx)
        self.log("train:loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output, loss = self.step(batch, batch_idx)
        # labels_true = batch["label"].numpy()
        # labels_pred = output.numpy().argmax(axis=1)
        return {
            "loss": loss,
            "labels_true": batch["label"],
            "labels_pred": output.argmax(dim=1)
        }

    def validation_epoch_end(self, outputs):
        self.log()
        # a = accuracy_score(labels_true, labels_pred)
        # p, r, f, _ = precision_recall_fscore_support(
        #     labels_true, labels_pred, average="weighted", zero_division=0
        # )
        # self.log_dict({
        #     "loss": loss,
        #     "accuracy": a,
        #     "precision": p,
        #     "recall": r,
        #     "f1": f,
        #     })

    def test_step():
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer, start_factor=1, end_factor=0, total_iters=self.config.epoch
        # )
        # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        # return [optimizer], [scheduler]
