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
    num_classes: int = 2
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
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=config.num_classes, compute_on_cpu=True)
        self.val_prec = torchmetrics.Precision(task="multiclass", num_classes=config.num_classes, average="weighted", compute_on_cpu=True)
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=config.num_classes, average="weighted", compute_on_cpu=True)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=config.num_classes, average="weighted", compute_on_cpu=True)

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
        self.val_acc.update(output, batch["label"])
        self.val_prec.update(output, batch["label"])
        self.val_recall.update(output, batch["label"])
        self.val_f1.update(output, batch["label"])
        self.log("eval:acc", self.val_acc)
        self.log("eval:precision", self.val_prec)
        self.log("eval:recall", self.val_recall)
        self.log("eval:f1score", self.val_f1)

    test_step = validation_step

    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        output = self(input_ids=input_ids)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer, start_factor=1, end_factor=0, total_iters=self.config.epoch
        # )
        # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        # return [optimizer], [scheduler]
