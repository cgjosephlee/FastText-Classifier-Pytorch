from tqdm.auto import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def train(model, dataloader):
    model.train()
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), model.config.lr)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=model.config.epoch * len(dataloader))
    optimizer = torch.optim.Adam(model.parameters(), model.config.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=0, total_iters=model.config.epoch
    )

    EPOCH_BEG = 1
    for epoch in range(EPOCH_BEG, model.config.epoch + 1):
        running_loss = 0.0
        for data in (pbar := tqdm(dataloader, desc=f"[epoch {epoch:>2}]")):
            # pbar.set_postfix({"lr": scheduler.get_last_lr()[0]})
            labels, input_ids = data
            optimizer.zero_grad()
            output = model(input_ids=input_ids)
            loss = criterion(output, labels)
            pbar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            # scheduler.step()  # update by batch
        print(
            f"""[epoch {epoch:>2}] train loss: {running_loss/len(dataloader):.3f} lr: {scheduler.get_last_lr()[0]:.3f}"""
        )
        scheduler.step()  # update by epoch


def test(model, dataloader, disable_progress=True):
    # model.evalute()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        running_loss = 0.0
        labels_true = []
        labels_pred = []
        labels_prob = []
        for data in tqdm(dataloader, disable=disable_progress):
            labels, input_ids = data
            output = model(input_ids=input_ids)
            loss = criterion(output, labels)
            running_loss += loss.item()
            output_pred = torch.max(nn.functional.softmax(output, 1), 1)
            labels_pred += output_pred[1].tolist()
            labels_prob += output_pred[0].tolist()
            labels_true += labels.tolist()
    acc = accuracy_score(labels_true, labels_pred)
    p, r, f, _ = precision_recall_fscore_support(
        labels_true, labels_pred, average="weighted", zero_division=0
    )  # WAF1-scores
    out = {
        "loss": running_loss / len(dataloader),
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f,
    }
    return out


def evalute(model, dataloader, disable_progress=True):
    # model.evalute()
    with torch.no_grad():
        labels_pred = []
        labels_prob = []
        for data in tqdm(dataloader, disable=disable_progress):
            _, input_ids = data
            output = model(input_ids=input_ids)
            output_pred = torch.max(nn.functional.softmax(output, 1), 1)
            labels_pred += output_pred[1].tolist()
            labels_prob += output_pred[0].tolist()
    out = {"labels_pred": labels_pred, "labels_prob": labels_prob}
    return out
