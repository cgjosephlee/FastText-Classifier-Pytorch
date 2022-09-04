# from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from .encoder import FastTextEncoder
from .model import FastTextClassifierConfig, FastTextClassifier
from .utils import train, test


def _tokenize(s):
    return s.split()


def collate_batch(batch):
    label_list = torch.LongTensor([x["label"] for x in batch])
    out = tokenizer(
        [_tokenize(x["text"]) for x in batch], return_tensors="pt", ft_mode=True
    )
    input_ids = out["input_ids"]
    return label_list.to(device), input_ids.to(device)


dataset_name = "ag_news"
train_iter = load_dataset(dataset_name, split="train")
test_iter = load_dataset(dataset_name, split="test")

config = FastTextClassifierConfig(
    num_classes=4,
    batch_size=256,
    lr=0.5,
    min_n=2,
    max_n=6,
    word_ngrams=2,
    dim=10,
    bucket=10000,
)

train_corpus = [_tokenize(x) for x in train_iter["text"]]
tokenizer = FastTextEncoder(train_corpus, config=config)
config.vocab_size = tokenizer.vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainloader = DataLoader(
    train_iter, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch
)
testloader = DataLoader(
    test_iter, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch
)

model = FastTextClassifier(config)
model.to(device)

train()

test(testloader)
