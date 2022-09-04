from collections import Counter
from gensim.models.fasttext_inner import (
    compute_ngrams,
    compute_ngrams_bytes,
    ft_hash_bytes,
)
import torch
import pickle

# from collections.abc import Iterable  # python >= 3.9
from typing import Iterable  # python < 3.9
from typing import Union, Optional, List, Dict
from .model import FastTextClassifierConfig

DEFAULT_RESERVED_TOKENS = ["<pad>", "<unk>", "</s>"]
DEFAULT_PAD_INDEX = 0
DEFAULT_UNK_INDEX = 1
DEFAULT_EOS_INDEX = 2


class FastTextEncoder:
    """
    FastText endocer with char ngram and word ngram implementations.
    """

    def __init__(
        self,
        texts: Union[Iterable[List[str]], List[List[str]]],
        config: Optional[FastTextClassifierConfig] = None,
        min_count: int = 1,
        # max_vocab_size = None,
        min_n: int = 0,
        max_n: int = 0,
        word_ngrams: int = 1,
        bucket: int = 2000000,
    ):
        """
        Build tokenizer. Require segmented sentences.

        config: used to import parameters, overwrites other arguments
        """
        # defaults
        self.reserved_tokens = DEFAULT_RESERVED_TOKENS
        self.pad_index = DEFAULT_PAD_INDEX
        self.unk_index = DEFAULT_UNK_INDEX
        self.eos_index = DEFAULT_EOS_INDEX

        self.min_count = min_count
        self.min_n = min_n
        self.max_n = max_n
        self.word_ngrams = word_ngrams
        self.bucket = bucket

        if config is not None:
            self.min_count = config.min_count
            self.min_n = config.min_n
            self.max_n = config.max_n
            self.word_ngrams = config.word_ngrams
            self.bucket = config.bucket

        self.tokens = Counter()

        for n, sequence in enumerate(texts):
            self.tokens.update(
                [x.strip() for x in sequence if x.strip() not in ("", None)]
            )

        self.corpus_count = n + 1
        self.corpus_total_words = sum(self.tokens.values())

        self.index_to_token = self.reserved_tokens.copy()
        self.token_to_index = {
            token: index for index, token in enumerate(self.reserved_tokens)
        }
        for token, count in self.tokens.items():
            if count >= self.min_count:
                self.index_to_token.append(token)
                self.token_to_index[token] = len(self.index_to_token) - 1

        self.__initNgrams()

        # release memory
        self.tokens = Counter()

    @property
    def vocab(self) -> List[str]:
        return self.index_to_token

    @property
    def vocab_size(self) -> int:
        return len(self.index_to_token)

    @property
    def parameters(self) -> Dict:
        out = {
            "min_count": self.min_count,
            "min_n": self.min_n,
            "max_n": self.max_n,
            "word_ngrams": self.word_ngrams,
            "bucket": self.bucket,
            "vocab_size": self.vocab_size,
        }
        return out

    def __initNgrams(self):
        """
        Initialize char ngrams for all vocabularies.
        """
        self.index_to_ngram = [[]] * len(
            self.index_to_token
        )  # caveats: all items are actually same object
        if self.max_n >= self.min_n and self.min_n > 0 and self.bucket > 0:
            for n, v in enumerate(self.index_to_token):
                # exclude preserved words
                if n >= len(self.reserved_tokens):
                    self.index_to_ngram[n] = self._compute_ngram_hashes(
                        v, self.min_n, self.max_n, self.bucket
                    )

    def decode(self, ids: List[int]) -> List[str]:
        vector = [self.index_to_token[i] for i in ids]
        return vector

    def encode(
        self,
        sequence: List[str],
        append_eos: bool = False,
        unk_to_zero: bool = False,
        remove_unk: bool = False,
    ) -> List[int]:
        vector = [self.token_to_index.get(token, self.unk_index) for token in sequence]
        if append_eos:
            vector.append(self.eos_index)
        if unk_to_zero:
            vector = [self.pad_index if x == self.unk_index else x for x in vector]
        if remove_unk:
            # but we will need correct length to do ngram hashing
            vector = [x for x in vector if x != self.unk_index]
        return vector

    def encode_ngram(self, sequence: List[str], input_ids: List[int]) -> List[int]:
        # char ngram
        char_ngrams = []
        for w, i in zip(sequence, input_ids):
            if i != self.eos_index:
                if i != self.unk_index:
                    char_ngrams += self.index_to_ngram[i]
                else:
                    # oov
                    char_ngrams += self._compute_ngram_hashes(
                        w, self.min_n, self.max_n, self.bucket
                    )

        # word ngram
        # we do not care oov, just hash it
        hashes = [self._hash(x.encode("UTF-8")) for x in sequence]
        word_ngrams = self._compute_wordNgram_hashes(
            hashes, self.word_ngrams, self.bucket
        )
        return char_ngrams + word_ngrams

    def _encode_ft(self, sequence: List[str]) -> List[int]:
        """
        FastText-like output.
        Ignore UNK. Ngram id goes after end of wid.
        """
        list_wids = []
        list_hashes = []
        list_ngrams = []  # here we separate this out
        for w in sequence:
            wid = self.token_to_index.get(w, self.unk_index)
            h = self._hash(w.encode("UTF-8"))
            list_hashes.append(h)
            if wid != self.unk_index:
                list_wids.append(wid)
                list_ngrams += self.index_to_ngram[wid]
            else:
                # oov
                list_ngrams += self._compute_ngram_hashes(
                    w, self.min_n, self.max_n, self.bucket
                )
        # word ngrams
        list_ngrams += self._compute_wordNgram_hashes(
            list_hashes, self.word_ngrams, self.bucket
        )

        list_ngrams = [x + self.vocab_size for x in list_ngrams]
        return list_wids + list_ngrams

    def _batch_encode(
        self,
        texts: Union[Iterable[List[str]], List[List[str]]],
        # padding=False,
        # truncation=False,
        # max_length=None,
        return_tensors: Optional[str] = None,
        unk_to_zero: bool = False,
    ) -> Dict:
        """
        Separate wid and ngram.
        """
        input_ids = [self.encode(x, unk_to_zero=unk_to_zero) for x in texts]
        input_ngrams = [
            self.encode_ngram(x, y) for x, y in zip(texts, input_ids)
        ]  ### what if unk == 0???

        if return_tensors == "pt":
            input_ids = self._pad_sequence_pt(input_ids)
            input_ngrams = self._pad_sequence_pt(input_ngrams)

        output = {"input_ids": input_ids, "input_ngrams": input_ngrams}
        return output

    def _batch_encode_ft(
        self,
        texts: Union[Iterable[List[str]], List[List[str]]],
        # padding=False,
        # truncation=False,
        # max_length=None,
        return_tensors: Optional[str] = None,
    ) -> Dict:
        input_ids = [self._encode_ft(x) for x in texts]
        len_ids = [len(x) for x in input_ids]
        if return_tensors == "pt":
            input_ids = self._pad_sequence_pt(input_ids)
            len_ids = torch.IntTensor(len_ids)
        output = {"input_ids": input_ids, "len_ids": len_ids}
        return output

    def __call__(
        self,
        texts: Union[Iterable[List[str]], List[List[str]]],
        ft_mode: bool = False,
        **kwargs
    ) -> Dict:
        if ft_mode:
            return self._batch_encode_ft(texts, **kwargs)
        else:
            return self._batch_encode(texts, **kwargs)

    def __contains__(self, word):
        return word in self.vocab

    def get_vector(self, embedding: torch.nn.Embedding, word: str):
        """
        embedding: embedding weights (vocab_size * diim)
        """
        wid = self.token_to_index.get(word, self.unk_index)
        if wid != self.unk_index:
            return embedding.weight[wid]
        elif self.max_n >= self.min_n and self.min_n > 0 and self.bucket > 0:
            # oov and ngram is enabled
            ngrams = self._compute_ngram_hashes(
                word, self.min_n, self.max_n, self.bucket
            )
            if len(ngrams) == 0:
                return embedding.weight[0]  # PAD and it's zeros
            ngrams = [x + self.vocab_size for x in ngrams]
            return torch.mean(embedding.weight[ngrams], dim=0)
        else:
            raise KeyError("cannot calculate vector for OOV word without ngrams")

    def get_sentence_vector(self, embedding: torch.nn.Embedding, sentence: List[str]):
        input_ids = self._encode_ft(sentence)
        return torch.mean(embedding.weight[input_ids], dim=0)

    def save_vocab(self, fout):
        with open(fout, "wb") as f:
            pickle.dump(self.vocab, f)

    @classmethod
    def load_vocab(cls, fin, **kwargs):
        with open(fin, "rb") as f:
            vocab = pickle.load(f)
        tokenizer = cls(vocab, **kwargs)
        tokenizer.corpus_count = None
        tokenizer.corpus_total_words = None
        return tokenizer

    @classmethod
    def _hash(cls, bytez: bytes) -> int:
        return ft_hash_bytes(bytez)

    @classmethod
    def _compute_ngrams(cls, word: str, min_n, max_n) -> List[str]:
        if max_n >= min_n and min_n > 0:
            return compute_ngrams(word, min_n, max_n)
        else:
            return []

    @classmethod
    def _compute_ngrams_bytes(cls, word: str, min_n, max_n) -> List[bytes]:
        if max_n >= min_n and min_n > 0:
            return compute_ngrams_bytes(word, min_n, max_n)
        else:
            return []

    @classmethod
    def _compute_ngram_hashes(cls, word: str, min_n, max_n, bucket) -> List[int]:
        if max_n >= min_n and min_n > 0 and bucket > 0:
            hashes = cls._compute_ngrams_bytes(word, min_n, max_n)
            return [cls._hash(x) % bucket for x in hashes]
        else:
            return []

    @classmethod
    def _compute_wordNgram_hashes(
        cls, word_hashes: List[int], word_ngrams: int, bucket: int
    ) -> List[int]:
        # ref: https://github.com/facebookresearch/fastText/blob/a20c0d27cd0ee88a25ea0433b7f03038cd728459/src/dictionary.cc#L312
        wordNgram_hashes = []
        if bucket > 0:
            for i in range(len(word_hashes)):
                h = word_hashes[i]
                for j in range(i + 1, min(len(word_hashes), i + word_ngrams)):
                    h = h * 116049371 + word_hashes[j]
                    wordNgram_hashes.append(h % bucket)
        return wordNgram_hashes

    @classmethod
    def _pad_sequence_pt(cls, seq):
        return torch.nn.utils.rnn.pad_sequence(
            [torch.IntTensor(x) for x in seq], batch_first=True
        )
