import torch
import torch.nn as nn

from tqdm import tqdm
import json
from pathlib import Path
import pickle

from nltk.tokenize import word_tokenize


class GloveTokenizer(object):
    def __init__(self, embedding, word2idx, emb_dim, cls, unk, pad, max_length=128, max_num_seq=12):
        self.embedding = embedding
        self.word2idx = word2idx
        self.emb_dim = emb_dim
        self.cls = cls
        self.unk = unk
        self.pad = pad
        self.max_length = max_length
        self.max_num_seq = max_num_seq

    def __call__(
        self,
        text,
        padding=False,
        truncation=False,
        max_length=None,
        seq_padding=False,
        seq_truncation=False,
        max_num_seq=None,
        return_tensors=None,
    ):
        if isinstance(text, str):
            tokens = [self.tokenize(text)]
        elif isinstance(text, list):
            tokens = [self.tokenize(t) for t in text]
        else:
            raise ValueError("text must be a string or a list of string")

        if max_length == None:
            max_length = self.max_length

        if max_num_seq == None:
            max_num_seq = self.max_num_seq

        if padding == True or padding == "longest":
            longest = max([len(t) for t in tokens])
            tokens = [t + [self.pad] * (longest - len(t)) for t in tokens]
        elif padding == "max_length":
            tokens = [t + [self.pad] * (max_length - len(t)) if max_length - len(t) >= 0 else t for t in tokens]

        if truncation == True or truncation == "longest_first":
            longest = min(max([len(t) for t in tokens]), max_length)
            tokens = [t[:longest] for t in tokens]

        if seq_padding == True or seq_padding == "max_num_seq":
            if len(tokens) < max_num_seq:
                tokens = tokens + [
                    [self.cls if i == 0 else self.pad for i in range(max_length)]
                    for _ in range(max_num_seq - len(tokens))
                ]

        if seq_truncation == True:
            tokens = tokens[:max_num_seq]

        mask = []
        for t in tokens:
            m = []
            for id in t:
                if id == self.pad:
                    m.append(0)
                else:
                    m.append(1)
            mask.append(m)

        if return_tensors == "pt":
            tokens = torch.tensor(tokens)
            mask = torch.tensor(mask)

        return {
            "input_ids": tokens,
            "attention_mask": mask,
        }

    def tokenize(self, text):
        assert isinstance(text, str), text
        words = word_tokenize(text)
        words = [w.lower() for w in words]
        tokens = [self.word2idx[w] if w in self.word2idx.keys() else self.unk for w in words]
        tokens = [self.cls] + tokens
        return tokens

    def get_emb_layer(self):
        return self.embedding

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            "max_length": self.max_length,
            "max_num_seq": self.max_num_seq,
            "emb_dim": self.emb_dim,
            "cls": self.cls,
            "unk": self.unk,
            "pad": self.pad,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

        torch.save(self.embedding.weight, path / "weight.pt")

        with open(path / "word2idx.pkl", "wb") as f:
            pickle.dump(self.word2idx, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def create(cls, glove_file_path, vocab, max_length, max_num_seq):
        embedding, word2idx, emb_dim, cls_tok, unk, pad = cls.create_emb_layer(glove_file_path, vocab)
        return cls(embedding, word2idx, emb_dim, cls_tok, unk, pad, max_length, max_num_seq)

    @classmethod
    def from_pretrained(cls, path):
        path = Path(path)

        with open(path / "config.json", "r") as f:
            config = json.load(f)

        weight = torch.load(path / "weight.pt")
        embedding = nn.Embedding.from_pretrained(weight)

        with open(path / "word2idx.pkl", "rb") as f:
            word2idx = pickle.load(f)
        return cls(
            embedding,
            word2idx,
            config["emb_dim"],
            config["cls"],
            config["unk"],
            config["pad"],
            config["max_length"],
            config["max_num_seq"],
        )

    @staticmethod
    def create_emb_layer(glove_file_path, vocab):

        vocab = list(vocab)
        idx = 0
        word2idx = {}
        vectors = []
        with open(glove_file_path, "r") as f:
            for l in tqdm(f):
                line = l.split()
                word = line[0]
                if word not in vocab:
                    continue
                word2idx[word] = idx
                idx += 1
                vect = list(map(float, line[1:]))
                vectors.append(vect)

        emb_dim = len(vectors[0])
        weights_matrix = torch.zeros((len(vocab), emb_dim))
        for i, word in enumerate(vocab):
            if word in word2idx.keys():
                weights_matrix[i] = torch.tensor(vectors[word2idx[word]])
            else:
                weights_matrix[i] = torch.empty((emb_dim,)).normal_(mean=0, std=0.6)

        # Add UNK token
        weights_matrix = torch.cat(
            (
                torch.empty(
                    (
                        3,
                        emb_dim,
                    )
                ).normal_(mean=0, std=0.6),
                weights_matrix,
            )
        )

        return nn.Embedding.from_pretrained(weights_matrix), {w: i + 3 for i, w in enumerate(vocab)}, emb_dim, 0, 2, 1
