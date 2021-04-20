from datasets import load_dataset
from nltk.tokenize import word_tokenize
import math

from tqdm import tqdm

import numpy as np


def describe(l):
    l = np.array(l)
    return {
        "min": l.min(),
        "max": l.max(),
        "mean": l.mean(),
        "10th percentile": np.percentile(l, 10),
        "30th percentile": np.percentile(l, 30),
        "50th percentile": np.percentile(l, 50),
        "70th percentile": np.percentile(l, 70),
        "90th percentile": np.percentile(l, 90),
    }


def random_pmr(l):
    p = 0
    for e in l:
        if e != 0:
            p += 1 / math.factorial(e)
    return p / len(l)


def main(path, cache_dir, data_dir=None):
    dataset = load_dataset(path, split="train+test+validation", cache_dir=cache_dir, data_dir=data_dir)
    num_sequences_per_example = []
    num_words_per_sequence = []
    for example in tqdm(dataset):
        num_sequences_per_example.append(len(example["sentences"]))
        for sentence in example["sentences"]:
            num_words_per_sequence.append(len(word_tokenize(sentence)))
    return describe(num_sequences_per_example), describe(num_words_per_sequence), random_pmr(num_sequences_per_example)
