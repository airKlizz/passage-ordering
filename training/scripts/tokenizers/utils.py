from nltk.tokenize import word_tokenize

from datasets import load_dataset

from .glove import GloveTokenizer

from tqdm import tqdm


def get_ordering_dataset(path, split="train", cache_dir=None):
    def add_text_column(example):
        example["text"] = " ".join(example["sentences"])
        return example

    dataset = load_dataset(path, split=split, cache_dir=cache_dir)
    dataset = dataset.map(add_text_column)
    return dataset


def get_vocab(dataset, columns, **load_kwargs):
    texts = []
    vocab = set()
    for column in columns:
        texts += dataset[column]
    for text in tqdm(texts, desc="Build the vocabulary"):
        words = word_tokenize(text)
        for word in words:
            vocab.add(word.lower())
    return vocab


def create_tokenizer_for_ordering_dataset(
    dataset_path, cache_dir, path, glove_file_path, max_length=128, max_num_seq=12
):
    dataset = get_ordering_dataset(dataset_path, cache_dir=cache_dir)
    vocab = get_vocab(dataset, ["text"])
    tok = GloveTokenizer.create(glove_file_path, vocab, max_length, max_num_seq)
    tok.save_pretrained(path)
