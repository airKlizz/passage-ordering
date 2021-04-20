try:
    from sequence_ordering_trainer import SequenceOrderingTrainer
except:
    from ..sequence_ordering_trainer import SequenceOrderingTrainer

from dataclasses import dataclass
from typing import Dict, List

import torch

import datasets

from .models.hier_attn import HierarchicalAttentionNetworksForSequenceOrdering
from .tokenizers.glove import GloveTokenizer


class BaselineForSequenceOrderingTrainer(SequenceOrderingTrainer):
    def __init__(
        self,
        model_name_or_path,
        tokenizer_name,
        model_cache_dir,
        max_length,
        with_title,
        wandb_project,
        wandb_run_name,
        **kwargs,
    ):
        super().__init__(
            max_length,
            with_title,
            wandb_project,
            wandb_run_name,
        )
        self.tokenizer = GloveTokenizer.from_pretrained(tokenizer_name)
        self.model = HierarchicalAttentionNetworksForSequenceOrdering(
            embedding=self.tokenizer.get_emb_layer(),
            rnn_hidden_size=256,
            rnn_num_layers=1,
            num_attn_heads=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            ffn_dim=4096,
            dropout=0.15,
        )

    def load_and_process_data(self, **load_dataset_kwargs):
        max_training_examples = load_dataset_kwargs.pop("max_training_examples", None)
        max_validation_examples = load_dataset_kwargs.pop("max_validation_examples", None)
        train_split = "train" if max_training_examples == None else f"train[:{max_training_examples}]"
        validation_split = (
            "validation" if max_validation_examples == None else f"validation[:{max_validation_examples}]"
        )

        print(f"Train split: {train_split}")
        print(f"Validation split: {validation_split}")

        self.train_dataset = datasets.load_dataset(split=train_split, **load_dataset_kwargs)
        self.valid_dataset = datasets.load_dataset(split=validation_split, **load_dataset_kwargs)

        # map convert_to_features batch wise
        self.train_dataset = self.train_dataset.map(self.convert_to_features, batched=True)
        self.valid_dataset = self.valid_dataset.map(self.convert_to_features, batched=True)

        # set the tensor type and the columns which the dataset should return
        columns = ["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"]
        self.train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(self.train_dataset["input_ids"]),
            torch.tensor(self.train_dataset["attention_mask"]),
            torch.tensor(self.train_dataset["decoder_input_ids"]),
            torch.tensor(self.train_dataset["decoder_attention_mask"]),
            torch.tensor(self.train_dataset["labels"]),
        )
        self.valid_dataset = torch.utils.data.TensorDataset(
            torch.tensor(self.valid_dataset["input_ids"]),
            torch.tensor(self.valid_dataset["attention_mask"]),
            torch.tensor(self.valid_dataset["decoder_input_ids"]),
            torch.tensor(self.valid_dataset["decoder_attention_mask"]),
            torch.tensor(self.valid_dataset["labels"]),
        )

    def convert_to_features(self, example_batch):

        encoder_inputs = [
            self.tokenizer(
                sentences,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                max_num_seq=self.tokenizer.max_num_seq,
                seq_padding="max_num_seq",
                seq_truncation=True,
            )
            for sentences in example_batch["shuffled_sentences"]
        ]
        decoder_inputs = [
            self.tokenizer(
                ["start"] + sentences[:-1],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                max_num_seq=self.tokenizer.max_num_seq,
                seq_padding="max_num_seq",
                seq_truncation=True,
            )
            for sentences in example_batch["sentences"]
        ]

        input_ids = []
        attention_mask = []
        for elem in encoder_inputs:
            input_ids.append(elem["input_ids"])
            attention_mask.append(elem["attention_mask"])

        decoder_input_ids = []
        decoder_attention_mask = []
        for elem in decoder_inputs:
            decoder_input_ids.append(elem["input_ids"])
            decoder_attention_mask.append(elem["attention_mask"])

        labels = [label for label in example_batch["label"]]
        # pad labels with -100 to ignore during training
        pad_labels = []
        for label in labels:
            if len(label) >= self.tokenizer.max_num_seq:
                pad_labels.append([l for l in label if l < self.tokenizer.max_num_seq])
                assert len(pad_labels[-1]) == self.tokenizer.max_num_seq
            else:
                pad_labels.append(label + [-100 for _ in range(self.tokenizer.max_num_seq - len(label))])

        encodings = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": pad_labels,
        }
        return encodings

    @staticmethod
    def data_collator(batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example[0] for example in batch])
        attention_mask = torch.stack([example[1] for example in batch])
        decoder_input_ids = torch.stack([example[2] for example in batch])
        decoder_attention_mask = torch.stack([example[3] for example in batch])
        labels = torch.stack([example[4] for example in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }
