from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
from datasets.splits import Split


@dataclass
class ClassArguments:
    """
    Arguments to init the TokenOrderingTrainer class.
    """

    max_length: int = field(metadata={"help": "Number max of tokens"})

    with_title: bool = field(metadata={"help": "True if the title is put in the decoder input"})

    wandb_project: str = field(metadata={"help": "Name of the wandb project"})
    wandb_run_name: str = field(metadata={"help": "Name of the wandb run"})

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    model_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file_path: Optional[str] = field(
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        metadata={"help": "Path for cached valid dataset"},
    )


@dataclass
class DatasetArguments:
    """
    Arguments to load the dataset.
    """

    path: str = field(
        metadata={"help": "Path to the dataset processing script with the dataset builder"},
    )
    max_training_examples: int = field(default=None)
    max_validation_examples: int = field(default=None)
    ignore_verifications: bool = field(default=None)
    save_infos: bool = field(default=None)
    name: Optional[str] = field(default=None)
    version: Optional[str] = field(default=None)
    data_dir: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
