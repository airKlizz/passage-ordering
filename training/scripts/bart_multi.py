try:
    from sequence_ordering_trainer import SequenceOrderingTrainer
except:
    from ..sequence_ordering_trainer import SequenceOrderingTrainer

from dataclasses import dataclass
from typing import Dict, List

import torch
from training.scripts.models.bart_multi import BartForSequenceOrderingWithMultiPointer
from transformers import BartTokenizerFast


class BartForSequenceOrderingWithMultiPointerTrainer(SequenceOrderingTrainer):
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
        self.tokenizer = BartTokenizerFast.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            cache_dir=model_cache_dir,
        )
        self.model = BartForSequenceOrderingWithMultiPointer.from_pretrained(
            model_name_or_path,
            cache_dir=model_cache_dir,
        )

    @staticmethod
    def data_collator(batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example["input_ids"] for example in batch])
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        decoder_input_ids = torch.stack([example["decoder_input_ids"] for example in batch])
        decoder_attention_mask = torch.stack([example["decoder_attention_mask"] for example in batch])
        labels = torch.stack([example["labels"] for example in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }
