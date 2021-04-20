import dataclasses
import logging
import os
import sys

import datasets
import numpy as np
import torch

from transformers import DataCollator, HfArgumentParser, Trainer, TrainingArguments, set_seed

try:
    import wandb
except:
    pass

try:
    from dataclass_utils import ClassArguments, DataTrainingArguments, DatasetArguments
except:
    from .dataclass_utils import ClassArguments, DataTrainingArguments, DatasetArguments


logger = logging.getLogger(__name__)


def use_wandb():
    return "wandb" in sys.modules


class SequenceOrderingTrainer(object):
    def __init__(
        self,
        max_length,
        with_title,
        wandb_project,
        wandb_run_name,
    ):
        self.max_length = max_length
        self.with_title = with_title

        self.optimizers = (None, None)

        if use_wandb():
            wandb.init(name=wandb_run_name, project=wandb_project, reinit=True)

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
        self.train_dataset = self.train_dataset.map(self.convert_to_features, batched=True, batch_size=100000)
        self.valid_dataset = self.valid_dataset.map(self.convert_to_features, batched=True, batch_size=100000)

        # set the tensor type and the columns which the dataset should return
        columns = ["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"]
        self.train_dataset.set_format(type="torch", columns=columns)
        self.valid_dataset.set_format(type="torch", columns=columns)

    def convert_to_features(self, example_batch):

        encoder = [" </s><s> ".join(sentences) + " </s><s>" for sentences in example_batch["shuffled_sentences"]]
        if not self.with_title:
            decoder = [" </s><s> " + " </s><s> ".join(sentences) for sentences in example_batch["sentences"]]
        elif self.with_title:
            decoder = [
                title + " " + section_title + " </s><s> " + " </s><s> ".join(sentences)
                for title, section_title, sentences in zip(
                    example_batch["title"], example_batch["section_title"], example_batch["sentences"]
                )
            ]
        else:
            raise ValueError("Error in with_title argument")
        labels = [label + [len(label)] for label in example_batch["label"]]

        encoder_inputs = self.tokenizer(
            encoder,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        decoder_inputs = self.tokenizer(
            decoder,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        encoder_sequence_idx = [
            ((ids == self.tokenizer.eos_token_id).nonzero()).squeeze().tolist() for ids in encoder_inputs["input_ids"]
        ]
        decoder_sequence_idx = [
            ((ids == self.tokenizer.eos_token_id).nonzero()).squeeze().tolist() for ids in decoder_inputs["input_ids"]
        ]

        assert len(encoder_sequence_idx) == len(decoder_sequence_idx)

        bsz = len(labels)
        # Default labels is -100 to ignore index (See https://pytorch.org/docs/stable/nn.html#crossentropyloss)
        extend_labels = torch.ones((bsz, self.max_length), dtype=torch.long) * -100
        for b_idx in range(bsz):
            i = 0
            for d_idx in decoder_sequence_idx[b_idx]:
                try:
                    extend_labels[b_idx, d_idx] = encoder_sequence_idx[b_idx][labels[b_idx][i]]
                except:
                    pass
                i += 1

        encodings = {
            "input_ids": encoder_inputs["input_ids"].tolist(),
            "attention_mask": encoder_inputs["attention_mask"].tolist(),
            "decoder_input_ids": decoder_inputs["input_ids"].tolist(),
            "decoder_attention_mask": decoder_inputs["attention_mask"].tolist(),
            "labels": extend_labels.tolist(),
        }
        return encodings

    @classmethod
    def train(cls, args_json_filename):
        parser = HfArgumentParser((ClassArguments, DatasetArguments, DataTrainingArguments, TrainingArguments))
        class_args, dataset_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(args_json_filename)
        )

        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info("Start training")
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
        )
        logger.info("Training/evaluation parameters %s", training_args)

        # Set seed
        set_seed(training_args.seed)

        ordering_trainer = cls(**dataclasses.asdict(class_args))

        logger.info("Load and process dataset")
        ordering_trainer.load_and_process_data(**dataclasses.asdict(dataset_args))
        logger.info("Dataset ready")
        # Initialize our Trainer
        hf_trainer = Trainer(
            model=ordering_trainer.model,
            args=training_args,
            train_dataset=ordering_trainer.train_dataset,
            eval_dataset=ordering_trainer.valid_dataset,
            data_collator=cls.data_collator,
            prediction_loss_only=True,
            optimizers=ordering_trainer.optimizers,
        )

        # Training
        if training_args.do_train:
            hf_trainer.train(
                model_path=class_args.model_name_or_path if os.path.isdir(class_args.model_name_or_path) else None
            )
            hf_trainer.save_model()
            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            if hf_trainer.is_world_master():
                ordering_trainer.tokenizer.save_pretrained(training_args.output_dir)

        # Evaluation
        results = {}
        if training_args.do_eval and training_args.local_rank in [-1, 0]:
            logger.info("*** Evaluate ***")

            eval_output = hf_trainer.evaluate()

            output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(eval_output.keys()):
                    logger.info("  %s = %s", key, str(eval_output[key]))
                    writer.write("%s = %s\n" % (key, str(eval_output[key])))

            results.update(eval_output)

        return None
