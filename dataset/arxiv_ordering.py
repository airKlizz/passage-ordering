# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace NLP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""arXiv ordering dataset."""

from __future__ import absolute_import, division, print_function

import json
import os

import datasets
import numpy as np

import pathlib

_CITATION = """
@misc{chen2016neural,
    title={Neural Sentence Ordering},
    author={Xinchi Chen and Xipeng Qiu and Xuanjing Huang},
    year={2016},
    eprint={1607.06952},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """
Dataset for sentence ordering using text from arXiv."""

_PATH = "dataset/arxiv/"

_SENTENCES = "sentences"
_SHUFFLED_SENTENCES = "shuffled_sentences"
_LABEL = "label"


class ArXivOrdering(datasets.GeneratorBasedBuilder):
    """arXiv ordering dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        info = datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    _SENTENCES: datasets.Sequence(datasets.Value("string")),
                    _SHUFFLED_SENTENCES: datasets.Sequence(datasets.Value("string")),
                    _LABEL: datasets.Sequence(datasets.Value("int8")),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/FudanNLP/NeuralSentenceOrdering",
            citation=_CITATION,
        )
        return info

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_path = os.path.join(pathlib.Path().absolute(), _PATH)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": os.path.join(data_path, "train.txt")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"path": os.path.join(data_path, "valid.txt")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"path": os.path.join(data_path, "test.txt")},
            ),
        ]

    def _generate_examples(self, path=None):
        """Yields examples."""
        with open(path, "r") as f:
            data = f.read()
            examples = data.split("\n\n")
            for i, example in enumerate(examples):
                lines = example.split("\n")
                sentences = lines[2:]
                if sentences == []:
                    continue
                shuffled_sentences, label = self.shuffle_sentences(sentences)
                yield i, {
                    _SENTENCES: sentences,
                    _SHUFFLED_SENTENCES: shuffled_sentences,
                    _LABEL: label,
                }

    def shuffle_sentences(self, sentences):
        sentences = np.array(sentences)
        permutation = np.random.permutation(len(sentences))
        return sentences[permutation].tolist(), np.argsort(permutation).tolist()
