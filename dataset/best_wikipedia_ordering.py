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
"""wikipedia ordering dataset."""

from __future__ import absolute_import, division, print_function

import json
import os

import datasets
import numpy as np

from statistics import mean

import pathlib

_CITATION = """
"""

_DESCRIPTION = """
Dataset for sentence ordering using introduction from the best wikipedia articles."""

_PATH = "dataset/best_wikipedia/"

_SENTENCES = "sentences"
_SHUFFLED_SENTENCES = "shuffled_sentences"
_LABEL = "label"


class BestWikipediaOrdering(datasets.GeneratorBasedBuilder):
    """wikipedia ordering dataset from introduction."""

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
            homepage="",
            citation=_CITATION,
        )
        return info

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_path = os.path.join(pathlib.Path().absolute(), _PATH)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": os.path.join(data_path, "train.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"path": os.path.join(data_path, "valid.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"path": os.path.join(data_path, "test.json")},
            ),
        ]

    def _generate_examples(self, path=None):
        """Yields examples."""
        with open(path, "r") as f:
            i = -1
            for line in f:
                article = json.loads(line)
                text = article["Introduction"]
                sentences = text.split("\n\n")
                sentences = [sentence.replace("\n", "") for sentence in sentences if sentence.strip() != ""]

                shuffled_sentences, label = self.shuffle_sentences(sentences)
                i += 1
                yield i, {
                    _SENTENCES: sentences,
                    _SHUFFLED_SENTENCES: shuffled_sentences,
                    _LABEL: label,
                }

    def shuffle_sentences(self, sentences):
        sentences = np.array(sentences)
        permutation = np.random.permutation(len(sentences))
        return sentences[permutation].tolist(), np.argsort(permutation).tolist()
