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
"""ROCStories Corpora ordering dataset."""

from __future__ import absolute_import, division, print_function

import json
import os

import datasets
import numpy as np

import csv

import pathlib

_CITATION = """
@inproceedings{mostafazadeh-etal-2016-corpus,
    title = "A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories",
    author = "Mostafazadeh, Nasrin  and
      Chambers, Nathanael  and
      He, Xiaodong  and
      Parikh, Devi  and
      Batra, Dhruv  and
      Vanderwende, Lucy  and
      Kohli, Pushmeet  and
      Allen, James",
    booktitle = "Proceedings of the 2016 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2016",
    address = "San Diego, California",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N16-1098",
    doi = "10.18653/v1/N16-1098",
    pages = "839--849",
}
"""

_DESCRIPTION = """
"""

_PATH = "dataset/rocstory/"

_SENTENCES = "sentences"
_SHUFFLED_SENTENCES = "shuffled_sentences"
_LABEL = "label"


class ROCOrdering(datasets.GeneratorBasedBuilder):
    """ROCStory ordering dataset."""

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
            homepage="https://cs.rochester.edu/datasets/rocstories/",
            citation=_CITATION,
        )
        return info

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_path = os.path.join(pathlib.Path().absolute(), _PATH)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": os.path.join(data_path, "ROCStory.train.csv")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"path": os.path.join(data_path, "ROCStory.validation.csv")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"path": os.path.join(data_path, "ROCStory.test.csv")},
            ),
        ]

    def _generate_examples(self, path=None):
        """Yields examples."""
        with open(path, "r") as f:
            csv_reader = csv.reader(f, delimiter=",")
            for i, elems in enumerate(csv_reader):
                if len(elems) != 7:
                    continue
                sentences = elems[-5:]

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
