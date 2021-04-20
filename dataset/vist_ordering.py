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
"""Visual Storytelling Dataset (VIST) ordering dataset."""

from __future__ import absolute_import, division, print_function

import json
import os

import datasets
import numpy as np

import json

import pathlib

_CITATION = """
@inproceedings{huang2016visual,
  title={Visual Storytelling},
  author={Huang, Ting-Hao K. and Ferraro, Francis and Mostafazadeh, Nasrin and Misra, Ishan and Devlin, Jacob and Agrawal, Aishwarya and Girshick, Ross and He, Xiaodong and Kohli, Pushmeet and Batra, Dhruv and others},
  booktitle={15th Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2016)},
  year={2016}
}
"""

_DESCRIPTION = """
We introduce the first dataset for sequential vision-to-language, and explore how this data may be used for the task of visual storytelling. The dataset includes 81,743 unique photos in 20,211 sequences, aligned to descriptive and story language. VIST is previously known as "SIND", the Sequential Image Narrative Dataset (SIND). 
"""


_PATH = "dataset/vist/"

_SENTENCES = "sentences"
_SHUFFLED_SENTENCES = "shuffled_sentences"
_LABEL = "label"


class VISTOrdering(datasets.GeneratorBasedBuilder):
    """VIST ordering dataset."""

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
            homepage="http://visionandlanguage.net/VIST/dataset.html",
            citation=_CITATION,
        )
        return info

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_path = os.path.join(pathlib.Path().absolute(), _PATH)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": os.path.join(data_path, "train.story-in-sequence.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"path": os.path.join(data_path, "val.story-in-sequence.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"path": os.path.join(data_path, "test.story-in-sequence.json")},
            ),
        ]

    def _generate_examples(self, path=None):
        """Yields examples."""
        with open(path, "r") as f:
            data = json.load(f)
            annotations = data["annotations"]
            for i in range(0, len(annotations), 5):
                sentences = []
                order = []
                album_id = []
                for j in range(i, i + 5):
                    annotation = annotations[j][0]
                    sentences.append(annotation["original_text"])
                    order.append(annotation["worker_arranged_photo_order"])
                    album_id.append(annotation["album_id"])

                assert len(sentences) == 5
                assert order == [0, 1, 2, 3, 4]
                assert len(set(album_id)) == 1

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
