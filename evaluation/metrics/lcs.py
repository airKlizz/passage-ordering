# coding=utf-8
# Copyright 2020 The HuggingFace NLP Authors.
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
""" ROUGE metric from Google Research github repo. """

# The dependencies in https://github.com/google-research/google-research/blob/master/rouge/requirements.txt
import absl  # Here to have a nice missing dependency error message early on
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy  # Here to have a nice missing dependency error message early on
import six  # Here to have a nice missing dependency error message early on
from rouge_score import rouge_scorer, scoring

_CITATION = """\
"""

_DESCRIPTION = """\
"""

_KWARGS_DESCRIPTION = """
Calculates average longest common subsequence
"""


class Lcs(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int8")),
                    "references": datasets.Sequence(datasets.Value("int8")),
                }
            ),
            codebase_urls=["https://github.com/google-research/google-research/tree/master/rouge"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/ROUGE_(metric)",
                "https://github.com/google-research/google-research/tree/master/rouge",
            ],
        )

    def _compute(self, predictions, references, use_agregator=True, use_stemmer=False):

        rouge_types = ["rougeL"]
        predictions = " ".join([str(p) for p in predictions])
        references = " ".join([str(r) for r in references])

        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
        if use_agregator:
            aggregator = scoring.BootstrapAggregator()
        else:
            scores = []

        for ref, pred in zip(references, predictions):
            score = scorer.score(ref, pred)
            if use_agregator:
                aggregator.add_scores(score)
            else:
                scores.append(score)

        if use_agregator:
            result = aggregator.aggregate()
        else:
            result = {}
            for key in scores[0]:
                result[key] = list(score[key] for score in scores)

        return result
