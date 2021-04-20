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
""" Kendall's tau by hand. """

import absl  # Here to have a nice missing dependency error message early on
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import six  # Here to have a nice missing dependency error message early on

_CITATION = """
https://github.com/shrimai/Topological-Sort-for-Sentence-Ordering/blob/master/topological_sort.py#L91-L105
"""

_DESCRIPTION = """
"""

_KWARGS_DESCRIPTION = """
Calculate Kendall’s tau, a correlation measure for ordinal data.
Args:
    predictions: list of predictions to score. Each predictions
        should be a list of list of rankings.
    references: list of reference for each prediction. Each predictions
        should be a list of list of rankings.
Returns:
    tau: The tau statistic,
"""


def kendalltau(X, Y):
    """
    It calculates the number of inversions required by the predicted
    order to reach the correct order.
    """
    pred_pairs, gold_pairs = [], []
    for i in range(len(Y)):
        for j in range(i + 1, len(Y)):
            pred_pairs.append((Y[i], Y[j]))
            gold_pairs.append((X[i], X[j]))
    common = len(set(pred_pairs).intersection(set(gold_pairs)))
    uncommon = len(gold_pairs) - common
    tau = 1 - (2 * (uncommon / len(gold_pairs)))

    return tau


class KendallTauShrimai(datasets.Metric):
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
            codebase_urls=[
                "https://github.com/shrimai/Topological-Sort-for-Sentence-Ordering/blob/master/topological_sort.py#L91-L105"
            ],
            reference_urls=["https://www.aclweb.org/anthology/J06-4002/"],
        )

    def _compute(self, predictions, references):
        result = {"tau": np.array([])}

        for prediction, reference in zip(predictions, references):
            tau = kendalltau(X=prediction, Y=reference)
            result["tau"] = np.append(result["tau"], tau)

        result["tau"] = result["tau"].mean()
        return result
