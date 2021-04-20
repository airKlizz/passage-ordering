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


def substitution(X, Y):

    assert len(X) == len(Y)

    permutation = {}
    for i, x in enumerate(X):
        permutation[x] = i
    for i in range(len(Y)):
        Y[i] = permutation[Y[i]]

    return Y


def get_nb_inv(X):

    nb_inv = 0
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if X[i] > X[j]:
                nb_inv += 1

    return nb_inv


def kendalltau(X, Y):

    new_Y = substitution(X, Y)
    nb_inv = get_nb_inv(new_Y)
    n = len(X)
    assert len(Y) == n, (X, Y)
    assert n != 0, (X, Y)
    binomial_coefficient = n * (n - 1) / 2
    if binomial_coefficient == 0:
        binomial_coefficient = 1
    tau = 1 - 2 * nb_inv / binomial_coefficient

    return tau


class KendallTauByHand(datasets.Metric):
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
