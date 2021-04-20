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
""" Kendall's tau from SciPy. """

import absl  # Here to have a nice missing dependency error message early on
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import six  # Here to have a nice missing dependency error message early on
from scipy.stats import kendalltau

_CITATION = """
"""

_DESCRIPTION = """\
Calculate Kendall’s tau, a correlation measure for ordinal data.

Kendall’s tau is a measure of the correspondence between two rankings. 
Values close to 1 indicate strong agreement, values close to -1 indicate strong disagreement. 
This is the 1945 “tau-b” version of Kendall’s tau [2], which can account for ties and which reduces to the 1938 “tau-a” version [1] in absence of ties.
This metrics is a wrapper around SciPy implementation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
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
    pvalue: The two-sided p-value for a hypothesis test whose null hypothesis is an absence of association, tau = 0.
"""


class KendallTauScipy(datasets.Metric):
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
            codebase_urls=["https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html"],
            reference_urls=["https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient"],
        )

    def _compute(self, predictions, references, initial_lexsort=None, nan_policy="propagate", method="auto"):
        result = {"tau": np.array([]), "pvalue": np.array([])}

        for prediction, reference in zip(predictions, references):
            tau, pvalue = kendalltau(
                x=prediction, y=reference, initial_lexsort=initial_lexsort, nan_policy=nan_policy, method=method
            )
            result["tau"] = np.append(result["tau"], tau)
            result["pvalue"] = np.append(result["pvalue"], pvalue)

        result["tau"] = result["tau"].mean()
        result["pvalue"] = result["pvalue"].mean()
        return result
