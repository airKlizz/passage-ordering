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
""" Perfect Match Ratio. """

import absl  # Here to have a nice missing dependency error message early on
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import six  # Here to have a nice missing dependency error message early on

_CITATION = """
"""

_DESCRIPTION = """\
Calculate the Perfect Match ratio which calculates the ratio of cases of exact match of
the whole sequence.
"""

_KWARGS_DESCRIPTION = """
Calculate the Perfect Match ratio.
Args:
    predictions: list of predictions to score. Each predictions
        should be a list of list of rankings.
    references: list of reference for each prediction. Each predictions
        should be a list of list of rankings.
Returns:
    ratio: PMR score
"""


class PMR(datasets.Metric):
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
            codebase_urls=[""],
            reference_urls=[""],
        )

    def _compute(self, predictions, references):
        result = {"pmr": np.array([])}

        for prediction, reference in zip(predictions, references):
            if prediction == reference:
                result["pmr"] = np.append(result["pmr"], 1)
            else:
                result["pmr"] = np.append(result["pmr"], 0)

        result["pmr"] = result["pmr"].mean()
        return result
