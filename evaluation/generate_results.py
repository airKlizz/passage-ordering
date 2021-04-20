import numpy as np
from datasets import load_metric

try:
    from evaluation.benchmark import Benchmark
except:
    from benchmark import Benchmark

from nltk import word_tokenize

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import pandas as pd

from pathlib import Path


def main(ben, path):

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    kendall = load_metric("evaluation/metrics/kendalltau.py")
    ben.load()

    references = ben.dataset.dataset[ben.dataset.y_column_name]

    num_passages = [len(l) for l in references]
    num_words = [
        len(word_tokenize(" ".join(sentences))) for sentences in ben.dataset.dataset[ben.dataset.x_column_name[0]]
    ]
    assert len(num_passages) == len(num_words) and len(references) == len(num_words)

    # Run predictions for all the scenarios
    for i in tqdm(range(len(ben.scenarios)), desc="Run scenarios"):
        s, predictions = ben.run_scenario(i, True)
        tau = []
        pmr = []
        for idx in range(len(predictions)):
            tau.append(
                kendall.compute(
                    predictions=[predictions[idx]],
                    references=[references[idx]],
                )["tau"]
            )
            pmr.append(int(predictions[idx] == references[idx]))

        assert len(num_passages) == len(predictions) and len(num_passages) == len(tau)

        data = {
            "id": list(range(len(num_passages))),
            "true_order": references,
            "predicted_order": predictions,
            "tau": tau,
            "pmr": pmr,
            "num_passages": num_passages,
            "num_words": num_words,
        }
        df = pd.DataFrame(data=data)
        name = ben.scenarios[i].name
        filename = f"{name}.csv"
        df.to_csv(path / filename)


if __name__ == "__main__":
    main()
