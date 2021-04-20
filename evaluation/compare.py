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


def plot_tau_function_of_num_passages(predictions, references, scores):
    # Get stats for every example
    num_passages = [len(l) for l in references]
    unique_num_passages = sorted(list(set(num_passages)))

    scores_per_num_passages = {k: {n: [] for n in unique_num_passages} for k in predictions.keys()}
    for l, s in scores.items():
        assert len(num_passages) == len(s)
        for n, v in zip(num_passages, s):
            scores_per_num_passages[l][n].append(v)

    for l, s in scores_per_num_passages.items():
        data = list(s.values()).copy()
        scores_per_num_passages[l]["mean"] = [np.mean(d) for d in data]
        scores_per_num_passages[l]["std"] = [np.std(d) for d in data]

    # Plot kendall score in function of the num of passages for every head
    fig, ax = plt.subplots()

    for i, (l, data) in enumerate(scores_per_num_passages.items()):
        eb = ax.errorbar(x=unique_num_passages, y=data["mean"], yerr=data["std"], label=l)
        eb[-1][0].set_linestyle("--")

    plt.legend(fontsize="xx-small")
    plt.savefig("kendall_tau_function_of_num_passages.png", dpi=1000)


def plot_tau_function_of_num_words(predictions, ben, scores):
    num_words = [
        len(word_tokenize(" ".join(sentences))) for sentences in ben.dataset.dataset[ben.dataset.x_column_name[0]]
    ]
    num_words = [n - n % 50 + 25 for n in num_words]
    unique_num_words = sorted(list(set(num_words)))

    scores_per_num_words = {k: {n: [] for n in unique_num_words} for k in predictions.keys()}
    for l, s in scores.items():
        assert len(num_words) == len(s)
        for n, v in zip(num_words, s):
            scores_per_num_words[l][n].append(v)

    for l, s in scores_per_num_words.items():
        data = list(s.values()).copy()
        scores_per_num_words[l]["mean"] = [np.mean(d) for d in data]
        scores_per_num_words[l]["std"] = [np.std(d) for d in data]

    # Plot kendall score in function of the num of passages for every head
    fig, ax = plt.subplots()

    for i, (l, data) in enumerate(scores_per_num_words.items()):
        eb = ax.errorbar(x=unique_num_words, y=data["mean"], yerr=data["std"], label=l)
        eb[-1][0].set_linestyle("--")

    plt.legend(fontsize="xx-small")
    plt.savefig("kendall_tau_function_of_num_words.png", dpi=1000)


def compare_scores(scores):
    sns.set(font_scale=0.2)
    comparison_heads = {
        n1: {
            n2: {
                "mean": np.abs(np.array(scores[n1]) - np.array(scores[n2])).mean(),
                "std": np.abs(np.array(scores[n1]) - np.array(scores[n2])).std(),
            }
            for n2 in scores.keys()
        }
        for n1 in scores.keys()
    }

    labels = list(comparison_heads.keys())
    mean_data = [[v2["mean"] for v2 in v1.values()] for v1 in comparison_heads.values()]
    std_data = [[v2["std"] for v2 in v1.values()] for v1 in comparison_heads.values()]

    fig, axes = plt.subplots(1, 2)
    sns.heatmap(mean_data, ax=axes[0], xticklabels=labels, yticklabels=labels, annot=True)
    sns.heatmap(std_data, ax=axes[1], xticklabels=labels, yticklabels=labels, annot=True)
    plt.savefig("compare_scores.png", dpi=1000)


def main():
    ben = Benchmark.from_json("evaluation/args/best_wikipedia-test.json")
    kendall = load_metric("evaluation/metrics/kendalltau.py")
    ben.load()

    # Run predictions for all the scenarios
    predictions = {}
    for i in tqdm(range(len(ben.scenarios)), desc="Run scenarios"):
        s, p = ben.run_scenario(i, True)
        predictions[ben.scenarios[i].name] = p

    references = ben.dataset.dataset[ben.dataset.y_column_name]

    # Run kendall score for every example
    scores = {k: [] for k in predictions.keys()}
    for n, p in tqdm(predictions.items(), desc="Run Kendall"):
        for idx in range(len(p)):
            scores[n].append(
                kendall.compute(
                    predictions=[p[idx]],
                    references=[references[idx]],
                )["tau"]
            )

    key_to_keep = ["Baseline", "Bart Simple", "Bart Deep", "Bart Multi All Heads"]

    def apply_key_to_keep(d, key_to_keep=key_to_keep):
        return {k: v for k, v in d.items() if k in key_to_keep}

    plot_tau_function_of_num_passages(apply_key_to_keep(predictions), references, apply_key_to_keep(scores))
    plot_tau_function_of_num_words(apply_key_to_keep(predictions), ben, apply_key_to_keep(scores))
    compare_scores(scores)


if __name__ == "__main__":
    main()
