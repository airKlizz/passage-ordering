from pathlib import Path
import pandas as pd
import numpy as np

from datasets import load_dataset

DATASET_PATH = "dataset/vist_ordering.py"
RESULTS_PATH = "results/vist"


def pretty(csv_path):
    return csv_path.name.replace(".csv", "").replace(" ", "_").lower()


class Explorer(object):
    def __init__(self, dataset_path, results_path):
        self.dataset = load_dataset(dataset_path, split="test")
        results_path = Path(results_path)
        self.dfs = {"{}".format(pretty(csv_path)): pd.read_csv(csv_path) for csv_path in results_path.glob("*.csv")}

    def get_results_key(self, key):
        df = self.dfs[key]
        return df["tau"].to_numpy().mean(), df["pmr"].to_numpy().mean()

    def get_results_keys(self):
        results = {
            "tau": {name: df["tau"].to_numpy().mean() for name, df in self.dfs.items()},
            "pmr": {name: df["pmr"].to_numpy().mean() for name, df in self.dfs.items()},
        }
        return pd.DataFrame(data=results)

    def get_comparison(self):
        comparison = {
            name1: {
                name2: {
                    "mean": np.abs(df1["tau"].to_numpy() - df2["tau"].to_numpy()).mean(),
                    "std": np.abs(df1["tau"].to_numpy() - df2["tau"].to_numpy()).std(),
                }
                for name2, df2 in self.dfs.items()
                if name2
            }
            for name1, df1 in self.dfs.items()
            if name1
        }
        return (
            np.array([[v2["mean"] for v2 in v1.values()] for v1 in comparison.values()]),
            np.array([[v2["std"] for v2 in v1.values()] for v1 in comparison.values()]),
        )

    def split_idxs(self, keys=["bart_simple", "bart_multi"]):
        assert len(keys) == 2, "split_idxs can compare only 2 keys"
        idxs1 = []
        idxs2 = []
        idxs3 = []
        for (idx, row1), (idx2, row2) in zip(self.dfs[keys[0]].iterrows(), self.dfs[keys[1]].iterrows()):
            assert idx == idx2
            if row1["tau"] > row2["tau"]:
                idxs1.append(idx)
            elif row1["tau"] < row2["tau"]:
                idxs2.append(idx)
            else:
                idxs3.append(idx)
        return idxs1, idxs2, idxs3

    def display_examples(self, idxs, keys=["bart_simple", "bart_multi"]):
        for idx in idxs:
            self.display_example(idx, keys)
            i = input("Press Enter to continue...")
            if i == "q":
                break

    def display_example(self, idx, keys=["bart_simple", "bart_multi"]):
        example = self.dataset[idx]
        print("\nPassages to order:")
        for p in example["shuffled_sentences"]:
            print(p)
        print()
        print("Gold order: {}".format(example["label"]))

        for key, df in self.dfs.items():
            if key not in keys:
                continue
            row = df.iloc[idx]
            print(f"\n{key} prediction:")
            print("---\nKendall's tau {:.4f} - PMR {:.4f}\n---".format(row["tau"], row["pmr"]))
            # for p in row["predicted_order"].strip('][').split(', '):
            #    print(example["shuffled_sentences"][int(p)])
            print(row["predicted_order"].strip("][").split(", "))

        print("~~~~~~~~~~")


"""

def order(example, simple=simple, multi=multi):
    sentences = example["shuffled_sentences"]
    simple_prediction = simple(sentences)
    multi_prediction = multi(sentences)
    simple_kendall = kendall.compute(predictions=[simple_prediction], references=[example["label"]])["tau"]
    multi_kendall = kendall.compute(predictions=[multi_prediction], references=[example["label"]])["tau"]
    simple_pmr = pmr.compute(predictions=[simple_prediction], references=[example["label"]])["pmr"]
    multi_pmr = pmr.compute(predictions=[multi_prediction], references=[example["label"]])["pmr"]
    return {"tau": [simple_kendall, multi_kendall], "pmr": [simple_pmr, multi_pmr]}, simple_prediction, multi_prediction, simple_kendall != multi_kendall
	

def display(example, simple=simple, multi=multi):
    results, simple_prediction, multi_prediction, to_display = order(example, simple, multi)
    if not to_display:
        return
    print("\n\n----------")
    print("Sentences to order:")
    for s in example["shuffled_sentences"]: print(s)
    print("")
    print("             \t| Simple \t| Multi")
    simple_tau = results["tau"][0]
    multi_tau = results["tau"][1]
    simple_pmr = results["pmr"][0]
    multi_pmr = results["pmr"][1]
    print(f"Kendall Tau \t| {simple_tau:.4f} \t| {multi_tau:.4f}")
    print(f"PMR         \t| {simple_pmr:.4f} \t| {simple_pmr:.4f}")
    print()
    print("Simple order:")
    for l in simple_prediction: print(example["shuffled_sentences"][l])
    print()
    print("Multi order:")
    for l in multi_prediction: print(example["shuffled_sentences"][l])
    print()
    print("Gold order:")
    for l in example["label"]: print(example["shuffled_sentences"][l])

for i in range(10):
   display(dataset[i])

"""
