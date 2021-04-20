import argparse

from training.scripts.bart import BartForSequenceOrderingTrainer
from training.scripts.bart_deep import BartForSequenceOrderingWithDeepPointerTrainer
from training.scripts.bart_multi import BartForSequenceOrderingWithMultiPointerTrainer
from training.scripts.baseline import BaselineForSequenceOrderingTrainer


def main(model, args_file):
    if model == "default":
        BartForSequenceOrderingTrainer.train(args_file)
    elif model == "deep":
        BartForSequenceOrderingWithDeepPointerTrainer.train(args_file)
    elif model == "multi":
        BartForSequenceOrderingWithMultiPointerTrainer.train(args_file)
    elif model == "baseline":
        BaselineForSequenceOrderingTrainer.train(args_file)
    else:
        raise ValueError("Head not implemented. Choose between default, deep and multi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--args_file", help="Path to the args json file", type=str)
    parser.add_argument(
        "--model",
        help="Type of the Poiter Head on top of Bart. Can be default, deep or multi",
        type=str,
        default="default",
    )
    args = parser.parse_args()

    main(args.model, args.args_file)
