def get_value(score, value):
    if value == "rouge1":
        return score["rouge1"][1][2]
    if value == "rouge2":
        return score["rouge2"][1][2]
    if value == "rougeL":
        return score["rougeL"][1][2]
    if value == "accuracy":
        try:
            return score["accuracy"]
        except:
            return score["overall_accuracy"]
    if value == "precision":
        return score["overall_precision"]
    if value == "recall":
        return score["overall_recall"]
    if value == "f1":
        return score["overall_f1"]
    if value == "kendalltau":
        return score["tau"]
    if value == "pmr":
        return score["pmr"]
    try:
        return score[value]
    except:
        raise ValueError(
            f"{value} not in the metric output and correspondant value not defined in metrics_utils.py. You should change {value} to a value in {score.keys()} or define a condition in metrics_utils.py"
        )
