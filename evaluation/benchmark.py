import json
from dataclasses import dataclass, field
from importlib.machinery import SourceFileLoader
from inspect import getmembers, isclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
import pandas as pd

try:
    from evaluation.metrics_utils import get_value
    from evaluation.model import Model
    from evaluation.models.ordering import OrderingModel
    from evaluation.models.ordering_baseline import OrderingBaselineModel
    from evaluation.models.ordering_deep import OrderingModelDeep
    from evaluation.models.ordering_multi import OrderingModelMulti
except:
    from metrics_utils import get_value
    from model import Model
    from models.ordering import OrderingModel
    from models.ordering_baseline import OrderingBaselineModel
    from models.ordering_deep import OrderingModelDeep
    from models.ordering_multi import OrderingModelMulti

ALL_MODEL_CLASS = {
    "ordering": OrderingModel,
    "ordering deep": OrderingModelDeep,
    "ordering multi": OrderingModelMulti,
    "ordering baseline": OrderingBaselineModel,
}


def get_model_class(model_class: str):
    if model_class in ALL_MODEL_CLASS.keys():
        return ALL_MODEL_CLASS[model_class]
    elif Path(model_class).is_file() and model_class[-3:] == ".py":
        module = SourceFileLoader("", model_class).load_module()
        return [cls[1] for cls in getmembers(module, isclass) if cls[1].__bases__[0] == Model][0]
    else:
        raise ValueError(
            f"model_class {model_class} must be a model class from {ALL_MODEL_CLASS} or the path to a python script."
        )


@dataclass
class Scenario:
    """
    Class defining a scenario
    """

    name: str = field()
    model_class: str = field()
    model_name: str = field()
    tokenizer_name: Optional[Union[None, str]] = field(default=None)

    model: Union[None, Model] = field(default=None)

    init_kwargs: Optional[Dict] = field(default_factory=dict)

    device: str = field(default="cpu")
    batch_size: int = field(default=1)
    quantization: bool = False
    onnx: Union[bool, str] = field(default=False)
    onnx_convert_kwargs: Optional[Dict] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def __str__(self):
        return f"Scenario(name={self.name})"


@dataclass
class Metric:
    """
    Class defining a metric
    """

    metric_name: str = field()
    values: List[str] = field()
    init_kwargs: Optional[Dict] = field(default_factory=dict)
    run_kwargs: Optional[Dict] = field(default_factory=dict)
    metric: Union[None, datasets.Metric] = field(default=None)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class Dataset:
    """
    Class defining a dataset
    """

    dataset_name: str = field()
    split: str = field()
    x_column_name: str = field()
    y_column_name: str = field()
    init_kwargs: Optional[Dict] = field(default_factory=dict)
    dataset: Union[None, datasets.Dataset] = field(default=None)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class Benchmark(object):
    """
    Class containing all parameters and functions of an evaluation benchmark.
    """

    def __init__(self, scenarios: List[Scenario], dataset: Dataset, metrics: List[Metric]):
        self.scenarios = scenarios
        self.dataset = dataset
        self.metrics = metrics

    @classmethod
    def from_args(
        cls,
        dataset_name: str,
        dataset_split: str,
        x_column_name: str,
        y_column_name: str,
        metric_name: str,
        metric_values: List[str],
        dataset_init_kwargs: Dict = {},
        metric_init_kwargs: Dict = {},
        metric_run_kwargs: Dict = {},
        scenarios: List[Scenario] = [],
    ):
        return cls(
            scenarios=scenarios,
            dataset=Dataset(
                dataset_name=dataset_name,
                split=dataset_split,
                x_column_name=x_column_name,
                y_column_name=y_column_name,
                init_kwargs=dataset_init_kwargs,
            ),
            metrics=[
                Metric(
                    metric_name=metric_name,
                    values=metric_values,
                    init_kwargs=metric_init_kwargs,
                    run_kwargs=metric_run_kwargs,
                )
            ],
        )

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            scenarios=[Scenario.from_dict(scenario) for scenario in data["scenarios"]],
            dataset=Dataset.from_dict(data["dataset"]),
            metrics=[Metric.from_dict(metric) for metric in data["metrics"]],
        )

    @classmethod
    def from_json(cls, file: str):
        with open(file, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def delete_scenario(self, scenario_name: str):
        for idx, scenario in enumerate(self.scenarios):
            if scenario_name == scenario.name:
                self.scenarios.pop(idx)
                return
        raise ValueError(f"{scenario_name} not in scenarios")

    def reset_scenarios(self):
        self.scenarios = []

    def add_scenario(
        self,
        name: str,
        model_class: str,
        model_name: str,
        tokenizer_name: Union[None, str] = None,
        init_kwargs: Union[None, Dict] = None,
        batch_size: int = 1,
        device: str = "cpu",
        quantization: bool = False,
        onnx: Union[bool, str] = False,
        onnx_convert_kwargs: Union[None, Dict] = None,
    ):
        self.scenarios.append(
            Scenario(
                name=name,
                model_class=model_class,
                model_name=model_name,
                tokenizer_name=tokenizer_name,
                init_kwargs=init_kwargs,
                batch_size=batch_size,
                device=device,
                quantization=quantization,
                onnx=onnx,
                onnx_convert_kwargs=onnx_convert_kwargs,
            )
        )

    def edit_dataset(
        self,
        name: str,
        split: str,
        x_column_name: str,
        y_column_name: str,
        init_kwargs: Union[None, Dict] = None,
    ):
        self.dataset = Dataset(
            dataset_name=name,
            split=split,
            x_column_name=x_column_name,
            y_column_name=y_column_name,
            init_kwargs=init_kwargs,
        )

    def add_metric(
        self,
        name: str,
        values: List[str],
        init_kwargs: Union[None, Dict] = None,
        run_kwargs: Union[None, Dict] = None,
    ):
        self.metrics.append(
            Metric(
                metric_name=name,
                values=values,
                init_kwargs=init_kwargs,
                run_kwargs=run_kwargs,
            )
        )

    def load_dataset(self, force=False):
        if self.dataset.dataset != None and force == False:
            print("INFO: Dataset already load. Force to reload.")
            return
        self.dataset.dataset = datasets.load_dataset(
            self.dataset.dataset_name,
            split=self.dataset.split,
            **self.dataset.init_kwargs,
        )

    def load_metrics(self, force=False):
        if self.metrics == None:
            raise ValueError("Metrics is empty. No metric to load.")
        for metric in self.metrics:
            if metric.metric != None and force == False:
                print("INFO: Metric already load. Force to reload.")
                return
            metric.metric = datasets.load_metric(metric.metric_name, **metric.init_kwargs)

    def load_models(self, force=False):
        if self.scenarios == None:
            raise ValueError("Scenarios is empty. No model to load.")
        for scenario in self.scenarios:
            if scenario.model != None and force == False:
                print(f"INFO: Model of {scenario.name} already load. Force to reload.")
                continue
            scenario.model = get_model_class(scenario.model_class)(
                name=scenario.name,
                model_name=scenario.model_name,
                tokenizer_name=scenario.tokenizer_name,
                device=scenario.device,
                quantization=scenario.quantization,
                onnx=scenario.onnx,
                onnx_convert_kwargs=scenario.onnx_convert_kwargs,
                **scenario.init_kwargs,
            )

    def load(self, force=False):
        self.load_models(force)
        self.load_dataset(force)
        self.load_metrics(force)

    def run(self, force_reload=False, verbose=True):
        if verbose:
            print("INFO: Load models, dataset and metric.")
        self.load(force_reload)

        if verbose:
            print("INFO: Run scenarios stats.")
        all_stats = None
        if self.scenarios == None:
            raise ValueError("Scenarios is empty. No scenario to run.")
        for scenario in self.scenarios:
            if verbose:
                print(f"INFO: Run scenario: {scenario.name}.")
            stats = self.run_scenario(scenario.name)
            if verbose:
                print(f"INFO: Scenario result: {scenario.name}\n{stats}")
            if all_stats == None:
                all_stats = stats
            else:
                for k, v in all_stats.items():
                    v.update(stats[k])

        return pd.DataFrame.from_dict(all_stats)

    def run_scenario(self, scenario_id: Union[int, str], return_predictions=False):
        if isinstance(scenario_id, int):
            try:
                scenario = self.scenarios[scenario_id]
            except IndexError:
                raise ValueError(f"{scenario_id} is not a correct index")
        if isinstance(scenario_id, str):
            found = False
            if self.scenarios == None:
                raise ValueError("Scenarios is empty.")
            for scenario in self.scenarios:
                if scenario.name == scenario_id:
                    found = True
                    break
            if found == False:
                raise ValueError(f"{scenario_id} is not a scenario name")

        predictions, inference_time_measures = scenario.model.predict(
            self.dataset.dataset,
            self.dataset.x_column_name,
            batch_size=scenario.batch_size,
        )
        stats = {
            "# of parameters": {scenario.name: scenario.model.count_parameters()},
            "latency (mean)": {scenario.name: np.mean(inference_time_measures)},
            "latency (90th percentile)": {scenario.name: np.percentile(inference_time_measures, 90)},
        }
        references = self.dataset.dataset[self.dataset.y_column_name]
        references = scenario.model.prepare_references(references)
        for metric in self.metrics:
            score = metric.metric.compute(
                predictions=predictions,
                references=references,
                **metric.run_kwargs,
            )
            if metric.values == None:
                raise ValueError("No values in metric.")
            for value in metric.values:
                stats["{}_{}".format(metric.metric_name, value)] = {scenario.name: get_value(score, value)}
        if return_predictions:
            return stats, predictions
        return stats
