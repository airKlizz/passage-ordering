import time
from pathlib import Path

import torch
import torch.quantization
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers.convert_graph_to_onnx import convert, optimize, quantize


class Model(object):
    """
    Parent class of all sub-models
    """

    def __init__(
        self,
        name,
        model_cls,
        model_name,
        tokenizer_name,
        device,
        quantization,
        onnx,
        onnx_convert_kwargs,
        pipeline_name=None,
    ):

        if quantization and device != "cpu":
            raise ValueError("Quantization only works with CPU.")

        self.name = name
        self.device = device

        if tokenizer_name == None:
            tokenizer_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if not onnx:
            self.model = model_cls.from_pretrained(model_name).eval().to(self.device)
            if quantization:
                self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            self.onnx_model = False
        else:
            self.model = OnnxModel(
                model_name=model_name,
                pipeline_name=pipeline_name,
                device=self.device,
                quantization=quantization,
                force=True if onnx == "force" else False,
                **onnx_convert_kwargs,
            )
            self.onnx_model = True

    def predict(self, dataset, x_column_name, batch_size):
        predictions = []
        inference_time_measures = []
        for i in tqdm(range(0, len(dataset), batch_size), desc="Prediction"):
            x = [dataset[i : i + batch_size][column_name] for column_name in x_column_name]
            start = time.time()
            predictions += self._predict(x)
            time_elapsed = (time.time() - start) / batch_size
            inference_time_measures.append(time_elapsed)
        return predictions, inference_time_measures

    def _predict(self, x):
        raise NotImplementedError

    def prepare_references(self, references):
        return list(map(self._prepare_reference, references))

    def _prepare_reference(self, reference):
        return reference

    def check_model_type(self, supported_models):
        if not isinstance(supported_models, list):  # Create from a model mapping
            supported_models = [item[1].__name__ for item in supported_models.items()]
        if self.model.__class__.__name__ not in supported_models and self.model.__class__.__name__ != "OnnxModel":
            raise ValueError(
                f"The model '{self.model.__class__.__name__}' is not supported. Supported models are {supported_models}",
            )

    def prepare_inputs(self, pt_batch):
        if not self.onnx_model:
            model_args_name = self.model.forward.__code__.co_varnames
            model_args_name = model_args_name[1:]  # start at index 1 to skip "self" argument
        else:
            model_args_name = [onnx_input.name for onnx_input in self.model.session.get_inputs()]
        prepared_inputs = {}
        for arg_name in model_args_name:
            if arg_name in pt_batch.keys():
                prepared_inputs[arg_name] = pt_batch[arg_name].to(self.device)
        return prepared_inputs

    def count_parameters(self):
        if not self.onnx_model:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return -1  # return -1 for Onnx models


class OnnxModel(object):
    def __init__(
        self,
        model_name,
        pipeline_name,
        model_path=None,
        device="cpu",
        quantization=False,
        opset=11,
        force=False,
        **convert_kwargs
    ):
        if model_path == None:
            model_path = f"onnx/{model_name}.onnx"
        model_path = Path(model_path)

        if not model_path.is_file() or force:
            convert(
                framework="pt",
                model=model_name,
                output=model_path,
                opset=opset,
                pipeline_name=pipeline_name,
                **convert_kwargs,
            )

        if quantization:
            model_path = optimize(model_path)
            model_path = quantize(model_path)

        self.model_path = str(model_path)
        self.provider = "CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider"
        self.session = self.create_model_for_provider()
        self.config = AutoConfig.from_pretrained(model_name)

    def create_model_for_provider(self):

        assert self.provider in get_all_providers(), f"provider {self.provider} not found, {get_all_providers()}"

        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load the model as a graph and prepare the CPU backend
        session = InferenceSession(self.model_path, options, providers=[self.provider])
        session.disable_fallback()

        return session

    def __call__(self, **inputs):
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in inputs.items()}
        outputs = self.session.run(None, inputs_onnx)
        return [torch.from_numpy(output) for output in outputs]
