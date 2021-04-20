import torch

from evaluation.model import Model
from training.scripts.tokenizers.glove import GloveTokenizer
from training.scripts.models.hier_attn import HierarchicalAttentionNetworksForSequenceOrdering


class OrderingBaselineModel(Model):
    """
    Class for the baseline for the ordering model
    """

    def __init__(
        self,
        name,
        model_name,
        tokenizer_name,
        device,
        quantization,
        onnx,
        onnx_convert_kwargs,
        ordering_parameters={},
    ):
        if quantization and device != "cpu":
            raise ValueError("Quantization only works with CPU.")

        self.name = name
        self.device = device

        self.tokenizer = GloveTokenizer.from_pretrained(tokenizer_name)

        if not onnx:
            self.model = (
                HierarchicalAttentionNetworksForSequenceOrdering.from_pretrained(model_name).eval().to(self.device)
            )
            if quantization:
                self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            self.onnx_model = False
        else:
            raise ValueError("Onnx not supported.")

        self.ordering_parameters = ordering_parameters

    def _predict(self, x):
        x = x[0]
        decoder_first_sequence_ids = self.tokenizer(
            "start",
            max_length=self.tokenizer.max_length,
            padding="max_length",
            truncation=True,
        )["input_ids"][0]
        max_num_seq = max([len(sentences) for sentences in x])
        encoder_inputs = [
            self.tokenizer(
                sentences,
                max_length=self.tokenizer.max_length,
                padding="max_length",
                truncation=True,
                max_num_seq=max_num_seq,
                seq_padding="max_num_seq",
                seq_truncation=True,
            )
            for sentences in x
        ]
        input_ids = []
        attention_mask = []
        for elem in encoder_inputs:
            input_ids.append(elem["input_ids"])
            attention_mask.append(elem["attention_mask"])

        outputs = self.model.order(
            input_ids=torch.tensor(input_ids).to(self.device),
            attention_mask=torch.tensor(attention_mask).to(self.device),
            decoder_first_sequence_ids=decoder_first_sequence_ids,
            **self.ordering_parameters,
        )

        assert len(input_ids) == len(outputs)
        assert len(input_ids) == len(x)
        for i, (output, sequences) in enumerate(zip(outputs, x)):
            assert len(output) == len(sequences), f"sequences: {sequences}\noutput: {output}\ninput_id: {input_ids[i]}"
        return outputs
