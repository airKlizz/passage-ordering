from training.scripts.models.bart_multi import BartForSequenceOrderingWithMultiPointer
from evaluation.model import Model


class OrderingModelMulti(Model):
    """
    Class for BART for the ordering model
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
        super().__init__(
            name,
            BartForSequenceOrderingWithMultiPointer,
            model_name,
            tokenizer_name,
            device,
            quantization,
            onnx,
            onnx_convert_kwargs,
        )
        self.ordering_parameters = ordering_parameters

    def _predict(self, x):
        x = x[0]
        pt_batch = self.tokenizer(
            [" </s> <s> ".join(sequences) + " </s> <s>" for sequences in x],
            padding=True,
            truncation=True,
            max_length=self.tokenizer.max_len,
            return_tensors="pt",
        )
        outputs = self.model.order(
            input_ids=pt_batch["input_ids"].to(self.device),
            attention_mask=pt_batch["attention_mask"].to(self.device),
            **self.ordering_parameters,
        )
        for output, sequences in zip(outputs, x):
            output.remove(max(output))
            for i in range(len(sequences)):
                if i not in output:
                    output.append(i)
            while max(output) > len(sequences) - 1:
                print(
                    f"INFO: Before second verification: sequences: {len(sequences)} - output: {len(output)} --- \n output:\n{output}"
                )
                output.remove(max(output))
            assert len(output) == len(sequences), f"sequences: {sequences} - output: {output}"
        return outputs
