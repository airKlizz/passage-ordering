from transformers import AutoTokenizer


class OrderingModel(object):
    def __init__(self, cls, model_name, tokenizer_name):
        self.model = cls.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, sequences, decoder_start_token_ids=[0, 2], num_beams=1):
        inputs = self.tokenizer(
            " </s><s> ".join(sequences) + " </s><s>",
            truncation=True,
            max_length=self.tokenizer.max_len,
            return_tensors="pt",
        )
        output = self.model.order(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_start_token_ids=decoder_start_token_ids,
            num_beams=num_beams,
        )
        output = output[0]
        output.remove(max(output))
        for i in range(len(sequences)):
            if i not in output:
                output.append(i)
        assert len(output) == len(sequences)
        return output

    def order(self, sequences, decoder_start_token_ids=[0, 2], num_beams=1):
        output = self(sequences, decoder_start_token_ids, num_beams)
        ordered_sequences = []
        for idx in output:
            ordered_sequences.append(sequences[idx])
        return ordered_sequences
