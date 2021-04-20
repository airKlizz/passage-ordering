from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from transformers.configuration_bart import BartConfig
from transformers.modeling_bart import EncoderLayer, DecoderLayer, LayerNorm
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

try:
    from transformers.modeling_bart import Attention
except:
    from transformers.modeling_bart import SelfAttention as Attention


SEED = 42
torch.manual_seed(SEED)

CONFIG_NAME = "model_config.json"
EMBEDDING_NAME = "embedding.pt"
WEIGHTS_NAME = "pytorch_model.bin"


def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    return attention_mask.eq(0)


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def create_causal_mask(tgt_len, dtype, device):
    tmp = fill_with_neg_inf(torch.zeros(tgt_len, tgt_len))
    mask = torch.arange(tmp.size(-1))
    tmp.masked_fill_(mask < (mask + 1).view(tmp.size(-1), 1), 0)
    causal_mask = tmp.to(dtype=dtype, device=device)
    return causal_mask


class WordEncoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, embedding, num_layers=1, dropout=0.0):
        super().__init__()

        # Basic network params
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding layer that will be shared with Decoder
        self.embedding = embedding

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, input_ids, attention_mask):

        """
        Args:
            input_ids `torch.Tensor` of shape `(batch, seq_len)`:
                input token ids.
            attention_mask `torch.Tensor` of shape `(batch, seq_len)`:
                1 for words and 0 for pad tokens.
        Output:
            output `torch.Tensor` of shape `(batch, 2*self.hidden_size)`:
                Embedding the sequences
        """

        bsz = input_ids.size(0)
        seq_len = input_ids.size(1)
        device = input_ids.device

        # Convert input_sequence to word embeddings
        word_embeddings = self.embedding(input_ids)
        assert word_embeddings.size() == (bsz, seq_len, self.embedding_size), word_embeddings.size()

        # Run the embeddings through the LSTM
        outputs, hidden = self.lstm(word_embeddings)
        last = attention_mask.sum(-1) - 1
        outputs = outputs[torch.arange(bsz), last]
        assert outputs.size() == (bsz, self.hidden_size), outputs.size()

        return outputs


class PointerHead(nn.Module):
    """Head for pointer ordering task."""

    def __init__(
        self,
        embed_dim,
        bias=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.scaling = self.embed_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz, self.embed_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key,
    ):
        """Input shape: Time(SeqLen) x Batch x Channel"""
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)

        q = self._shape(q, tgt_len, bsz)
        k = self._shape(k, -1, bsz)

        assert k is not None
        assert q is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz, tgt_len, src_len)

        return attn_weights


class HierarchicalAttentionNetworksForSequenceOrdering(PreTrainedModel):
    def __init__(
        self,
        embedding,
        rnn_hidden_size,
        rnn_num_layers,
        num_attn_heads,
        num_encoder_layers,
        num_decoder_layers,
        ffn_dim,
        dropout,
    ):

        super().__init__(PretrainedConfig())
        self.embedding = embedding
        self.word_embed_dim = embedding.embedding_dim

        # model config
        self.model_config = {
            "rnn_hidden_size": rnn_hidden_size,
            "rnn_num_layers": rnn_num_layers,
            "num_attn_heads": num_attn_heads,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "ffn_dim": ffn_dim,
            "dropout": dropout,
        }

        self.word_encoder = WordEncoder(
            hidden_size=rnn_hidden_size,
            embedding_size=self.word_embed_dim,
            embedding=embedding,
            num_layers=rnn_num_layers,
            dropout=dropout,
        )

        self.sentence_embed_dim = rnn_hidden_size

        self.config = BartConfig(
            d_model=self.sentence_embed_dim,
            encoder_attention_heads=num_attn_heads,
            decoder_attention_heads=num_attn_heads,
            attention_dropout=0.0,
            dropout=dropout,
            activation_dropout=0.0,
            encoder_ffn_dim=ffn_dim,
            decoder_ffn_dim=ffn_dim,
        )

        self.encoder_layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(num_decoder_layers)])

        self.pointer_head = PointerHead(self.sentence_embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / CONFIG_NAME, "w") as f:
            json.dump(self.model_config, f)

        torch.save(self.embedding.weight, path / EMBEDDING_NAME)

        torch.save(self.state_dict(), path / WEIGHTS_NAME)

    @classmethod
    def from_pretrained(cls, path):
        path = Path(path)

        with open(path / CONFIG_NAME, "r") as f:
            config = json.load(f)

        weight = torch.load(path / EMBEDDING_NAME)
        config["embedding"] = nn.Embedding.from_pretrained(weight)

        model = cls(**config)
        model.load_state_dict(torch.load(path / WEIGHTS_NAME))

        return model

    def forward(
        self,
        input_ids,
        decoder_input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
        last_encoder_hidden_states=None,
        past_key_values=None,
        labels=None,
    ):

        """
        Args:
            input_ids/decoder_input_ids `torch.Tensor` of shape `(batch, num_seq, seq_len)`:
                input token ids.
            attention_mask/decoder_attention_mask `torch.Tensor` of shape `(batch, num_seq, seq_len)`:
                1 for words and 0 for pad tokens.
        Output:

        """

        bsz, num_seq, seq_len = input_ids.size()
        assert decoder_input_ids.size(0) == bsz
        assert decoder_input_ids.size(2) == seq_len
        decoder_num_seq = decoder_input_ids.size(1)

        if attention_mask == None:
            attention_mask = input_ids.new_ones(input_ids.size())
        if decoder_attention_mask == None:
            decoder_attention_mask = decoder_input_ids.new_ones(decoder_input_ids.size())

        if last_encoder_hidden_states is None:

            """ Sentence Encoder """

            # Word Encoder
            input_ids = input_ids.view(bsz * num_seq, seq_len)
            attention_mask = attention_mask.view(bsz * num_seq, seq_len)
            x = self.word_encoder.forward(input_ids, attention_mask)
            x = x.view(bsz, num_seq, self.sentence_embed_dim)
            attention_mask = attention_mask.view(bsz, num_seq, seq_len)

            # Prepare inputs
            attention_mask = (
                attention_mask.sum(-1) == 1
            )  # == 1 means there is only the [CLS] token which means it is a pad sequence
            x = x.transpose(0, 1)
            assert x.size() == (num_seq, bsz, self.sentence_embed_dim), x.size()
            assert attention_mask.size() == (bsz, num_seq), attention_mask.size()
            assert attention_mask[0][0] == 0, "non-padding elements should be 0."

            # Run Sentence Encoder layers
            for encoder_layer in self.encoder_layers:
                x, attn = encoder_layer(x, attention_mask)
            last_encoder_hidden_states = x.transpose(0, 1)
            assert last_encoder_hidden_states.size() == (
                bsz,
                num_seq,
                self.sentence_embed_dim,
            ), last_encoder_hidden_state.size()

        else:

            attention_mask = (
                attention_mask.sum(-1) == 1
            )  # == 1 means there is only the [CLS] token which means it is a pad sequence
            assert attention_mask.size() == (bsz, num_seq), attention_mask.size()

        """ Sentence Decoder """

        # Word Encoder
        decoder_input_ids = decoder_input_ids.view(bsz * decoder_num_seq, seq_len)
        decoder_attention_mask = decoder_attention_mask.view(bsz * decoder_num_seq, seq_len)
        x = self.word_encoder.forward(decoder_input_ids, decoder_attention_mask)
        x = x.view(bsz, decoder_num_seq, self.sentence_embed_dim)
        decoder_attention_mask = decoder_attention_mask.view(bsz, decoder_num_seq, seq_len)

        # Prepare decoder inputs
        decoder_attention_mask = (
            decoder_attention_mask.sum(-1) == 1
        )  # == 1 means there is only the [CLS] token which means it is a pad sequence
        assert decoder_attention_mask.size() == (bsz, decoder_num_seq), decoder_attention_mask.size()
        x = x.transpose(0, 1)
        encoder_hidden_states = last_encoder_hidden_states.transpose(0, 1)
        decoder_causal_mask = create_causal_mask(decoder_num_seq, decoder_input_ids.dtype, decoder_input_ids.device)

        # Run Sentence Decoder layers
        for idx, decoder_layer in enumerate(self.decoder_layers):
            layer_state = past_key_values[idx] if past_key_values is not None else None
            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=attention_mask,
                decoder_padding_mask=decoder_attention_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
            )

        last_decoder_hidden_states = x.transpose(0, 1)
        assert last_decoder_hidden_states.size() == (
            bsz,
            decoder_num_seq,
            self.sentence_embed_dim,
        ), last_decoder_hidden_states.size()

        """ Pointer Head """

        logits = self.pointer_head(
            query=last_decoder_hidden_states.transpose(1, 0),
            key=last_encoder_hidden_states.transpose(1, 0),
        )
        assert logits.size() == (bsz, decoder_num_seq, num_seq)
        assert attention_mask.size() == (bsz, num_seq)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, decoder_num_seq, 1)
        logits = logits.masked_fill(attention_mask, float("-inf"))

        outputs = (logits, last_encoder_hidden_states, last_decoder_hidden_states)

        loss = None
        if labels is not None:
            assert labels.size(0) == bsz
            assert labels.size(1) == decoder_num_seq
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

    def order(self, input_ids, decoder_first_sequence_ids, pad_tok=1, num_beams=1, attention_mask=None):

        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert input_ids is not None, "Input_ids is not defined."
        assert input_ids.dim() == 3, "Input prompt should be of shape (batch, num_seq, seq_len)."

        bsz = input_ids.size(0)
        num_seq = input_ids.size(1)

        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        num_seq_per_batch = (attention_mask.sum(-1) == 1).eq(False).sum(-1).tolist()

        decoder_input_ids = torch.tensor(decoder_first_sequence_ids).repeat(bsz, 1, 1).to(input_ids.device)

        if num_beams > 1:
            raise ValueError("Beam search not implemented yet.")
        else:

            last_encoder_hidden_states = None
            predictions_per_batch = [[] for _ in range(bsz)]
            done_per_batch = torch.zeros(bsz, dtype=torch.bool)
            for i in range(num_seq):

                decoder_attention_mask = decoder_input_ids.eq(pad_tok).eq(0)
                model_inputs = {
                    "input_ids": input_ids,
                    "decoder_input_ids": decoder_input_ids,
                    "attention_mask": attention_mask,
                    "decoder_attention_mask": decoder_attention_mask,
                    "last_encoder_hidden_states": last_encoder_hidden_states,
                }
                outputs = self(**model_inputs)
                scores = outputs[0]
                scores = scores.argsort(-1, descending=True)[:, -1, :]
                last_encoder_hidden_states = outputs[1]

                prediction_per_batch = scores.new_ones(bsz) * -1
                for idx in range(bsz):

                    # the batch is already done so the prediction doesn't make sense
                    if done_per_batch[idx]:
                        prediction_per_batch[idx] = 0
                        continue

                    # add the best prediction not already ordered
                    for prediction in scores[idx]:
                        if prediction not in predictions_per_batch[idx]:
                            prediction_per_batch[idx] = prediction
                            predictions_per_batch[idx].append(int(prediction))
                            break

                    assert prediction_per_batch[idx] != -1

                # Add sentences to decoder_input_ids
                decoder_input_ids = torch.cat(
                    (decoder_input_ids, input_ids[torch.arange(bsz), prediction_per_batch].unsqueeze(1)), dim=1
                )

                for idx in range(bsz):
                    if len(predictions_per_batch[idx]) == num_seq_per_batch[idx]:
                        done_per_batch[idx] = True

                if done_per_batch.all():
                    break

            return predictions_per_batch
