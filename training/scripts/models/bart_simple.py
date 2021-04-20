from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

from transformers import PretrainedBartModel, BartModel, BartConfig
from transformers.modeling_outputs import ModelOutput

from .ordering_utils import OrderingMixin


class BartPointerHead(nn.Module):
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
    ) -> Tuple[Tensor, Optional[Tensor]]:
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


@dataclass
class Seq2SeqOrderingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor]
    logits: torch.FloatTensor = None
    last_hidden_state: Optional[List[torch.FloatTensor]] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class BartForSequenceOrdering(PretrainedBartModel, OrderingMixin):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.pointer = BartPointerHead(config.d_model)
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id

    def is_sequence_ordering_model(self):
        return True

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        encoder_sequence_last_hidden_state = outputs.encoder_last_hidden_state
        decoder_sequence_last_hidden_state = outputs.last_hidden_state

        encoder_sequence_attention_mask = (input_ids == self.eos_token_id).float()
        if use_cache:
            decoder_sequence_attention_mask = (decoder_input_ids[:, -1:] == self.eos_token_id).float()
        else:
            decoder_sequence_attention_mask = (decoder_input_ids == self.eos_token_id).float()

        sequence_attention_mask = torch.bmm(
            decoder_sequence_attention_mask.unsqueeze(2), encoder_sequence_attention_mask.unsqueeze(1)
        )

        logits = self.pointer(
            query=decoder_sequence_last_hidden_state.transpose(1, 0),
            key=encoder_sequence_last_hidden_state.transpose(1, 0),
        )
        # logits: shape = (bsz, decoder_len, encoder_len), X_ij = probability of j to be the sentence after i

        assert sequence_attention_mask.size() == logits.size(), f"{sequence_attention_mask.size()}, {logits.size()}"

        logits[sequence_attention_mask == 0] = float("-inf")

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqOrderingOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, input_ids, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        return {
            "input_ids": input_ids,  # input_ids is needed for sequence mask
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        ((enc_out, enc_mask), past_key_values) = past
        reordered_past = []
        for layer_past in past_key_values:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        new_enc_out = enc_out if enc_out is None else enc_out.index_select(0, beam_idx)
        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)

        past = ((new_enc_out, new_enc_mask), reordered_past)
        return past

    def get_encoder(self):
        return self.model.encoder
