import logging

import torch
from torch import nn
from transformers import AutoConfig

log = logging.getLogger(__name__)


class Decoder(nn.Module):
    def __init__(self, config, decoder):
        super().__init__()

        self.config = config
        vocab_size = self.config.train.model.vocab_size
        self.decoder = decoder
        self.transformer_config = AutoConfig.from_pretrained(
            self.config.train.trainer.model.transformer
        )
        self.logit_out = nn.Linear(
            self.config.train.model.dec_hid_size, vocab_size, bias=False
        )
        self.logit_out.weight.data.normal_(
            mean=0.0, std=self.transformer_config.init_std
        )

    def forward(self, input, hid_state, template_decode_input):
        dec_input = self.shift_tokens_right(
            input["input_ids"],
            self.transformer_config.pad_token_id,
            self.transformer_config.decoder_start_token_id,
        )
        out = self.decoder(dec_input, encoder_hidden_states=hid_state)
        dec = out.last_hidden_state

        logits = self.logit_out(dec)
        logits = logits.permute(1, 0, 2) + template_decode_input
        logits = logits.permute(1, 0, 2)
        return dec, logits

    def shift_tokens_right(self, input_ids, pad_token_id, decoder_start_token_id):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
