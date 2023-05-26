import logging

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel

from dataloaders.naacl_dataset import PAD_TOK
from models.naacl.dag import get_latent_chain
from models.naacl.decoder import Decoder
from models.naacl.encoder import Encoder
from models.naacl.framenet_relations import get_frames_for_upper_layers

log = logging.getLogger(__name__)

# create a pytorch dummy model
class NaaclModel(nn.Module):
    def __init__(self, config, transformer, vocab, vocab2):
        super().__init__()
        self.config = config

        self.vocab = vocab
        self.vocab2 = vocab2

        transformer = AutoModel.from_pretrained(transformer)
        frame_embedding0 = nn.Embedding(
            self.config.train.model.num_latent_values,
            self.config.train.model.latent_dim,
            padding_idx=self.config.train.model.frame_pad_idx,
        )

        self.template_to_frame = nn.Linear(
            self.config.train.model.template,
            self.config.train.model.num_latent_values,
            bias=False,
        )
        self.template_to_vocab = nn.Linear(
            self.config.train.model.num_latent_values,
            self.config.train.model.vocab_size,
            bias=False,
        )
        self.theta_layer = nn.Linear(
            self.config.train.model.enc_hid_size, self.config.train.model.template
        )
        # TODO: need to remove hardcoded value
        self.frame_to_seq_len = nn.ModuleList(
            [
                nn.Linear(
                    config.train.model.num_of_children[0], self.config.data.seq_len
                ),
                nn.Linear(
                    config.train.model.num_of_children[1], self.config.data.seq_len
                ),
            ]
        )

        # ? Layer 0
        self.encoder0 = Encoder(self.config, transformer.encoder)
        self.latent_root = nn.ModuleList(
            [
                get_latent_chain(
                    self.config,
                    layer_idx=0,
                    frame_embedding=frame_embedding0,
                ),
                get_latent_chain(
                    self.config,
                    layer_idx=1,
                    frame_embedding=frame_embedding0,
                ),
            ]
        )
        self.decoder0 = Decoder(self.config, transformer.decoder)

        self.transformer_config = AutoConfig.from_pretrained(
            self.config.train.trainer.model.transformer
        )

        self.contrastive_layer = nn.Linear(
            self.config.train.model.dec_hid_size
            + self.config.train.model.num_latent_values,
            self.config.train.model.dec_hid_size
            + self.config.train.model.num_latent_values,
        )
        self.enc_dropout = nn.Dropout(self.config.train.model.enc_dropout)

    def forward(self, data, encode_only=False):
        result = {}
        data["frame_0"] = data["frame"].to(self.config.device)
        for idx in range(len(self.config.train.model.num_of_children)):
            (
                enc_output,
                frame_logits,
                dec_output,
                logits,
                frame_classifier,
                q_log_q,
                latent_embs,
                latent_gumbles,
                template_decode_input,
            ) = self.layer(data, idx, encode_only)

            eos_mask = data["text"][0]["input_ids"].eq(
                self.transformer_config.eos_token_id
            )
            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError(
                    "All examples must have the same number of <eos> tokens."
                )

            sentence_representation = dec_output[eos_mask, :].view(
                dec_output.size(0), -1, dec_output.size(-1)
            )[:, -1, :]

            if idx == 0:
                for i in range(latent_gumbles.shape[1]):
                    result[f"p_{i}"] = torch.cat(
                        [sentence_representation, latent_gumbles[:, i, :]], dim=1
                    )
                    result[f"p_{i}"] = torch.tanh(
                        self.contrastive_layer(result[f"p_{i}"])
                    )
                    result[f"p_{i}"] = nn.functional.normalize(result[f"p_{i}"], dim=1)
            else:
                result[f"q"] = torch.cat(
                    [sentence_representation, latent_gumbles[:, 0, :]], dim=1
                )
                result[f"q"] = torch.tanh(self.contrastive_layer(result[f"q"]))
                result[f"q"] = nn.functional.normalize(result[f"q"], dim=1)

            obj = {
                f"frame_logits_{idx}": frame_logits,
                f"logits_{idx}": logits,
                f"frame_classifier_{idx}": frame_classifier,
                f"q_log_q_{idx}": q_log_q,
                f"latent_embs_{idx}": latent_embs,
                f"latent_gumbles_{idx}": latent_gumbles,
                f"template_decode_input_{idx}": template_decode_input,
            }

            if idx == 0:
                curr_f_vals = torch.argmax(latent_gumbles, -1)
                data["frame_1"] = get_frames_for_upper_layers(
                    self.config, curr_f_vals, self.vocab2
                )
                data["frame_1"] = data["frame_1"].to(self.config.device)

            result.update(obj)

        return result

    def drop_encoding(self, input, encoding):
        # 0 == start token, 479 == full stop (.)
        mask = torch.logical_or(input.eq(0), input.eq(479))
        mask = torch.roll(mask, 1, dims=1)
        val = encoding[mask]
        dropped_val = self.enc_dropout(val)
        encoding[mask] = dropped_val
        return encoding

    def layer(self, data, model_idx=0, encode_only=False):
        batch, batch_lens = data["text"]
        f_vals = data[f"frame_{model_idx}"]

        # ? encode
        enc_output = self.encoder0(batch)
        enc_output = self.drop_encoding(batch.input_ids, enc_output)

        enc_theta = self.theta_layer(enc_output).mean(1)
        p_theta_sampled = F.softmax(enc_theta, -1)
        template_input = torch.tanh(self.template_to_frame(p_theta_sampled))
        template_decode_input = self.template_to_vocab(template_input)

        # ? encode to latent variable
        enc_output_avg = torch.sum(enc_output, dim=1) / batch_lens.view(-1, 1).type(
            torch.FloatTensor
        ).to(self.config.device)
        initial_query = enc_output_avg
        (
            frame_logits,
            latent_gumbles,
            latent_embs,
            q_log_q,
            frames_to_frames,
            frame_classifier,
            vocab_to_frames,
        ) = self.latent_root[model_idx].forward(
            enc_output,
            batch_lens,
            initial_query,
            f_vals,
            template_input=template_input,
        )  # (batch, num_clauses, num_frames)

        latent_embs = torch.tanh(
            self.frame_to_seq_len[model_idx](latent_embs.permute(0, 2, 1))
        )
        latent_embs = latent_embs.permute(0, 2, 1)

        decoder_in = torch.cat(
            [latent_embs, enc_output], dim=1
        )  # (batch, seq_len * 2, hidden_size)

        dec_output, logits = None, None
        if not encode_only:
            dec_output, logits = self.decoder0(batch, decoder_in, template_decode_input)

        return (
            enc_output,
            frame_logits,
            dec_output,
            logits,
            frame_classifier,
            q_log_q,
            latent_embs,
            latent_gumbles,
            template_decode_input,
        )
