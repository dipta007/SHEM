import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query_dim = self.config.train.model.enc_hid_size
        self.memory_dim = self.config.train.model.latent_dim
        self.output_dim = self.config.train.model.num_latent_values
        # this is the W for computing scores
        self.linear_in = nn.Linear(self.query_dim, self.memory_dim, bias=False)
        # Multiply context vector concated with hidden
        self.linear_out = nn.Linear(self.memory_dim, self.output_dim, bias=False)

    def forward(self, input, memory, mem_lens=None, template_decode_input=None):
        Wh = self.linear_in(input).unsqueeze(1)  # [batch, 1, mem_dim]
        memory_t = memory.transpose(1, 2)  # [batch, dim, seq_len]
        scores = torch.bmm(Wh, memory_t)  # [batch, 1, seq_len]

        if mem_lens is not None:  # mask out the pads
            mask = sequence_mask(mem_lens, self.config.data.seq_len)
            mask = mask.unsqueeze(1)
            scores.data.masked_fill_(~(mask), -float("inf"))

        scale = (
            torch.sqrt(torch.tensor([memory.shape[2]]))
            .view(1, 1, 1)
            .to(self.config.device)
        )

        # [batch, 1, seq_len], scores for each batch
        # ? NAACL paper - algo 1 - line 2
        scores = F.softmax(scores / scale, dim=2)

        # [batch, dim], context vectors for each batch
        # ? NAACL paper - algo 1 - line 3
        context = torch.bmm(scores, memory).squeeze(dim=1)
        # cat = torch.cat([context, input], 1)
        cat = torch.tanh(context) + torch.tanh(Wh.squeeze())
        # ? NAACL paper - algo 1 - line 4
        attn_output = self.linear_out(cat)
        frame_to_frame = self.linear_out(torch.tanh(Wh))
        vocab_to_frame = self.linear_out(torch.tanh(memory))
        return attn_output, scores, frame_to_frame, vocab_to_frame


def sequence_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (
        torch.arange(0, max_len)
        .type_as(lengths)
        .repeat(batch_size, 1)
        .lt(lengths.unsqueeze(1))
    )
