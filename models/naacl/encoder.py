from torch import nn


class Encoder(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder

    def forward(self, input):
        out = self.encoder(input["input_ids"], attention_mask=input["attention_mask"])
        encoder_hidden_state = out.last_hidden_state
        return encoder_hidden_state
