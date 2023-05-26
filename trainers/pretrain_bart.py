import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

EPOCHS = 100
BATCH_SIZE = 64
DEVICE = "cuda"
LR = 1e-4

model = AutoModelForMaskedLM.from_pretrained("./data/BART/base")
config = AutoConfig.from_pretrained("./data/BART/base")
tokenizer = AutoTokenizer.from_pretrained("./data/BART/base")

model = model.to(DEVICE)


class NaaclDatasetForPretrain:
    def __init__(self, config, tokenizer, data_path):
        self.config = config
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.data = self.get_data()

    def shift_tokens_right(self, input_ids):
        pad_token_id = self.config.pad_token_id
        decoder_start_token_id = self.config.decoder_start_token_id
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def get_data(self):
        data = []
        with open(self.data_path, "r") as f:
            for i, line in tqdm(enumerate(f)):
                line = line.replace("<TUP>", ".")
                enc = self.tokenizer(
                    line,
                    padding="max_length",
                    truncation=True,
                    max_length=170,
                    return_tensors="pt",
                )
                dec = self.shift_tokens_right(enc.input_ids)[0]

                mask_pos = random.randint(1, sum(enc.attention_mask[0]) - 2)
                enc = enc.input_ids[0]
                labels = enc.clone()

                enc[mask_pos] = self.tokenizer.mask_token_id  # 50264
                data.append([enc, dec, labels])

                # if i % 1000 == 0:
                #     break

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


dataset = NaaclDatasetForPretrain(config, tokenizer, "data/NAACL/train_0.6_TUP.txt")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
dataset = NaaclDatasetForPretrain(config, tokenizer, "data/NAACL/valid_0.6_TUP.txt")
valid = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_loss = 1000000

for epoch in range(EPOCHS):
    losses = []
    model.train()
    with tqdm(enumerate(dataloader), unit="batch", total=len(dataloader)) as tepoch:
        for _, v in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            enc, dec, labels = v
            enc, dec, labels = enc.to(DEVICE), dec.to(DEVICE), labels.to(DEVICE)
            loss = model(enc, decoder_input_ids=dec, labels=labels)[0]
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())
            losses.append(loss.item())

    print(f"Epoch {epoch} train loss: {sum(losses) / len(losses)}")

    with torch.no_grad():
        model.eval()
        losses = []
        for v in valid:
            enc, dec, labels = v
            enc, dec, labels = enc.to(DEVICE), dec.to(DEVICE), labels.to(DEVICE)
            loss = model(enc, decoder_input_ids=dec, labels=labels)[0]
            losses.append(loss.item())

        val_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch} valid loss: {val_loss}")
        if val_loss < best_val_loss:
            print("Saving model on epoch", epoch)
            best_val_loss = val_loss
            model.save_pretrained("data/NAACL/bart-base")
