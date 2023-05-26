import random

import nltk
import torch
from nltk.corpus import framenet as fn

from base.base_dataset import BaseDataset

nltk.download("framenet_v17", download_dir="./")
nltk.data.path.append("./")

import spacy

from dataloaders.naacl_dataset import NOFRAME_TOK, PAD_TOK
from utils.utils import DotDict, get_file_line, get_file_line_count

# spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")


def get_swcc_like_evt(e):
    e = e.split()
    arg0 = e[0]
    arg1 = e[-1]
    pred = " ".join(e[1:-1])
    return f"{pred} {arg0} {arg1}"


def get_frames(word):
    if word is None:
        return [NOFRAME_TOK]
    try:
        lex_frames = fn.lus(rf"(?i){word}")
        sem_frames = []
        for v in lex_frames:
            sem_frames.append(v.frame.name)
        sem_frames = list(set(sem_frames))
        return sem_frames
    except Exception as e:
        print("Exception in frames:", e)
        return [NOFRAME_TOK]


def get_verb_from_sentence(sentence):
    st = nlp(sentence)
    for tok in st:
        if tok.pos_ == "VERB":
            return tok.lemma_
    return st[1].lemma_


class SimilarityDataset(BaseDataset):
    def __init__(self, config, paths, vocab, vocab2, hard=True):
        super().__init__(paths)

        self.config = config
        self.hard = hard
        self.vocab = vocab
        self.vocab2 = vocab2

    def __len__(self):
        return get_file_line_count(self.paths[0])

    def __getitem__(self, idx):
        line = get_file_line(self.paths[0], idx)
        line = line.split(" | ")
        line = [x.strip().lower() for x in line]

        e1, f1 = " ".join(line[0:3]), self.form_latent_variable(line[0:3])
        e2, f2 = " ".join(line[3:6]), self.form_latent_variable(line[3:6])
        e3, f3 = e1, f1
        e4, f4 = e2, f2
        score = 0.0
        if self.hard:
            e3, f3 = " ".join(line[6:9]), self.form_latent_variable(line[6:9])
            e4, f4 = " ".join(line[9:12]), self.form_latent_variable(line[9:12])
        else:
            score = float(line[-1])

        e1 = get_swcc_like_evt(e1)
        e2 = get_swcc_like_evt(e2)
        e3 = get_swcc_like_evt(e3)
        e4 = get_swcc_like_evt(e4)

        return e1, f1, f1, e2, f2, f2, e3, f3, f3, e4, f4, f4, score

    def form_latent_variable(self, text):
        return " ".join([NOFRAME_TOK] + [PAD_TOK] * 4)
        text = " ".join(text)
        verb = get_verb_from_sentence(text)
        frames = get_frames(verb)
        random.shuffle(frames)
        frames = frames[: self.config.data.frame_len]
        frames = frames + (self.config.data.frame_len - len(frames)) * [PAD_TOK]
        frames = " ".join(frames)
        return frames

    def collate_fn(self, batch):
        # encode all the data
        batch = list(zip(*batch))
        for idx in [0, 3, 6, 9]:
            encoded_txt = self.vocab(
                list(batch[idx]),
                padding="max_length",
                truncation=True,
                max_length=self.config.data.seq_len,
                return_tensors="pt",
            )
            lengths = torch.LongTensor(
                [torch.sum(x).item() for x in encoded_txt.attention_mask]
            )
            batch[idx] = (encoded_txt, lengths)

        for idx in [1, 2, 4, 5, 7, 8, 10, 11]:
            batch[idx] = torch.tensor(
                [[self.vocab2.stoi[tok] for tok in line.split()] for line in batch[idx]]
            )

        batch[12] = torch.tensor(batch[12], dtype=torch.float64)

        # make the batch named for convenience
        ret = DotDict(
            {
                "e": batch[0],
                "f_e": batch[1],
                "f_e_ref": batch[2],
                "pos": batch[3],
                "f_pos": batch[4],
                "f_pos_ref": batch[5],
                "e2": batch[6],
                "f_e2": batch[7],
                "f_e2_ref": batch[8],
                "neg": batch[9],
                "f_neg": batch[10],
                "f_neg_ref": batch[11],
                "score": batch[12],
            }
        )

        return ret
