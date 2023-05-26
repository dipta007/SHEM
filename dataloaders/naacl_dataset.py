import copy
import logging
import pickle
from audioop import reverse

import torch

from base.base_dataset import BaseDataset
from utils.utils import DotDict

log = logging.getLogger(__name__)


# Reserved Special Tokens
PAD_TOK = "<pad>"
SOS_TOK = "<s>"  # start of sentence
EOS_TOK = "</s>"  # end of sentence
UNK_TOK = "<unk>"
TUP_TOK = " . "
DIST_TOK = "<DIST>"  # distractor token for NC task
NOFRAME_TOK = "__NOFRAME__"

# These are the values that should be used during evalution to keep things consistent
MIN_EVAL_SEQ_LEN = 8
MAX_EVAL_SEQ_LEN = 50


class SentenceDatasetWithBart(BaseDataset):
    def __init__(self, config, paths, fields, vocab, vocab2, obsv_prob=1.0):
        super().__init__(paths, fields)

        self.config = config
        self.vocab = vocab
        self.vocab2 = vocab2
        self.obsv_prob = obsv_prob
        self.data = self.get_data()

    def get_data(self):
        num_of_words = 5  # with pad tok
        num_clauses = 5
        cut_off = num_of_words * num_clauses - 1
        is_observed = lambda x: self.vocab2.stoi[x] != self.vocab2.stoi[NOFRAME_TOK]

        data = []

        def get_frames(linef):
            true_frame = linef.split()[:num_clauses]
            obs_frames_idx = [
                idx for idx, fr in enumerate(true_frame) if is_observed(fr)
            ]
            obs_frames = [fr for fr in true_frame if is_observed(fr)]
            probs = self.obsv_prob * torch.ones(len(obs_frames))
            selector = torch.bernoulli(probs)

            ref_frames = obs_frames

            masked_frames = [
                ref_frames[idx] if selector[idx] == 1 else NOFRAME_TOK
                for idx, _ in enumerate(ref_frames)
            ]

            # PAD if not enough
            masked_frames = masked_frames + [PAD_TOK] * (
                num_clauses - len(masked_frames)
            )
            ref_frames = ref_frames + [PAD_TOK] * (num_clauses - len(ref_frames))

            # Join
            masked_frames = " ".join(masked_frames)
            ref_frames = " ".join(ref_frames)

            return masked_frames, ref_frames, " ".join(true_frame), obs_frames_idx

        def get_events(linee, obs_frames_idx):
            line = [word.lower() if word != "<TUP>" else word for word in linee.split()]
            true_text = " ".join(line[:cut_off])
            text = true_text
            clause_split = text.split("<TUP>")
            good_clause = "<TUP>".join([clause_split[idx] for idx in obs_frames_idx])

            # for transformer model, replace <TUP> with "."
            good_clause = good_clause.replace("<TUP>", ".")
            text = text.replace("<TUP>", ".")
            true_text = true_text.replace("<TUP>", ".")

            return good_clause, text, true_text

        with open(self.paths[0], "r") as f:
            with open(self.paths[1], "r") as ft:
                for event_line in f:
                    frame_line = ft.readline().strip()
                    event_line = event_line.strip()
                    masked_frames, ref_frames, _, obs_frames_idx = get_frames(
                        frame_line
                    )
                    good_clause, _, _ = get_events(event_line, obs_frames_idx)

                    if (
                        "__NOFRAME__" in ref_frames.split()
                        or len(good_clause.split()) == 0
                    ):
                        # log.error(f"Faulty: {event_line} {frame_line}")
                        pass
                    else:
                        data.append(
                            [good_clause, good_clause, ref_frames, masked_frames]
                        )

                    # if len(data) == 100:
                    #     break

                return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        # ? sorting
        batch.sort(key=lambda x: len(x[0].split()), reverse=True)

        # #? zipping to collate inside batch
        values = list(zip(*batch))

        x = []
        # #? padding and make tensor
        for v in values[:2]:
            v = self.vocab(
                list(v),
                padding="max_length",
                truncation=True,
                max_length=self.config.data.seq_len,
                return_tensors="pt",
            )
            lengths = torch.LongTensor([torch.sum(x).item() for x in v.attention_mask])

            x.append((v, lengths))

        for v in values[2:]:
            v = torch.tensor(
                [[self.vocab2.stoi[tok] for tok in line.split()] for line in v]
            )
            x.append(v)

        # #? make dictionary
        ret = {x: y for x, y in zip(self.fields, x)}

        return DotDict(ret)


def load_vocab(filename):
    # load vocab from json file
    with open(filename, "rb") as fi:
        voc = pickle.load(fi)
        return voc
