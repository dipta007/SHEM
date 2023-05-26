import logging

import numpy as np
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataloaders.similarity_dataset import SimilarityDataset
from utils.utils import DotDict, load_pickle, to_device

log = logging.getLogger(__name__)

import numpy as np


def crappyhist(a, bins=50, width=140):
    h, b = np.histogram(a, bins)

    for i in range(0, bins):
        print(
            "{:12.5f}  | {:{width}s} {}".format(
                b[i], "#" * int(width * h[i] / np.amax(h)), h[i], width=width
            )
        )
    print("{:12.5f}  |".format(b[bins]))


def print_stats(q1, rq1, vocab, vocab2, it):
    def clean(x, end=2):
        x = x.tolist()
        return x[: x.index(end) + 1]

    e1 = q1["text"]
    f1 = q1["f_ref"]

    print("Passed:")
    print(vocab.batch_decode(clean(e1[0]["input_ids"][0])))
    print([vocab2.itos[x] for x in f1[0]])

    print("Result:")
    for i in range(2):
        logits = rq1[f"logits_{i}"]
        words = torch.argmax(logits, dim=-1)
        print(f"Layer {i}:")
        print(vocab.batch_decode(clean(words[0], 2)))

        f_logits = rq1[f"latent_gumbles_{i}"]
        frames = torch.argmax(f_logits, dim=-1)
        print([vocab2.itos[x] for x in frames[0]])


class Similarity:
    def __init__(self, config, model, path, hard=True):
        self.config = config
        self.model = model
        self.hard = hard
        self.vocab = AutoTokenizer.from_pretrained(self.config.data.vocab)
        log.info(f"Vocab size: {len(self.vocab)}")
        self.vocab2 = load_pickle(self.config.data.vocab2)
        log.info(f"Vocab2 size: {len(self.vocab2)}")
        self.dataset = SimilarityDataset(
            self.config,
            [path],
            self.vocab,
            self.vocab2,
            hard=hard,
        )
        self.dataloader = DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=4,
            collate_fn=self.dataset.collate_fn,
        )

    @torch.no_grad()
    def test(self, return_all=False):
        self.model.eval()
        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        results = []
        scores = []
        abs = []
        cds = []
        for it, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            q1 = DotDict(
                {
                    "text": batch["e"],
                    "target": batch["e"],
                    "frame": batch["f_e"],
                    "f_ref": batch["f_e_ref"],
                }
            )
            p1 = DotDict(
                {
                    "text": batch["pos"],
                    "target": batch["pos"],
                    "frame": batch["f_pos"],
                    "f_ref": batch["f_pos_ref"],
                }
            )
            q2 = DotDict(
                {
                    "text": batch["e2"],
                    "target": batch["e2"],
                    "frame": batch["f_e2"],
                    "f_ref": batch["f_e2_ref"],
                }
            )
            p2 = DotDict(
                {
                    "text": batch["neg"],
                    "target": batch["neg"],
                    "frame": batch["f_neg"],
                    "f_ref": batch["f_neg_ref"],
                }
            )

            # print('-----------------------------------------------------')
            # e1 = q1["text"]
            # f1 = q1["f_ref"]

            # e2 = p1["text"]
            # f2 = p1["f_ref"]

            # e3 = q2["text"]
            # f3 = q2["f_ref"]

            # e4 = p2["text"]
            # f4 = p2["f_ref"]
            # print(self.vocab.batch_decode(e1[0]["input_ids"]))
            # print([self.vocab2.itos[x] for x in f1[0]])

            # print(self.vocab.batch_decode(e2[0]["input_ids"]))
            # print([self.vocab2.itos[x] for x in f2[0]])

            # print('-'*50)
            # print('-'*50)

            # print(self.vocab.batch_decode(e3[0]["input_ids"]))
            # print([self.vocab2.itos[x] for x in f3[0]])

            # print(self.vocab.batch_decode(e4[0]["input_ids"]))
            # print([self.vocab2.itos[x] for x in f4[0]])
            # print('-----------------------------------------------------\n')

            q1 = to_device(q1, self.config.device)
            p1 = to_device(p1, self.config.device)
            q2 = to_device(q2, self.config.device)
            p2 = to_device(p2, self.config.device)

            rq1 = self.model(q1)
            rp1 = self.model(p1)
            rq2 = self.model(q2)
            rp2 = self.model(p2)

            # print('-----------------------------------------------------')
            # print(f'Input {it}:')
            # print_stats(q1, rq1, self.vocab, self.vocab2, it)
            # print('----------------')
            # print_stats(p1, rp1, self.vocab, self.vocab2, it)
            # if self.hard:
            #     print('----------------')
            #     print_stats(q2, rq2, self.vocab, self.vocab2, it)
            #     print('----------------')
            #     print_stats(p2, rp2, self.vocab, self.vocab2, it)
            # print('-----------------------------------------------------')

            c1 = rq1["q"]
            c2 = rp1["q"]

            c3 = rq2["q"]
            c4 = rp2["q"]

            ab_sim = cosine_similarity(c1, c2)
            if self.hard:
                cd_sim = cosine_similarity(c3, c4)

                # print(ab_sim.item())
                # print(cd_sim.item())

                ret = ab_sim > cd_sim

                abs += ab_sim.tolist()
                cds += cd_sim.tolist()

                results += ret.tolist()
            else:
                results += ab_sim.tolist()
                scores += batch["score"].tolist()

            # if it == 10:
            #     break

        result = None
        if self.hard:
            abcd = [x - y for x, y in zip(abs, cds)]
            # crappyhist(abcd)
            results = np.array(results)
            result = results.mean() * 100.0
        else:
            result, _ = stats.spearmanr(results, scores)

        log.info(f"Score: {result}")

        return result
