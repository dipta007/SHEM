import copy
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataloaders.inv_narr_dataset import (
    MAX_EVAL_SEQ_LEN,
    MIN_EVAL_SEQ_LEN,
    InverseNarrativeDatasetForBart,
)
from models.naacl.masked_cross_entropy import masked_cross_entropy
from utils.utils import load_pickle, to_device

log = logging.getLogger(__name__)


def is_it_correct(nll, sz, num_of_models, target=0):
    pps = []
    for i in range(sz):
        curr_pp = torch.exp(nll[i] / num_of_models)
        # print("NEG-LOSS {} PPL {}".format(nll[i].item(), curr_pp.item()))
        pps.append(curr_pp.data.item())

    min_index = np.argmin(pps)
    return min_index == target


def get_rank(nll, sz, num_of_models, all_texts_str, target=0):
    global vals, rows
    pps = []
    for i in range(sz):
        curr_pp = torch.exp(nll[i] / num_of_models)
        # print("NEG-LOSS {} PPL {}".format(nll[i].item(), curr_pp.item()))
        pps.append((curr_pp.data.cpu().numpy(), i))

    pps.sort()
    for i, (pp, ind) in enumerate(pps):
        if ind == target:
            return 1.0 / (i + 1)


class InverseNarrative:
    def __init__(self, config, model, path):
        self.config = config
        self.model = model
        self.vocab = AutoTokenizer.from_pretrained(self.config.data.vocab)
        log.info(f"Vocab size: {len(self.vocab)}")
        self.vocab2 = load_pickle(self.config.data.vocab2)
        log.info(f"Vocab2 size: {len(self.vocab2)}")
        dataset = InverseNarrativeDatasetForBart(
            self.config,
            [path],
            vocab=self.vocab,
            min_src_seq_length=MIN_EVAL_SEQ_LEN,
            max_src_seq_length=MAX_EVAL_SEQ_LEN,
        )
        self.dataloader = DataLoader(dataset, 1, shuffle=False)

        if not self.config.max_decode:
            self.config.max_decode = 2000

    @torch.no_grad()
    def test(self, return_all=False):
        print("RANKING")
        num_of_models = len(self.config.train.model.num_of_children)
        ranked_acc = 0.0
        ranked_accs = [0.0] * num_of_models

        rank = 0.0
        ranks = [0.0] * num_of_models

        for iteration, all_texts in enumerate(tqdm(self.dataloader)):
            assert len(all_texts) == 7, "seed, actual, 5 distractions"
            all_texts = to_device(all_texts, self.config.device)
            all_texts_str = [
                self.vocab.batch_decode(text[0]["input_ids"])[0]
                for text in all_texts[1:]
            ]

            src_tup, src_lens = all_texts[0]
            dummy_f_vals = torch.LongTensor([[0, 0, 0, 0, 0]]).to(self.config.device)

            all_texts = all_texts[1:]

            nll = [0.0 for _ in range(6)]
            nlls = [[0] * 6 for _ in range(num_of_models)]
            for j in range(0, len(all_texts)):
                tup = all_texts[j]
                next_tup = copy.deepcopy(tup)

                losses = [0.0] * num_of_models
                words = [0.0] * num_of_models
                total_loss = 0.0
                total_words = 0.0

                data = {
                    "text": (src_tup, src_lens),
                    "frame": dummy_f_vals,
                }
                # Latent and hidden have been initialized with the first tuple
                pred = self.model(data, encode_only=True)

                for i in range(num_of_models):
                    _, logits = self.model.decoder0(
                        tup[0],
                        pred[f"latent_embs_{i}"],
                        pred[f"template_decode_input_{i}"],
                    )

                    ce_loss = masked_cross_entropy(
                        logits, next_tup[0].input_ids, next_tup[1]
                    )

                    losses[i] += ce_loss.item() * next_tup[1].sum()
                    words[i] += next_tup[1].sum()

                    total_loss += ce_loss.item() * next_tup[1].sum()
                    total_words += next_tup[1].sum()

                total_loss = total_loss / total_words
                for i in range(num_of_models):
                    losses[i] = losses[i] / words[i].float()

                NLL = total_loss
                PPL = torch.exp(total_loss)
                # print("Chain-NLL = {}".format(NLL))
                # print("Chain-PPL = {}\n".format(PPL))
                # cprint(f"Iteration {iteration+1}: j {j//2} ppl {PPL}", "yellow")
                # cprint(f"{all_texts_str[j//2]}")

                nll[j] = NLL
                for i, loss in enumerate(losses):
                    NLL = loss
                    PPL = torch.exp(loss)
                    # print("\t"*(i+1), "Chain-NLL_{} = {}".format(i, NLL))
                    # print("\t"*(i+1), "Chain-PPL_{} = {}\n".format(i, PPL))
                    # cprint(f'\t\tIteration {iteration+1}: j {j//2} ppl {PPL}', 'red')
                    nlls[i][j] = NLL

            assert len(nll) == 6, "6 targets."
            for i in range(num_of_models):
                assert len(nlls[i]) == 6, f"6 targets on layer {i+1}"

            ranked_acc += is_it_correct(nll, len(all_texts), num_of_models)
            rank += get_rank(nll, len(all_texts), num_of_models, all_texts_str)
            for i in range(num_of_models):
                ranked_accs[i] += is_it_correct(nlls[i], len(all_texts), 1)
                ranks[i] += get_rank(nlls[i], len(all_texts), 1, all_texts_str)

            if (iteration + 1) == self.config.max_decode:
                print("Max decode reached. Exiting.")
                break

        # ? Ranked Accuracy Calculation
        ranked_acc /= (iteration + 1) * 1 / 100  # multiplying to get percent
        print("Average acc(%): {}\n".format(ranked_acc))

        for i in range(num_of_models):
            ranked_accs[i] /= (iteration + 1) * 1 / 100  # multiplying to get percent
            print("\t" * (i + 1), "Average acc(%)_{}: {}\n".format(i, ranked_accs[i]))

        # ? Rank Calculation
        rank /= (iteration + 1) * 1 / 100  # multiplying to get percent
        print("Average MRR(%): {}\n".format(rank))

        for i in range(num_of_models):
            ranks[i] /= (iteration + 1) * 1 / 100  # multiplying to get percent
            print("\t" * (i + 1), "Average MRR(%)_{}: {}\n".format(i, ranks[i]))

        if return_all:
            return ranked_acc, ranked_accs, rank, ranks
        return ranked_acc
