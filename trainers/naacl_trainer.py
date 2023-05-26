import logging

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from base.base_dataset import get_data
from base.base_trainer import BaseTrainer
from dataloaders.naacl_dataset import PAD_TOK, load_vocab
from inference.similarity import Similarity
from metrics import get_set_f1, get_set_precision, get_set_recall
from models.naacl.infonce_loss import get_all_infonce_loss
from models.naacl.masked_cross_entropy import masked_cross_entropy
from utils.utils import save_model, wandb_summary

log = logging.getLogger(__name__)


class NaaclTrainer(BaseTrainer):
    def __init__(self, config, model, optimizer):
        self.config = config

        self.vocab = AutoTokenizer.from_pretrained(self.config.data.vocab)
        log.info(f"Vocab size: {len(self.vocab)}")
        self.vocab2 = load_vocab(self.config.data.vocab2)
        log.info(f"Vocab2 size: {len(self.vocab2)}")

        self.config.train.model.vocab_size = len(self.vocab)
        self.config.train.model.num_latent_values = len(self.vocab2)
        self.config.train.model.frame_pad_idx = self.vocab2.stoi[PAD_TOK]

        log.info("Loading Training Data....")
        _, self.train_dataloader = get_data(
            self.config,
            self.config.data.train,
            dataset_args={
                "vocab": self.vocab,
                "vocab2": self.vocab2,
                "obsv_prob": self.config.obsv_prob,
            },
            dataloader_args={
                "batch_size": self.config.data.batch_size,
                "shuffle": True,
            },
        )
        log.info("Loading Validation Data....")
        _, self.val_dataloader = get_data(
            self.config,
            self.config.data.val,
            dataset_args={
                "vocab": self.vocab,
                "vocab2": self.vocab2,
                "obsv_prob": 0.0,
            },
            dataloader_args={
                "batch_size": self.config.data.batch_size,
                "shuffle": False,
            },
        )

        model = model(config=self.config, vocab=self.vocab, vocab2=self.vocab2)
        super().__init__(config, optimizer, model)

        self.best_metric = {
            "hard": -float("inf"),
            "hard_ext": -float("inf"),
            "trans": -float("inf"),
        }

    def get_similarity(self, mode="train"):
        log.info("Testing Hard data...")
        hard = Similarity(
            self.config, self.model, f"./data/inference/hard.txt"
        ).test()
        log.info("Testing Hard extended data...")
        hard_ext = Similarity(
            self.config, self.model, f"./data/inference/hard_extend.txt"
        ).test()
        log.info("Testing Transitive data...")
        trans = Similarity(
            self.config,
            self.model,
            f"./data/inference/transitive.txt",
            hard=False,
        ).test()

        self.log_dict[f"{mode}/hard"] = hard
        self.log_dict[f"{mode}/hard_ext"] = hard_ext
        self.log_dict[f"{mode}/trans"] = trans

        wandb_summary(f"max_{mode}/hard", hard, max)
        wandb_summary(f"max_{mode}/hard_ext", hard_ext, max)
        wandb_summary(f"max_{mode}/trans", trans, max)

    def test(self, mode="train"):
        super().test()

        self.get_similarity(mode)

    def get_loss(self, pred, data, mode):
        target, target_lens = data.target
        total_loss = 0.0
        num_of_models = len(self.config.train.model.num_of_children)
        logprobs, lengths = 0.0, 0.0

        for model_idx in range(num_of_models):
            logits = pred[f"logits_{model_idx}"]
            q_log_q = pred[f"q_log_q_{model_idx}"]
            frame_classifier = pred[f"frame_classifier_{model_idx}"]
            frame_classifier_total = frame_classifier.sum((1, 2))
            frame_classifier_total = -frame_classifier_total.mean()
            q_log_q_total = q_log_q.sum(-1).mean()

            ce_loss = masked_cross_entropy(logits, target["input_ids"], target_lens)

            if model_idx == 0:
                loss = (
                    self.config.train.model.beta_1 * ce_loss
                    + q_log_q_total
                    + frame_classifier_total
                )
            else:
                loss = (
                    self.config.train.model.beta_2 * ce_loss
                    + q_log_q_total
                    + frame_classifier_total
                )

            self.log_dict[f"{mode}/loss_{model_idx}"] += loss.item()
            self.log_dict[f"{mode}/ce_loss_{model_idx}"] += ce_loss.item()
            self.log_dict[f"{mode}/kl_loss_{model_idx}"] += q_log_q_total.item()
            self.log_dict[
                f"{mode}/fcls_loss_{model_idx}"
            ] += frame_classifier_total.item()

            total_loss += loss

            # ? ppl calculation
            logprobs += ce_loss * target_lens.sum()
            lengths += target_lens.sum()

            c_logprobs = ce_loss * target_lens.sum()
            c_lengths = target_lens.sum()
            c_nll = c_logprobs / c_lengths
            c_ppl = torch.exp(c_nll)
            self.log_dict[f"{mode}/nll_{model_idx}"] += c_nll.item()
            self.log_dict[f"{mode}/ppl_{model_idx}"] += c_ppl.item()

        nll = logprobs / lengths
        ppl = torch.exp(nll)

        infonce_loss = self.config.train.model.delta * get_all_infonce_loss(
            self.config, pred
        )
        total_loss += infonce_loss
        self.log_dict[f"{mode}/nce_loss"] += infonce_loss.item()

        self.log_dict[f"{mode}/nll"] += nll.item()
        self.log_dict[f"{mode}/ppl"] += ppl.item()
        self.log_dict[f"{mode}/loss"] += total_loss.item()
        return total_loss

    def run_metrics(self, data, pred, mode):
        def preprocess(frames):
            # print(frames)
            frames = list(filter(lambda x: x > 2, frames))
            frames = list(set(frames))
            return frames

        y_true = data["fref"]
        num_of_models = len(self.config.train.model.num_of_children)
        for model_idx in range(num_of_models):
            latent_gumbels = pred[f"frame_logits_{model_idx}"]
            latent_gumbels = F.softmax(latent_gumbels, dim=-1)
            y_pred = torch.argmax(latent_gumbels, dim=-1)

            precision, recall, f1 = 0.0, 0.0, 0.0
            for y, y_p in zip(y_true, y_pred):
                y = preprocess(y.tolist())
                y_p = preprocess(y_p.tolist())

                n_pre = get_set_precision(y, y_p)
                n_re = get_set_recall(y, y_p)
                n_f1 = get_set_f1(y, y_p)

                precision += n_pre
                recall += n_re
                f1 += n_f1

            precision /= y_true.size(0)
            recall /= y_true.size(0)
            f1 /= y_true.size(0)

            self.log_dict[f"{mode}/precision_{model_idx}"] += precision
            self.log_dict[f"{mode}/recall_{model_idx}"] += recall
            self.log_dict[f"{mode}/f1_{model_idx}"] += f1

        # if mode == "val" and f"{mode}/wiki_inv" not in self.log_dict:
        #     self.wiki_inv = self.inv_narr.test()
        #     self.log_dict[f"{mode}/wiki_inv"] = self.wiki_inv
        # elif mode == "val":
        #     self.log_dict[f"{mode}/wiki_inv"] += self.wiki_inv

    def check_for_improvement(self):
        super().check_for_improvement()

        for key in ["hard", "hard_ext", "trans"]:
            if self.log_dict[f"val/{key}"] >= self.best_metric[key]:
                self.best_metric[key] = self.log_dict[f"val/{key}"]
                save_model(
                    self.model,
                    f"{self.config.train.model_dir}/{self.config.exp_name}_{key}.pt",
                )
