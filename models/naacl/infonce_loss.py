import torch
import torch.nn.functional as F

from utils.utils import safe_log


def get_pos_neg_mask(logits):
    mask = torch.ones(logits.shape[0])
    mask = torch.diag_embed(mask)
    return mask.to(logits.device), (1 - mask).to(logits.device)


def get_all_infonce_loss(config, pred):
    loss = 0.0
    for i in range(5):
        loss += infonce_loss(config, pred["q"], pred[f"p_{i}"])
    loss = loss / 5
    return loss


def infonce_loss(config, q, p):
    sim_qp = torch.einsum("nc,nc->n", [q, p])
    sim_qp = sim_qp / config.train.model.moco_t
    sim_qp = torch.exp(sim_qp)

    sim_qn = torch.einsum("nc,ck->nk", [q, p.T])
    sim_qn = sim_qn / config.train.model.moco_t
    _, neg_mask = get_pos_neg_mask(sim_qn)
    neg = sim_qn.masked_select(neg_mask.bool())
    neg = torch.reshape(neg, (sim_qn.shape[0], sim_qn.shape[1] - 1))
    neg = torch.exp(neg)
    neg = neg.sum(dim=-1)

    pos = sim_qp
    loss = -safe_log(pos / (pos + neg))
    loss = loss.mean()
    return loss
