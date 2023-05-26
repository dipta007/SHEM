import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.naacl.attention import Attention

log = logging.getLogger(__name__)


class LatentNode(nn.Module):
    "a node in the latent dag graph, represents a latent variable"

    def __init__(self, config, embeddings, nodeid="0", use_attn=True, layer_idx=0):
        """
        Args
            num_frames (int) : number of total unique frames
            dim (int tuple) :  (query dimension, encoder input (memory) dimension, latent embedding dimension (output))
            nodeid (str) : an optional id for naming the node
            embeddings (nn.Embeddings) : Pass these if you want to create the embeddings, else just go with default
            use_attn: use attention on the latent node or not
            use_cuda: use of cuda or not
            nohier_mode (bool) : Run the NOHIER model instead
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.nodeid = nodeid
        self.embeddings = embeddings

        self.tau = torch.tensor([config.train.model.tau]).to(self.config.device)
        self.num_frames = self.config.train.model.num_latent_values

        if use_attn:
            self.attn = Attention(config)
        else:
            self.attn = None
        self.LogSoftmax = nn.LogSoftmax(dim=-1)

        self.neighbors = []  # list of LatentNodes
        self.parents = []

        self.reset()

    def isroot(self):
        return self.parents == []

    def isleaf(self):
        return self.neighbors == []

    def add_neighbor_(self, neighbor):
        """
        Args
            neighbor (LatenNode) : Latent node to add as a neighbor to this node
        """
        neighbor.parents.append(self)
        self.neighbors.append(neighbor)
        # This is important so the neighbors actually get updated
        self.add_module(neighbor.nodeid, neighbor)

    def reset(self):
        self.latent_emb = None
        self.gumbel_samples = None
        self.q_log_q = None
        self.frame_to_frame_probs = None
        self.vocab_to_frame = None

    def frames_onehot(self, f_vals_clause):
        frames_clause_ = f_vals_clause.unsqueeze(0).unsqueeze(-1)
        one_hot = torch.zeros(
            frames_clause_.size()[0], frames_clause_.size()[1], self.num_frames
        ).to(self.config.device)
        one_hot.scatter_(2, frames_clause_, 1)
        return one_hot

    def frames_multihot(self, f_vals):
        frames_clause_ = f_vals.unsqueeze(0)
        one_hot = torch.zeros(
            frames_clause_.size()[0], frames_clause_.size()[1], self.num_frames
        ).to(self.config.device)
        one_hot.scatter_(2, frames_clause_, 1)
        return one_hot

    def infer_(
        self,
        input_memory,
        input_lens,
        init_query=None,
        f_vals=None,
        template_input=None,
    ):
        """
        Calculate the current value of Z for this variable (deterministic sample),
        given the values of parents and the input memory, then update this variables value
        Args:
            input_memory: (batch_size, seq_len, num_hidden) The input encoder states (what is attended to)
            input_lens: (batch_size, ) it contains the length (just a number) of each input and will be used for masking
            init_query: (batch_size, num_hidden) the average of hiddens from observed tokens
            f_vals: (batch_size, num_clauses) observed (and not) frames
            prev_latent: (batch_size, num_frames) these are the parameters of gumbel_softmax to sample the next frame
        """

        if not self.isroot():
            which_row = int(self.nodeid.split("_")[-1])
            f_val_curr = f_vals[:, which_row]
            # W/O __NO_FRAME__ elements
            l_o = (f_val_curr > 0).view((-1, 1)).float()
            # W/O __NO_FRAME__, <UNK>, <PAD> elements
            l_c = (f_val_curr > 2).view((-1, 1)).float()

            if self.layer_idx == 0:
                fval_one_hot = self.frames_onehot(f_val_curr)
            else:
                fval_one_hot = self.frames_multihot(f_vals)

            prev_latent = self.parents[0].latent_emb
            V, scores, frame_to_frame, vocab_to_frame = self.attn(
                prev_latent, input_memory, input_lens
            )
            V += template_input
        else:
            root_row = 0
            f_val_curr = f_vals[:, root_row]
            # W/O __NO_FRAME__ elements
            l_o = (f_val_curr > 0).view((-1, 1)).float()
            # W/O __NO_FRAME__, <UNK>, <PAD> elements
            l_c = (f_val_curr > 2).view((-1, 1)).float()

            if self.layer_idx == 0:
                fval_one_hot = self.frames_onehot(f_val_curr)
            else:
                fval_one_hot = self.frames_multihot(f_vals)

            V, scores, frame_to_frame, vocab_to_frame = self.attn(
                init_query, input_memory, input_lens
            )  # unnormalized gumbel parameter
            V += template_input

        V_raw = V
        V = V + (l_o * torch.norm(V) * fval_one_hot.squeeze(0))  # hard injection
        self.V = V
        # V = (1-l_o)*V + (l_o * (torch.norm(V) * fval_one_hot.squeeze(0) + V)) # soft injection

        if True or self.layer_idx == 0:
            gumbel_samples = F.gumbel_softmax(logits=V, tau=self.tau)
        else:
            mean_gumbel_samples = torch.zeros_like(V)
            SAMPLE_NO = 2
            for _ in range(SAMPLE_NO):
                gumbel_samples = F.gumbel_softmax(logits=V, tau=self.tau)
                mean_gumbel_samples += gumbel_samples
            gumbel_samples = mean_gumbel_samples / SAMPLE_NO

            # gumbel_samples = F.softmax(V)

        if self.layer_idx == 0:
            w_f = self.config.train.model.gamma_1
            w_kl = self.config.train.model.alpha_1
        else:
            w_f = self.config.train.model.gamma_2
            w_kl = self.config.train.model.alpha_2

        self.frame_classifier = (
            w_f * l_c * self.LogSoftmax(V_raw) * fval_one_hot.squeeze(0)
        )
        frame_to_frame_probs = frame_to_frame.squeeze()
        self.frame_to_frame_probs = frame_to_frame_probs

        self.vocab_to_frame = vocab_to_frame
        self.latent_emb = torch.mm(gumbel_samples, self.embeddings.weight)
        self.gumbel_samples = gumbel_samples  # (batch, num_frames)

        probs = w_kl * (1 - l_o) * F.softmax(V, -1)
        self.q_log_q = (probs * torch.log(probs + 1e-10)).sum(-1)  # (batch, )

    def infer_all_(
        self,
        input_memory,
        input_lens,
        init_query=None,
        f_vals=None,
        template_input=None,
    ):
        self.infer_(
            input_memory, input_lens, init_query, f_vals, template_input=template_input
        )
        for _, neighbor in enumerate(self.neighbors):
            neighbor.infer_all_(
                input_memory,
                input_lens,
                init_query,
                f_vals,
                template_input=template_input,
            )

    def forward(
        self, input_memory, input_lens, init_query, f_vals=None, template_input=None
    ):
        """
        Args:
            input_memory: (batch_size, seq_len, num_hidden)
            input_lens: (batch_size,) it contains the length (just a number)
                            of each input and will be used for masking
            init_query: [batch_size, num_hidden] the average of hiddens from observed tokens
            f_vals: observed (and not) frames (batch, num_clauses)
        """
        self.infer_all_(
            input_memory,
            input_lens,
            init_query,
            f_vals=f_vals,
            template_input=template_input,
        )
        latent_gumbles = self.collect_latent_gumbles()
        latent_embs = self.collect_embs()
        q_log_q = self.collect_q_log_q()
        frames_to_frames = self.collect_frame_to_frame()
        frame_classifier = self.collect_classifier()
        vocab_to_frames = self.collect_vocab_to_frame()
        frame_logits = self.collect_V()
        self.reset_values()
        return (
            frame_logits,
            latent_gumbles,
            latent_embs,
            q_log_q,
            frames_to_frames,
            frame_classifier,
            vocab_to_frames,
        )

    def collect_V(self):
        V_list = [self.V]
        for neighbor in self.neighbors:
            V_list += neighbor.collect_V()

        if self.isroot():
            return torch.stack(V_list, dim=1)
        else:
            return V_list

    def collect_vocab_to_frame(self):
        vocab_to_frames = [self.vocab_to_frame]
        for neighbor in self.neighbors:
            vocab_to_frames += neighbor.collect_vocab_to_frame()

        if self.isroot():
            return torch.stack(vocab_to_frames, dim=1)
        return vocab_to_frames

    def collect_classifier(self):
        frame_classifier = [self.frame_classifier]
        for neighbor in self.neighbors:
            frame_classifier += neighbor.collect_classifier()

        if self.isroot():
            return torch.stack(frame_classifier, dim=1)
        return frame_classifier

    def collect_frame_to_frame(self):
        frame_to_frame_list = [self.frame_to_frame_probs]
        for neighbor in self.neighbors:
            frame_to_frame_list += neighbor.collect_frame_to_frame()

        if self.isroot():
            return torch.stack(frame_to_frame_list, dim=1)
        return frame_to_frame_list

    def collect_q_log_q(self):
        q_log_list = [self.q_log_q]
        for neighbor in self.neighbors:
            q_log_list += neighbor.collect_q_log_q()

        if self.isroot():
            return torch.stack(q_log_list, dim=-1)
        else:
            return q_log_list

    def collect_latent_gumbles(self):
        latent_list = [self.gumbel_samples]
        for neighbor in self.neighbors:
            latent_list += neighbor.collect_latent_gumbles()

        if self.isroot():
            return torch.stack(latent_list, dim=1)
        else:
            return latent_list

    def collect_embs(self):
        emb_list = [self.latent_emb]
        for neighbor in self.neighbors:
            emb_list += neighbor.collect_embs()
        if self.isroot():
            return torch.stack(emb_list, dim=1)
        else:
            return emb_list

    def reset_values(self):
        self.reset()
        for neighbor in self.neighbors:
            neighbor.reset_values()

    def num_of_nodes(self):
        num_of_neighbors = len(self.neighbors)
        for neighbor in self.neighbors:
            num_of_neighbors += neighbor.num_of_nodes()
        return num_of_neighbors


def get_latent_chain(config, layer_idx=0, frame_embedding=None):
    """
    An example function of building trees/dags to use in SSDVAE
    """
    padding_idx = config.train.model.frame_pad_idx
    num_of_neighbors = config.train.model.num_of_children[layer_idx]
    num_frames = config.train.model.num_latent_values

    if frame_embedding is None:
        frame_embedding = nn.Embedding(
            num_frames, config.train.model.latent_dim, padding_idx=padding_idx
        )
    log.info(f"frame_embedding: {frame_embedding.weight.size()}")

    neighbors = [None] * num_of_neighbors
    for i in range(num_of_neighbors):
        id_str = "Level_{}".format(i)

        # if its a root
        if i == 0:
            id_str = "ROOT"

        neighbors[i] = LatentNode(
            config, nodeid=id_str, embeddings=frame_embedding, layer_idx=layer_idx
        )

    for i in range(num_of_neighbors - 2, -1, -1):
        neighbors[i].add_neighbor_(neighbors[i + 1])

    return neighbors[0]
