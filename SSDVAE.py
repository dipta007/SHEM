import torch
import torch.nn as nn
from torch.autograd import Variable
from EncDec import Encoder, Decoder
import torch.nn.functional as F
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK, TUP_TOK

class SSDVAE(nn.Module):
    def __init__(self, emb_size, hsize, vocab, latents, latent_vocab=None, cell_type="GRU", layers=2, bidir=True, pretrained=True, use_cuda=True, dropout=0.10, frame_max=None, latent_dim=None, latent_emb_dim=None, verb_max_idx=None, use_packed=True, base_layer=True, template=20):
        """
        Args:
            emb_size (int) : size of input word embeddings i.e. 400/500/600/700
            hsize (int or tuple) : size of the hidden state (for one direction of encoder). If this is an integer then it is assumed to be the 
                                    size for the encoder, and decoder is set the same. If a Tuple, then it should contain (encoder size, dec size)
            vocab (Vocab object): vocab for words
            latents (LatentNode) : The root of a latent node tree (Note: Size of latent embedding dims should be 2*hsize / 2*hsize[0] if bidir!)
            latent_vocab (Vocab Object): vocab for latents
            cell_type (str) : 'LSTM' or 'GRU'
            layers (int) : number of layers for encoder and decoder
            bidir (bool) : use bidirectional encoder?
            pretrained: use pretrained model or not
            use_cuda: use cuda or not
            dropout: if dropout > 0.0, use dropout
            frame_max: numbder of total unique frames
            latent_dim: tuple (encoder dim, latent_dim)
            latent_vocab_embedding: (frame_max, vocab_size)  defined for p(w|h,f) (reconstruction)
                                    each clause has a frame (frame<= frame_max) we map each of them
                                    to a vocab token
            ?verb_max_idx: not used any more
            use_packed: is it use_packed on rnn or not
            base_layer: is it base layer or not? on base layer, encode words, else encode latent vars
        """
        super(SSDVAE, self).__init__()

        self.embd_size = emb_size # word embedding size: 300
        self.latent_dim = latent_dim  # Frame embedding size: 500
        print('SSDVAE word_emb: ', self.embd_size, 'SSDVAE latent_dim: ', self.latent_dim)

        self.vocab = vocab 
        self.vocab_size = len(self.vocab.stoi.keys()) # size of word vocabulary / number of total unique words # 40000+
        self.sos_idx = self.vocab.stoi[SOS_TOK]
        self.eos_idx = self.vocab.stoi[EOS_TOK]
        self.pad_idx = self.vocab.stoi[PAD_TOK]
        self.tup_idx = self.vocab.stoi[TUP_TOK]

        self.vocab2 = latent_vocab
        self.vocab2_size = frame_max
        self.pad2_idx = self.vocab2.stoi[PAD_TOK]
        
        self.cell_type = cell_type # GRU
        self.layers = layers # 2
        self.bidir = bidir # 1
        
        self.latent_root = latents
        
        self.frame_max = frame_max # 504
        self.use_cuda = use_cuda
        self.use_packed = use_packed
        self.base_layer = base_layer

        self.template = template

        if isinstance(hsize, tuple):
            self.enc_hsize, self.dec_hsize = hsize # 512, 512
        elif bidir:
            self.enc_hsize = hsize
            self.dec_hsize = 2*hsize
        else:
            self.enc_hsize = hsize
            self.dec_hsize = hsize

        in_embedding = nn.Embedding(self.vocab_size, self.embd_size, padding_idx=self.pad_idx if self.base_layer else self.pad2_idx) # 40000+, 300 if self.base_layer else 504, 500
        out_embedding = nn.Embedding(self.vocab_size, self.embd_size, padding_idx=self.pad_idx) # 504, 500

        self.template_to_frame = nn.Linear(self.template, self.frame_max,bias=False)
        self.template_to_vocab = nn.Linear(self.frame_max, self.vocab_size,bias=False)
        self.theta_layer = nn.Linear(self.layers*self.enc_hsize,self.template)

        if pretrained:
            print("Using Pretrained")
            in_embedding.weight.data = self.vocab.vectors
            out_embedding.weight.data = vocab.vectors

        self.encoder = Encoder(self.embd_size, self.enc_hsize,
                                in_embedding, self.cell_type, self.layers, self.bidir, use_cuda=use_cuda, base_layer=self.base_layer)
        self.decoder = Decoder(self.embd_size, self.dec_hsize, self.vocab_size, out_embedding, self.cell_type, self.layers, attn_dim=(self.latent_dim, self.dec_hsize), use_cuda=use_cuda, dropout=dropout)

        self.logits_out = nn.Linear(self.dec_hsize, self.vocab_size) # Weights to calculate logits, out [batch, vocab_size]
        # 512, 500
        self.latent_in = nn.Linear(self.latent_dim, self.layers*self.dec_hsize) # Compute the query for the latents from the last encoder output vector
        # 500, 2*512

        if use_cuda:
            self.decoder = self.decoder.cuda()
            self.encoder = self.encoder.cuda()
            self.logits_out = self.logits_out.cuda()
            self.latent_in = self.latent_in.cuda()
            self.theta_layer = self.theta_layer.cuda()
            self.template_to_frame = self.template_to_frame.cuda()
            self.template_to_vocab = self.template_to_vocab.cuda()

        else:
            self.decoder = self.decoder
            self.encoder = self.encoder
            self.logits_out = self.logits_out
            self.latent_in = self.latent_in
            self.theta_layer = self.theta_layer

    def set_use_cuda(self, value):
        self.use_cuda = value
        self.encoder.use_cuda = value
        self.decoder.use_cuda = value
        self.decoder.attention.use_cuda = value
        self.latent_root.set_use_cuda(value)


    def forward(self, input, seq_lens, prev_latent_embs=None, f_vals=None, beam_size=-1, str_out=False, max_len_decode=50, min_len_decode=0, n_best=1, encode_only=False, template_decode_input=None, template_input=None):
        """
        Args:
            input: (batch_size, max_seq_lens), input of the words in numbers
            seq_lens: (batch_size,) length of the sequences
            prev_latent_embs: (batch_size, latent_emb), previous latent embs for upper layers (not base layers)
            f_vals: Observed frames values
            beam_size: beam size of beam decoding, if beam < 0, then greedy decoding
            str_out: Give the output in text/string (actual output)
            max_len_decode: Maximum length in decoding
            min_len_decode: Minimum length in decoding
            ?n_best: not using any more
            encode_only: Only encoding, no string output/decode
        """
        batch_size = input.size(0)
        if str_out: # use batch size 1 if trying to get actual output
            assert batch_size == 1

        # INIT THE ENCODER
        ehidden = self.encoder.initHidden(batch_size)
        # if not self.base_layer: # For the upper layers
        #     seq_lens = torch.tensor([prev_latent_embs.size(1)] * prev_latent_embs.size(0))
        # enc_output, ehidden = self.encoder(input if self.base_layer else prev_latent_embs, 
        #                                     ehidden, seq_lens, use_packed=self.use_packed) # (batch_size, layers * enc_hidden_size)
        enc_output, ehidden = self.encoder(input, ehidden, seq_lens, use_packed=self.use_packed) # (batch_size, layers * enc_hidden_size)

        # if template_decode_input is None:
        if True:
            enc_theta = self.theta_layer(enc_output).mean(1) # [batch_size,template]
            p_theta_sampled = F.softmax(enc_theta,-1).cuda()
            self.template_input = F.tanh(self.template_to_frame(p_theta_sampled))
            self.template_decode_input = self.template_to_vocab(self.template_input)
        else:
            self.template_input = template_input
            self.template_decode_input = template_decode_input

        if self.use_cuda:
            enc_output_avg = torch.sum(enc_output, dim=1) / Variable(seq_lens.view(-1, 1).type(torch.FloatTensor).cuda())
        else:
            enc_output_avg = torch.sum(enc_output, dim=1) / Variable(seq_lens.view(-1, 1).type(torch.FloatTensor))
        
        initial_query = enc_output_avg
        latent_values, diffs, latent_embs, q_log_q, frames_to_frames, frame_classifier, scores = self.latent_root.forward(enc_output, seq_lens, initial_query, f_vals, template_input=self.template_input) # (batch, num_clauses, num_frames)
        
        self.scores = scores
        self.latent_gumbels = latent_values
        self.frames_to_frames = frames_to_frames
        self.frame_classifier = frame_classifier
        self.q_log_q = q_log_q
        self.latent_embs = latent_embs

        top_level = latent_embs[:, 0, :]
        dhidden = torch.tanh(self.latent_in(top_level).view(self.layers, batch_size, self.dec_hsize))

        if encode_only:
            if self.use_cuda:
                self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim)).cuda())
            else:
                self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim)))
            return dhidden, latent_embs

        if str_out:
            if beam_size <=0:
                # GREEDY Decoding
                if self.use_cuda:
                    self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim).cuda())) #initialize the input feed 0
                else:
                    self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim)))

                return self.greedy_decode(input, dhidden, latent_embs, max_len_decode)
            else:
                # BEAM Decoding
                return self.beam_decode(input, dhidden, latent_embs, beam_size, max_len_decode, min_len_decode=min_len_decode)


        # This is for TRAINING, use teacher forcing
        if self.use_cuda:
            self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim).cuda())) #initialize the input feed 0
        else:
            self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim)))

        return self.train(input, dhidden, latent_embs, diffs)


    def train(self, input, dhidden, latent_embs, diffs, return_hid=False, use_eos=False):
        dec_outputs = []
        logits = []
        batch_size = input.size(0)
        input_size = input.size(1) # Dont need to process last since no eos

        for i in range(input_size):
            # Choose input for this step
            if i == 0:
                tens = torch.LongTensor(input.shape[0]).zero_() + self.sos_idx
                if self.use_cuda:
                    dec_input = Variable(tens.cuda()) # Decoder input init with sos
                else:
                    dec_input = Variable(tens)
            else:
                dec_input = input[:, i-1]
            dec_output, dhidden, logit, frame_to_vocab = self.decoder(dec_input, dhidden, latent_embs, self.template_decode_input)

            dec_outputs += [dec_output]
            logits += [logit]

        dec_outputs = torch.stack(dec_outputs, dim=0) 
        logits = torch.stack(logits, dim=0) 
        self.logits = logits
        self.frame_to_vocab = frame_to_vocab
        if return_hid:
            return latent_embs, self.latent_root, dhidden, dec_outputs, self.template_input, self.template_decode_input
        else:
            self.decoder.reset_feed_() 
            return latent_embs, self.latent_root, diffs, dec_outputs, self.template_input, self.template_decode_input
