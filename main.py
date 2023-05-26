########################################
#   module for training the DAVAE model
#
#
########################################
import torch
import torch.nn as nn
from torchtext.data import Iterator as BatchIter
from  torch import distributions
from show_inf import *
import argparse
import numpy as np
import random
import math
from torch.autograd import Variable
from sklearn import metrics
from EncDec import Encoder, Decoder, Attention, fix_enc_hidden, kl_divergence
import torch.nn.functional as F
import data_utils as du
from SSDVAE import SSDVAE
from DAG import example_tree
from masked_cross_entropy import masked_cross_entropy
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK
import time
from torchtext.vocab import GloVe
from report_md import *
import pickle
import gc
import glob
import sys
import os
from framenet_relations import get_frames_for_upper_layers
from cloze_utils import do_ranking
from data_utils import MAX_EVAL_SEQ_LEN, MIN_EVAL_SEQ_LEN

v2 = None


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)

def get_scores_for_frame(model, y_true):
    def preprocess(frames):
        # print(frames)
        frames = list(filter(lambda x: x > 2, frames))
        frames = list(set(frames))
        return frames
    
    def get_score_for_instance(y_true, y_pred):
        precision = len(set(y_true).intersection(set(y_pred))) / (1 if len(y_pred) == 0 else len(y_pred))
        recall = len(set(y_true).intersection(set(y_pred))) / (1 if len(y_true) == 0 else len(y_true))
        f1 = (2 * precision * recall) / (1 if (precision + recall) == 0 else (precision + recall))
        return precision * 100.0, recall * 100.0, f1 * 100.0

    latent_gumbels = model.latent_gumbels
    y_pred = torch.argmax(latent_gumbels, dim=-1)
        
    precision, recall, f1 = 0.0, 0.0, 0.0
    t_st, p_st = "", ""
    for y, y_p in zip(y_true, y_pred):
        n_pre, n_re, n_f1 = get_score_for_instance(preprocess(y.tolist()), preprocess(y_p.tolist()))
        precision += n_pre
        recall += n_re
        f1 += n_f1
        if False:
            t_st += " ".join([v2.itos[v] for v in y])
            t_st += " || "
            p_st += " ".join([v2.itos[v] for v in y_p])
            p_st += " || "

    
    precision /= y_true.size(0)
    recall /= y_true.size(0)
    f1 /= y_true.size(0)

    if False:
        with open('./debug.txt', 'a') as fp:
            fp.write("****** f1\n")
            # fp.write(f"{len(y_pred)}, {len(y_true)}, {len(set(y_true).intersection(set(y_pred)))}")
            fp.write(f"Passed: {t_st}")
            fp.write('\n')
            fp.write(f"Predicted: {p_st}")
            fp.write('\n')
            fp.write(f'Precision {precision}')
            fp.write('\n')
            fp.write(f'Recall {recall}')
            fp.write('\n')
            fp.write(f'F1 {f1}')
            fp.write('\n')
            fp.write("****** f1\n\n")
    return precision, recall, f1


def monolithic_compute_loss(iteration, model, target, target_lens, latent_values, latent_root, diff, dec_outputs, use_cuda, args, train=True, topics_dict=None, real_sentence=None, next_frames_dict=None, word_to_frame=None, show=False, base_layer=True, model_no=0, true_f_vals=[]):
    """
    use this function for validation loss. NO backprop in this function.
    """
    logits = model.logits.transpose(0,1).contiguous() # convert to [batch, seq, vocab]
    q_log_q = model.q_log_q
    frame_classifier = model.frame_classifier
    frame_classifier_total = -frame_classifier.sum((1,2)).mean()
    q_log_q_total = q_log_q.sum(-1).mean()
    precision, recall, f1 = get_scores_for_frame(model, true_f_vals)

    if use_cuda:
        ce_loss = masked_cross_entropy(logits, Variable(target.cuda()), Variable(target_lens.cuda()))
    else:
        ce_loss = masked_cross_entropy(logits, Variable(target), Variable(target_lens))

    if base_layer:
        loss = ce_loss + q_log_q_total + frame_classifier_total
    else:
        loss = ce_loss + q_log_q_total
    
    if train==True and show==True:
        print_iter_stats(iteration, loss, ce_loss, q_log_q_total, topics_dict, real_sentence, next_frames_dict, frame_classifier_total, word_to_frame, args, show=True)
    return loss, ce_loss # tensor


def print_iter_stats(iteration, loss, ce_loss, q_log_q_total, topics_dict, real_sentence, next_frames_dict, frame_classifier_total, word_to_frame, args, show=False):
    if iteration%10==0:
        print("Iteration: ", iteration)
        print("Total: ", loss.cpu().data)
        print("CE: ", ce_loss.cpu().data)
        print("q_log_q_total: ", q_log_q_total.cpu().data)
        print("frame_classifier_total: ", frame_classifier_total.cpu().data)
        print('-'*50)
        if False:
            print("sentence: ", " ".join(real_sentence))
            topics_to_md('chain: ', topics_dict)
            templates = np.arange(args.template).reshape((-1,5))
            topics_to_md('words: ', word_to_frame)
            print('-'*50)


def check_save_model_path(save_model):
    save_model_path = os.path.abspath(save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def run_models(models, batch, batch_lens, target, target_lens, f_vals, f_ref, vocab, vocab2, iteration, use_cuda=True, train=True, args=None):
    tot_loss, tot_ce_loss = 0, 0
    # for model in models:
    #     model.zero_grad()
        
    losses = []
    ce_losses = []
    template_input = None
    template_decode_input = None
    for i, model in enumerate(models):
        base_layer = True
        frames_infer = None
        curr_f_vals = f_vals
        if i > 0:
            base_layer = False
            frames_infer = models[i-1].latent_embs
            curr_f_vals = torch.argmax(models[i-1].latent_gumbels, -1)
            curr_f_vals = get_frames_for_upper_layers(args, curr_f_vals, vocab2, args.obsv_prob)

        latent_values, latent_root, diff, dec_outputs, template_input, template_decode_input = model(batch,  batch_lens,  frames_infer,  f_vals=curr_f_vals, template_decode_input=template_decode_input, template_input=template_input)

        if False:
            topics_dict, real_sentence, next_frames_dict, word_to_frame = show_inference(model, batch, vocab, vocab2, f_vals, f_ref, args)
        else:
            topics_dict, real_sentence, next_frames_dict, word_to_frame = None, None, None, None
        loss, ce_loss = monolithic_compute_loss(iteration, model, target, target_lens, latent_values, latent_root,
                                        diff, dec_outputs, use_cuda, args=args, topics_dict=topics_dict, real_sentence=real_sentence, next_frames_dict=next_frames_dict,
                                        word_to_frame=word_to_frame, train=train, show=True, base_layer=base_layer, model_no=i, true_f_vals=f_ref if i == 0 else curr_f_vals)

        tot_loss += loss
        tot_ce_loss += ce_loss

        losses.append(loss)
        ce_losses.append(ce_loss)

    return tot_loss, tot_ce_loss, losses, ce_losses


def get_wiki_inv_score(args, train, models, vocab, vocab2):
    args.data_mode = 'train' if train else 'valid'
    args.valid_narr = './data/wiki_inv/obs_{}_0.6_TUP_DIST.txt'.format(args.data_mode)
    dataset_wiki = du.NarrativeClozeDataset(args.valid_narr, vocab, src_seq_length=MAX_EVAL_SEQ_LEN, min_seq_length=MIN_EVAL_SEQ_LEN, LM=False)
    wiki_data_len = len(dataset_wiki)
    print('ranking_dataset: ', wiki_data_len)
    # Batch size during decoding is set to 1
    wiki_batches = BatchIter(dataset_wiki, 1, sort_key=lambda x:len(x.actual), train=False, device=-1)
    wiki_acc, wiki_accs = do_ranking(args, models, wiki_batches, vocab, vocab2, wiki_data_len, True, return_all=True)

    return wiki_acc, wiki_accs



def classic_train(args, args_dict, args_info):
    """
    Train the model in the ol' fashioned way, just like grandma used to
    Args
        args (argparse.ArgumentParser)
        args_dict: dict object for args information
        args_info: information on args parser
    """
    if args.cuda and torch.cuda.is_available():
        print("Using cuda")
        use_cuda = True
    elif args.cuda and not torch.cuda.is_available():
        print("You do not have CUDA, turning cuda off")
        use_cuda = False
    else:
        use_cuda = False

    # Load the vocabs
    print("\nLoading Vocab")
    print('args.vocab: ', args.vocab)
    vocab, verb_max_idx = du.load_vocab(args.vocab)
    print("Vocab Loaded, Size {}".format(len(vocab.stoi.keys())))
    print(vocab.itos[:40])
    args_dict["vocab"] = len(vocab.stoi.keys())

    vocab2 = du.load_vocab(args.frame_vocab_address, is_Frame=True)
    # global v2
    # v2 = vocab2
    print("Frames-Vocab Loaded, Size {}".format(len(vocab2.stoi.keys())))
    print(vocab2.itos[:40])
    total_frames = len(vocab2.stoi.keys())
    args_dict["vocab2"] = total_frames
    args.total_frames = total_frames
    args.num_latent_values = args.total_frames
    print('total frames: ', args.total_frames)

    if args.use_pretrained:
        pretrained = GloVe(name='6B', dim=args.emb_size, unk_init=torch.Tensor.normal_)
        vocab.load_vectors(pretrained)
        print("Vectors Loaded")

    # Loading training dataset
    print("Loading Training Dataset")
    dataset = du.SentenceDataset(path=args.train_data, path2=args.train_frames, vocab=vocab, vocab2=vocab2,
                                    num_clauses=args.num_clauses, add_eos=False, is_ref=True, obsv_prob=args.obsv_prob)
    print("Finished Loading Training Dataset {} examples".format(len(dataset)))
    batches = BatchIter(dataset, args.batch_size, sort_key = lambda x:len(x.text), train=True, sort_within_batch=True, device=-1)
    data_len = len(dataset) # 451192

    # Loading validation dataset
    print("Loading Validation Dataset.")
    val_dataset = du.SentenceDataset(path=args.valid_data, path2=args.valid_frames, vocab=vocab, vocab2=vocab2,
                                        num_clauses=args.num_clauses, add_eos=False, is_ref=True, obsv_prob=0.0, print_valid=True)
    print("Finished Loading Validation Dataset {} examples.".format(len(val_dataset)))
    val_batches = BatchIter(val_dataset, args.batch_size, sort_key = lambda x:len(x.text), train=False, sort_within_batch=True, device=-1)


    children = list(map(int, args.num_of_children.split()))
    # Create the model
    if args.load_model:
        print("Loading the Model")
        models = []
        for i in range(args.num_of_models):
            model = torch.load(args.load_model)
            models.append(model)
    else:
        print("Creating the Model")
        bidir_mod = 2 if args.bidir else 1

        models = []

        frame_embedding = nn.Embedding(args.total_frames, args.latent_dim, padding_idx=vocab2.stoi['<pad>'])
        for i in range(args.num_of_models):
            # Base layers
            use_pretrained = args.use_pretrained
            base_layer = True
            use_packed = True
            if i > 0: # Upper layers
                use_pretrained = False
                base_layer = False
                use_packed = False

            latents = example_tree(args.num_latent_values, (bidir_mod*args.enc_hid_size, args.latent_dim), frame_max=args.total_frames,
                                padding_idx=vocab2.stoi['<pad>'], use_cuda=use_cuda, nohier_mode=args.nohier, num_of_childs=children[i], 
                                base_layer=base_layer, frame_embedding=frame_embedding) # assume bidirectional

            hidsize = (args.enc_hid_size, args.dec_hid_size)
            model = SSDVAE(args.emb_size, hidsize, vocab, latents, vocab2, layers=args.nlayers, use_cuda=use_cuda,
                            pretrained=use_pretrained, dropout=args.dropout, frame_max=args.total_frames, template=args.template,
                            latent_dim=args.latent_dim, verb_max_idx=verb_max_idx, use_packed=use_packed, base_layer=base_layer)

            models.append(model)


    # create the optimizer
    if args.load_opt:
        print("Loading the optimizer state")
        optimizer = torch.load(args.load_opt)
    else:
        print("Creating the optimizer anew")
        params = []
        for model in models:
            params += list(model.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr)

    start_time = time.time() # start of epoch 1
    curr_epoch = 1
    min_ppl = 1e10

    # ? Not using this anymore
    for idx,item in enumerate(val_batches):
        if idx==0:
            break
        token_rev = [vocab.itos[int(v.numpy())] for v in item.target[0][-1]]
        frame_rev = [vocab2.itos[int(v.numpy())] for v in item.frame[0][-1]]
        ref_frame = [vocab2.itos[int(v.numpy())] for v in item.ref[0][-1]]

        print('token_rev:', token_rev, len(token_rev), "lengths: ", item.target[1][-1])
        print('frame_rev:', frame_rev, len(frame_rev), "lengths: ", item.frame[1][-1])
        print('ref_frame:', ref_frame, len(ref_frame), "lengths: ", item.ref[1][-1])
        print('-'*50)
    

    patience = 0
    last_improvement, last_it = -1, -1
    # print('Model_named_params:{}'.format(model.named_parameters()))
    for iteration, bl in enumerate(batches): # this will continue on forever (shuffling every epoch) till epochs finished
        batch, batch_lens = bl.text
        f_vals, f_vals_lens = bl.frame
        target, target_lens = bl.target
        f_ref, _ = bl.ref

        if use_cuda:
            batch = Variable(batch.cuda())
            f_vals= Variable(f_vals.cuda())
        else:
            batch = Variable(batch)
            f_vals= Variable(f_vals)

        loss, _, _, _ = run_models(models, batch, batch_lens, target, target_lens, f_vals, f_ref, vocab, vocab2, iteration, use_cuda, args=args)

        # backward propagation
        loss.backward()
        # Gradient clipping
        for model in models:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # Optimizer
        # optimizer.step()
        if (iteration+1)%8 == 0:
            optimizer.step()
            # optimizer.zero_grad()
            for model in models:
                model.zero_grad()

        # End of an epoch - run validation
        if iteration%10==0 or iteration * args.batch_size > data_len * curr_epoch:
            print("\nFinished Training Epoch/iteration {}/{}".format(curr_epoch, iteration))
            # do validation
            valid_logprobs=0.0
            valid_lengths=0.0
            valid_loss = 0.0

            valid_logprobs_per_layer = [0.0] * len(models)
            valid_lengths_per_layer = [0.0] * len(models)
            valid_loss_per_layer = [0.0] * len(models)
            valid_batch_total_size = 0

            with torch.no_grad():
                for v_iteration, bl in enumerate(val_batches):
                    batch, batch_lens = bl.text
                    f_vals, f_vals_lens = bl.frame
                    target, target_lens = bl.target
                    f_ref, _ = bl.ref
                    batch_lens = batch_lens.cpu()
                    if use_cuda:
                        batch = Variable(batch.cuda())
                        f_vals = Variable(f_vals.cuda())
                    else:
                        batch = Variable(batch)
                        f_vals = Variable(f_vals)

                    loss, ce_loss, losses, ce_losses = run_models(models, batch, batch_lens, target, target_lens, f_vals, f_ref, vocab, vocab2, iteration, use_cuda, train=False, args=args)
                    valid_loss += ce_loss.data.clone()
                    valid_logprobs += ce_loss.data.clone().cpu().numpy() * target_lens.sum().cpu().data.numpy()
                    valid_lengths += (target_lens.sum().cpu().data.numpy() * args.num_of_models)
                    
                    for i, cl in enumerate(ce_losses):
                        valid_loss_per_layer[i] += cl.data.clone()
                        valid_logprobs_per_layer[i] += cl.data.clone().cpu().numpy() * target_lens.sum().cpu().data.numpy()
                        valid_lengths_per_layer[i] += target_lens.sum().cpu().data.numpy()
                    
                    valid_batch_total_size += 1


            nll = valid_logprobs / valid_lengths
            ppl = np.exp(nll)
            valid_loss = valid_loss/(v_iteration+1)
            print("\n\nCombined")
            print("**Validation loss {:.2f}.**\n".format(valid_loss.item()))
            print("**Validation NLL {:.2f}.**\n".format(nll))
            print("**Validation PPL {:.2f}.**\n".format(ppl))
            print()

            for i, (valid_logprobs, valid_lengths, valid_loss) in enumerate(zip(valid_logprobs_per_layer, valid_lengths_per_layer, valid_loss_per_layer)):
                c_nll = valid_logprobs / valid_lengths
                c_ppl = np.exp(c_nll)
                c_valid_loss = valid_loss/(v_iteration+1)
                print("\t"*(i+1), "Level ", i)
                print("\t"*(i+1), "**Validation loss {:.2f}.**\n".format(c_valid_loss.item()))
                print("\t"*(i+1), "**Validation NLL {:.2f}.**\n".format(c_nll))
                print("\t"*(i+1), "**Validation PPL {:.2f}.**\n".format(c_ppl))
                print()
                if ppl < min_ppl:
                    args_dict[f'min_ppl_{i}'] = c_ppl

            if ppl < min_ppl:
                min_ppl = ppl
                last_improvement, last_it = curr_epoch, iteration
                args_dict["min_ppl"] = min_ppl
                dir_path = os.path.dirname(os.path.realpath(__file__))
                # dir_path = '/p/work/xxxxS'
                not_include_in_file_name = ["min_ppl", "vocab", "vocab2", "frame_max"]
                save_file = "".join(["_"+str(key)+"_"+str(value) for key,value in args_dict.items() if key not in not_include_in_file_name and not key.startswith('min_ppl')])
                args_to_md(model="chain", args_dict=args_dict)
                for i, model in enumerate(models):
                    model_path = os.path.join(dir_path+f"/saved_models/model_{i}_chain_"+save_file+".pt")
                    torch.save(model, model_path)
                config_path = os.path.join(dir_path+"/saved_configs/chain_"+save_file+".pkl")
                with open (config_path, "wb") as f:
                    pickle.dump((args_dict, args_info), f)

                patience = 0
            
            print("\n")
            print(f'\t==> Last improvement on Epoch: {last_improvement}, Iteration: {last_it}')
            print('\t==> min_ppl {:4.4f} '.format(min_ppl))
            for i in range(args.num_of_models):
                print('\t==> min_ppl of', i, '{:4.4f} '.format(args_dict[f'min_ppl_{i}']))
            print("\n\n")

            if ppl > min_ppl and iteration * args.batch_size > data_len * curr_epoch:
                patience += 1
                if patience > 10:
                    break

        # Increase the number of epoch
        if iteration * args.batch_size > data_len * curr_epoch:
            get_wiki_inv_score(args, False, models, vocab, vocab2)
            print(f'Epoch {curr_epoch} took time: ', time.time() - start_time)
            curr_epoch += 1
            start_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SSDVAE_ext')
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument('--vocab', type=str, help='the vocabulary pickle file')
    parser.add_argument('--emb_size', type=int, default=300, help='size of word embeddings')
    parser.add_argument('--enc_hid_size', type=int, default=512, help='size of encoder hidden')
    parser.add_argument('--dec_hid_size', type=int, default=512, help='size of encoder hidden')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--log_every', type=int, default=200)
    parser.add_argument('--save_after', type=int, default=500)
    parser.add_argument('--validate_after', type=int, default=2500)
    parser.add_argument('--optimizer', type=str, default='adam', help='adam, adagrad, sgd')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='batch size')
    parser.add_argument('--seed', type=int, default=11, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--bidir', type=bool, default=True, help='Use bidirectional encoder')
    # parser.add_argument('-src_seq_length', type=int, default=50, help="Maximum source sequence length")
    parser.add_argument('-max_decode_len', type=int, default=50, help='Maximum prediction length.')
    parser.add_argument('-save_model', default='model', help="""Model filename""")
    parser.add_argument('-num_latent_values', type=int, default=400, help='How many values for each categorical value')
    parser.add_argument('-latent_dim', type=int, default=512, help='The dimension of the latent embeddings')
    parser.add_argument('-use_pretrained', type=bool, default=True, help='Use pretrained glove vectors')
    parser.add_argument('-commit_c', type=float, default=0.25, help='loss hyperparameters')
    parser.add_argument('-commit2_c', type=float, default=0.15, help='loss hyperparameters')
    parser.add_argument('-dropout', type=float, default=0.0, help='loss hyperparameters')
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--num_clauses', type=int, default=5)
    parser.add_argument('--load_opt', type=str)
    parser.add_argument('--nohier', action='store_true', help='use the nohier model instead')
    parser.add_argument('--frame_max', type=int, default=700)
    parser.add_argument('--obsv_prob', type=float, default=1.0,help='the percentage of observing frames')
    parser.add_argument('--template', type=int, default=20)
    parser.add_argument('--exp_num', type=str, default=1)
    parser.add_argument('--max_decode', type=int, default=2000, help="""max sentences to be evaluated/decoded.""")


    args = parser.parse_args()
    path = os.path.dirname(os.path.realpath(__file__))
    args.model = 'chain'
    args.command = ' '.join(sys.argv)

    args.train_data = './data/naacl/train_0.6_TUP.txt'
    args.train_frames = './data/naacl/train_0.6_frame.txt'

    args.valid_data = './data/naacl/valid_0.6_TUP.txt'
    args.valid_frames = './data/naacl/valid_0.6_frame.txt'

    args.test_data = './data/naacl/test_0.6_TUP.txt'
    args.vocab = './data/naacl/vocab_40064_verb_max_13572.pkl'
    args.frame_vocab_address = './data/naacl/vocab_frame_scenerio_'+str(args.frame_max)+'.pkl'
    args.frame_vocab_ref = './data/naacl/vocab_frame_all.pkl'
    args.latent_dim = args.frame_max
    args.num_latent_values = args.frame_max

    # Change here for the layer formation
    args.num_of_models = 2
    args.num_of_children = "5 3"

    args_info = {}
    for arg in vars(args):
        args_info[arg] = getattr(args, arg)
    print('parser_info:')
    for item in args_info:
        print(item, ": ", args_info[item])
    print('-'*50)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    keys = ["emb_size", "nlayers", "num_of_models", "num_of_children",
            "lr", "batch_size", "num_clauses", "num_latent_values", "template",
            "latent_dim", "dropout", "bidir", "obsv_prob", "frame_max", "exp_num", "seed"]
    args_dict = {key : str(value) for key,value in vars(args).items() if key in keys}

    experiment_name = f"exp_{args_dict['exp_num']}"

    classic_train(args, args_dict, args_info)
