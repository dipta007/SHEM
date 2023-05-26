###########################################
# Model for generating samples from model
#
###########################################
import torch
from torchtext.data import Iterator as BatchIter
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import data_utils as du
from tqdm import tqdm
import copy
import os
import csv 


from DAG import example_tree
from EncDec import Encoder, Decoder, Attention, fix_enc_hidden
from masked_cross_entropy import masked_cross_entropy
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK, TUP_TOK, MAX_EVAL_SEQ_LEN, MIN_EVAL_SEQ_LEN
from decode_utils import transform, get_tups, get_pred_events
from framenet_relations import get_frames_for_upper_layers


def find_file(name, path):
    found = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if name in file and root == path:
                found.append(file)
    if len(found) > 1:
        for i, f in enumerate(found):
            print(f'{i}. {f}')
        print()

        ind = input(f'{len(found)} files found which one to choose?    ')
        ind = int(ind)
        return found[ind]
    return found[0]

def get_models(args, use_cuda):
    models = []
    for i in range(args.num_of_models):
        model_address = args.model_prefix + f'model_{i}_' + args.model_postfix
        with open(model_address, 'rb') as fi:
            if not use_cuda:
                model = torch.load(fi, map_location=lambda storage, loc : storage)
            else:
                model = torch.load(fi, map_location=torch.device('cuda'))
        models.append(model)

    for model in models:
        if not hasattr(model.latent_root, 'nohier'):
            model.latent_root.set_nohier(args.nohier) #for backwards compatibility

        model.decoder.eval()
        model.set_use_cuda(use_cuda)

    return models

def generate(args):
    """
    Use the trained model for decoding
    Args
        args (argparse.ArgumentParser)
    """
    if args.cuda and torch.cuda.is_available():
        device = 0
        use_cuda = True
    elif args.cuda and not torch.cuda.is_available():
        print("You do not have CUDA, turning cuda off")
        device = -1
        use_cuda = False
    else:
        device = -1
        use_cuda=False

    #Load the vocab
    vocab , _ = du.load_vocab(args.vocab)
    vocab2 = du.load_vocab(args.frame_vocab_address,is_Frame=True)

    args.total_frames = len(vocab2.stoi.keys())

    eos_id = vocab.stoi[EOS_TOK]
    pad_id = vocab.stoi[PAD_TOK]

    # Create the model
    models = get_models(args, use_cuda)

    if args.ranking: # default is HARD one, the 'Inverse Narrative Cloze' in the paper
        dataset = du.NarrativeClozeDataset(args.valid_narr, vocab, src_seq_length=MAX_EVAL_SEQ_LEN, min_seq_length=MIN_EVAL_SEQ_LEN, LM=False)
        print('ranking_dataset: ',len(dataset))
        # Batch size during decoding is set to 1
        batches = BatchIter(dataset, 1, sort_key=lambda x:len(x.actual), train=False, device=-1)
    else:
        # dataset = du.SentenceDataset(args.valid_data, vocab, src_seq_length=MAX_EVAL_SEQ_LEN, min_seq_length=MIN_EVAL_SEQ_LEN, add_eos=False) # put in filter pred later
        dataset = du.SentenceDataset(path=args.valid_data, path2=args.valid_frames, vocab=vocab, vocab2=vocab2, num_clauses=args.num_clauses, add_eos=False, is_ref=True, obsv_prob=0.0, print_valid=True)
        # Batch size during decoding is set to 1
        batches = BatchIter(dataset, args.batch_size, sort_key=lambda x:len(x.text), train=False, device=-1)

    data_len = len(dataset)

    # Create the model
    

    # For reconstruction
    if args.perplexity:
        print('calculating perplexity')
        loss, losses = calc_perplexity(args, models, batches, vocab, vocab2, data_len)
        
        NLL = loss
        PPL = np.exp(loss)
        print("Chain-NLL = {}".format(NLL))
        print("Chain-PPL = {}\n".format(PPL))

        for i, loss in enumerate(losses):
            NLL = loss
            PPL = np.exp(loss)
            print("\t"*(i+1), "Chain-NLL_{} = {}".format(i, NLL))
            print("\t"*(i+1), "Chain-PPL_{} = {}\n".format(i, PPL))

        return PPL
    elif args.schema:
        generate_from_seed(args, models, batches, vocab, data_len)
    elif args.ranking: # True
        ranked_acc = do_ranking(args, models, batches, vocab, vocab2, data_len, use_cuda)
        return ranked_acc
    else:
#        sample_outputs(models, vocab)
        reconstruct(args, models, batches, vocab)


def is_it_correct(nll, sz, num_of_models, target=0):
    pps = []
    for i in range(sz):
        curr_pp = torch.exp(nll[i] / num_of_models)
        # print("NEG-LOSS {} PPL {}".format(nll[i].item(), curr_pp.item()))
        pps.append(curr_pp.data.cpu().numpy())

    min_index = np.argmin(pps)
    return min_index == target


# Inverse Narrative Cloze
def do_ranking(args, models, batches, vocab, vocab2, data_len, use_cuda, return_all=False):
    print("RANKING")
    ranked_acc = 0.0
    ranked_accs = [0.0] * len(models)

    tup_idx = vocab.stoi[TUP_TOK]
    all_csv = []

    for iteration, bl in enumerate(tqdm(batches)):
        all_texts = [bl.actual, bl.actual_tgt, bl.dist1, bl.dist1_tgt, bl.dist2, bl.dist2_tgt, bl.dist3, bl.dist3_tgt, bl.dist4, bl.dist4_tgt, bl.dist5, bl.dist5_tgt] # each is a tup

        assert len(all_texts) == 12, "12 = 6 * 2."

        with torch.no_grad():
            all_texts_vars = []

            for tup in all_texts:
                all_texts_vars.append((Variable(tup[0]), tup[1]))

            # run the model for all 6 sentences
            first_tup = -1
            for i in range(bl.actual[0].shape[1]):
                if bl.actual[0][0, i] == tup_idx:
                    first_tup = i
                    break
            if first_tup == -1:
                print("WARNING: First TUP is -1")
            src_tup = Variable(bl.actual[0][:, :first_tup+1].view(1, -1))
            src_lens = torch.LongTensor([src_tup.shape[1]])
            f_vals = torch.LongTensor([[0,0,0,0,0]])

            if use_cuda:
                src_tup = src_tup.cuda()
                src_lens = src_lens.cuda()
                f_vals = f_vals.cuda()

            nll = []
            nlls = []
            for i, model in enumerate(models):
                nlls.append([])
                # initialize for every model
                # will iterate 2 at a time using iterator and next
                vars_iter = iter(all_texts_vars)
                
                frames_infer = None
                curr_f_vals = f_vals
                if i > 0:
                    frames_infer = models[i-1].latent_embs
                    curr_f_vals = torch.argmax(models[i-1].latent_gumbels, -1)
                    curr_f_vals = get_frames_for_upper_layers(args, curr_f_vals, vocab2)

                dhidden, latent_embs = model(src_tup, src_lens, frames_infer, f_vals=curr_f_vals, encode_only=True)

                # Latent and hidden have been initialized with the first tuple
                for j, tup in enumerate(vars_iter):
                    ## INIT FEED AND DECODE before every sentence.
                    if use_cuda:
                        model.decoder.init_feed_(Variable(torch.zeros(1, model.decoder.attn_dim).cuda()))
                    else:
                        model.decoder.init_feed_(Variable(torch.zeros(1, model.decoder.attn_dim)))

                    next_tup = next(vars_iter)
                    if use_cuda:
                        _, _, _, dec_outputs, _, _  = model.train(tup[0].cuda(), dhidden, latent_embs, [])
                    else:
                        _, _, _, dec_outputs, _, _  = model.train(tup[0], dhidden, latent_embs, [])

                    logits = model.logits.transpose(0,1).contiguous() # convert to [batch, seq, vocab]
                    curr_nll = masked_cross_entropy(logits, next_tup[0].cuda(), Variable(next_tup[1]).cuda())

                    nlls[i] += copy.deepcopy([curr_nll])

                    if i == 0: # For the first model
                        nll += copy.deepcopy([curr_nll])
                    else: # For the other models
                        nll[j] += curr_nll

            assert len(nll) == 6, "6 targets."
            for i in range(len(models)):
                assert len(nlls[i]) == 6, f"6 targets on layer {i+1}"
            
            ranked_acc += is_it_correct(nll, len(all_texts_vars) // 2, args.num_of_models)
            for i in range(len(models)):
                ranked_accs[i] += is_it_correct(nlls[i], len(all_texts_vars) // 2, 1)

            # CSV generation
            now = []
            for i in range(len(all_texts_vars) // 2):
                curr = [iteration]
                txt = " ".join([vocab.itos[v] for v in all_texts[i*2][0][0]])
                curr.extend([txt])
                for j in range(len(models)):
                    ppl = torch.exp(nlls[j][i] / 1).item()
                    curr.append(str(round(ppl, 2)))
                    curr.append(is_it_correct(nlls[j], len(all_texts_vars) // 2, 1))
                ppl_combined = torch.exp(nll[i] / args.num_of_models).item()
                curr.append(str(round(ppl_combined, 2)))
                curr.append(is_it_correct(nll, len(all_texts_vars) // 2, args.num_of_models))
                curr.append(i == 0)
                now.append(curr)
                # print(iteration, txt, ppl, is_it_correct(nlls[i], len(all_texts_vars) // 2, 1), ppl_combined, is_it_correct(nll, len(all_texts_vars) // 2, args.num_of_models), i==0)
            # print(now)
            now.append([])
            all_csv.extend(now)

            # low perplexity == top ranked sentence - correct answer is the first one of course
            
            # all_texts_str = [transform(text[0].data.numpy()[0], vocab.itos) for text in all_texts_vars]
            # for v in all_texts_str:
            #     print(v)
            # print("ALL: {}".format(all_texts_str))
            # min_index = np.argmin(pps)
            # if min_index == 0:
            #     ranked_acc += 1
                # print("TARGET: {}".format(transform(all_texts_vars[1][0].data.numpy()[0], vocab.itos)))
                # print("CORRECT: {}".format(transform(all_texts_vars[1][0].data.numpy()[0], vocab.itos)))
            #else:
                # print the ones that are wrong
                # print("TARGET: {}".format(transform(all_texts_vars[1][0].data.numpy()[0], vocab.itos)))
                # print("WRONG: {}".format(transform(all_texts_vars[min_index+2][0].data.numpy()[0], vocab.itos)))

            if (iteration+1) == args.max_decode:
                print("Max decode reached. Exiting.")
                break

    ranked_acc /= (iteration+1) * 1/100 # multiplying to get percent
    print("Average acc(%): {}\n".format(ranked_acc))

    for i in range(len(models)):
        ranked_accs[i] /= (iteration+1) * 1/100 # multiplying to get percent
        print("\t"*(i+1), "Average acc(%)_{}: {}\n".format(i, ranked_accs[i]))

    if return_all:
        return ranked_acc, ranked_accs
    return ranked_acc


def calc_perplexity_avg_line(args, model, batches, vocab, data_len):
    total_loss = 0.0
    iters = 0
    for iteration, bl in enumerate(batches):
        print(iteration)
        batch, batch_lens = bl.text
        target, target_lens = bl.target
        if args.cuda:
            batch = Variable(batch.cuda(), volatile=True)
        else:
            batch = Variable(batch, volatile=True)

        _, _, _, dec_outputs, _, _  = model(batch, batch_lens)

        logits = model.logits_out(dec_outputs).cpu()

        logits = logits.transpose(0,1).contiguous() # convert to [batch, seq, vocab]

        ce_loss = masked_cross_entropy(logits, Variable(target), Variable(target_lens))
        total_loss = total_loss + ce_loss.data[0]

        iters += 1

    print(iters)
    print(data_len)

    return total_loss / data_len

def calc_perplexity(args, models, batches, vocab, vocab2, data_len):
    total_loss = 0.0
    iters = 0
    total_words = 0

    losses = [0.0] * args.num_of_models
    words = [0] * args.num_of_models
    for bl in tqdm(batches):
        batch, batch_lens = bl.text
        target, target_lens = bl.target
        f_vals, f_vals_lens = bl.frame

        with torch.no_grad():
            if args.cuda:
                batch = Variable(batch.cuda())
                f_vals = Variable(f_vals.cuda())
            else:
                batch = Variable(batch)
                f_vals = Variable(f_vals.cuda())

            for i, model in enumerate(models):
                frames_infer = None
                curr_f_vals = f_vals
                if i > 0:
                    frames_infer = models[i-1].latent_embs
                    curr_f_vals = torch.argmax(models[i-1].latent_gumbels, -1)
                    curr_f_vals = get_frames_for_upper_layers(args, curr_f_vals, vocab2)

                _, _, _, _, _, _  = model(batch, batch_lens, frames_infer, f_vals=curr_f_vals)
                logits = model.logits.transpose(0,1).contiguous() # convert to [batch, seq, vocab]

                ce_loss = masked_cross_entropy(logits, Variable(target).cuda(), Variable(target_lens).cuda())

                losses[i] += ce_loss.cpu().item() * target_lens.float().sum()
                words[i] += target_lens.sum()
                
                total_loss += ce_loss.cpu().item() * target_lens.float().sum()
                total_words += target_lens.sum()

            iters += 1

    print(iters)
    print(data_len)

    total_avg_loss = total_loss / total_words.float()
    avg_losses = []
    for i in range(args.num_of_models):
        curr_ppl = losses[i] / words[i].float()
        avg_losses.append(curr_ppl)

    return total_avg_loss, avg_losses


def sample_outputs(model, vocab):
    model.latent_root.prune_()
    for _ in range(100):
        val1 = np.random.randint(313)
        val2 = np.random.randint(32)
        val3 = np.random.randint(38)
        val4 = np.random.randint(12)
        val5 = np.random.randint(6)
        values = [247,12,15,val4,1]
        outputs = model.decode(values)

        print("Reconstruct: {}\n\n".format(transform(outputs, vocab.itos)))


def generate_from_seed(args, model, batches, vocab, data_len):
    """
    Generate a script from a seed tuple
    Args
        args (argparse.ArgumentParser)
        seeds (BatchIter) : BatchIter object for a file of seeds, the seed file should be in the
        same format as normal validation data
    """
    for iteration, bl in enumerate(batches):
        batch, batch_lens = bl.text
        target, target_lens = bl.target
        if args.cuda:
            batch = Variable(batch.cuda(), volatile=True)
        else:
            batch = Variable(batch, volatile=True)


        src_lens= torch.LongTensor([batch.size(1)])
        dhidden, latent_values = model(batch, src_lens, encode_only=True) #get latent encoding for seed
        model.decoder.init_feed_(Variable(torch.zeros(1, model.decoder.attn_dim)))
        _, _, dhidden, dec_outputs  = model.train(batch, 1, dhidden, latent_values, [], return_hid=True)  #decode seed

        #print("seq len {}, decode after {} steps".format(seq_len, i+1))
        # beam set current state to last word in the sequence
        beam_inp = batch[:, -1]

                # init beam initializesthe beam with the last sequence element
        outputs = model.beam_decode(beam_inp, dhidden, latent_values, args.beam_size, args.max_len_decode, init_beam=True)


        print("TRUE: {}".format(transform(batch.data.squeeze(), vocab.itos)))
        print("Reconstruct: {}\n\n".format(transform(outputs, vocab.itos)))


def reconstruct(args, model, batches, vocab):
    for iteration, bl in enumerate(batches):
        batch, batch_lens = bl.text
        target, target_lens = bl.target
        if args.cuda:
            batch = Variable(batch.cuda(), volatile=True)
        else:
            batch = Variable(batch, volatile=True)

        outputs = model(batch, batch_lens, str_out=True, beam_size=args.beam_size, max_len_decode=args.max_len_decode)

        print("TRUE: {}".format(transform(batch.data.squeeze(), vocab.itos)))
        print("Reconstruct: {}\n\n".format(transform(outputs, vocab.itos)))


def schema_constraint(cands, prev_voc, curr_verbs, min_len_decode=0, step=0, eos_idx=EOS_TOK):
    """
    Constraints to use during decoding,
    Prevents the model from producing schemas that are obviously wrong (have repeated
    predicates or the same arguments as subject and object

    Args:
        cands (Tensor [batch x vocab]) : the probabilities over the vocab for each batch/beam
        prev_voc (Tensor [batch]) : the previous output for each batch/beam
        curr_verbs (list of lists [batch x *]) : A list of lists whose kth element is a list of vocab ids of previously used
        predicates in the kth beam
        tup_idx (int) : the vocab id of the <TUP> symbol
    """
    LOW = -1e20
    K = cands.shape[0]

    for i in range(K): #for each beam
        #Replace previous vocabulary items with low probability
        beam_prev_voc = prev_voc[i]
        cands[i, beam_prev_voc] = LOW

        #Replace verbs already used with low probability
        for verb in curr_verbs[i]:
            cands[i, verb] = LOW

        if step < min_len_decode:
            cands[i, eos_idx] = LOW

    return cands


def update_verb_list(verb_list, b, tup_idx=4):
    """
    Update currently used verbs for Beam b
    verb_list is a beam_size sized list of list, with the ith list having a list of verb ids used in the ith beam
    so far
    """
    #First need to update based on prev ks
    if len(b.prev_ks) > 1:
        new_verb_list = [[]]*b.size
        for i in range(b.size):
            new_verb_list[i] = list(verb_list[b.prev_ks[-1][i]])
    else:
        new_verb_list =verb_list

    #update the actual lists
    if len(b.next_ys) == 2:
        for i, li in enumerate(new_verb_list):
            li.append(b.next_ys[-1][i])

    elif len(b.next_ys) > 2:
        for i, li in enumerate(new_verb_list):
            if b.next_ys[-2][b.prev_ks[-1][i]] == tup_idx:
                li.append(b.next_ys[-1][i])

    return new_verb_list