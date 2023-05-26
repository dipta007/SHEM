import torch
from torchtext.data import Iterator as BatchIter
from torch.autograd import Variable
import torch.nn.functional as F
import data_utils as du
from torch.distributions import Categorical

from tqdm import tqdm
from main import get_frames_for_upper_layers


def show_inference(model, batch, vocab, vocab2, f_vals, f_ref, args, num_clauses=5):
    latent_gumbels = model.latent_gumbels
    frame_to_vocab = model.frame_to_vocab
    scores = model.scores[0,0,:,:].data.cpu().squeeze()
    _, scores = torch.sort(scores,-1,descending=True)
    scores = scores[:,:15]

    # predicted sentence
    logits = model.logits.transpose(0,1).contiguous()
    logits = logits[0,:,:].data.cpu()
    logits = F.softmax(logits, dim=1)
    pred_sentence = torch.argmax(logits, dim=1)
    pred_sentence = [vocab.itos[int(v.numpy())] for v in pred_sentence]

    latent_gumbels = model.latent_gumbels
    frames_to_frames = model.frames_to_frames
    frame_to_frame = frames_to_frames[0,:,:].data.cpu()
    next_frames, next_frames_indicies = torch.sort(frame_to_frame, -1, descending=True)
    next_frames = next_frames[:,:15]
    next_frames_indicies = next_frames_indicies[:,:15]

    word_to_frame = {}
    real_sentence = batch[0,:].data.cpu()
    real_sentence = [vocab.itos[int(v.numpy())] for v in real_sentence]
    for k in range(min(len(real_sentence), scores.size(0))):
        if k%1==0:
            word_to_frame[real_sentence[k]] = [vocab2.itos[int(v.numpy())] for v in scores[k,:]]
    # print(word_to_frame)

    beta = frame_to_vocab[0,:,:].data.cpu()
    frames_infer = latent_gumbels[0,:,:].data.cpu()
    frames_infer_idx = torch.argmax(frames_infer, -1)
    real_f_vals = f_vals[0,:].data.cpu()
    ref_f = f_ref[0,:].data

    beta_sort, beta_sort_indicies = torch.sort(beta, -1, descending=True)
    beta_sort = beta_sort[: ,:15]
    beta_sort_indicies = beta_sort_indicies[: ,:15]
    topics_dict = {}
    next_frames_dict = {}
    topics_dict['ref_frames'] = []
    topics_dict['fval_frames'] = []
    topics_dict['infered_frames'] = []
    for k in range(num_clauses):
        real_frame = vocab2.itos[real_f_vals[k]]
        which_frame = vocab2.itos[frames_infer_idx[k]]
        args_meaning = [vocab.itos[item.cpu().numpy()] for item in beta_sort_indicies[k]]
        next_frame_meaning = [vocab2.itos[item.cpu().numpy()] for item in next_frames_indicies[k]]
        ref_frames_meaning = [vocab2.itos[item.cpu().numpy()] for item in ref_f]
        topics_dict[which_frame] = args_meaning
        next_frames_dict[which_frame] = next_frame_meaning
        topics_dict['infered_frames'] += [which_frame]
        topics_dict['fval_frames'] += [real_frame]
        topics_dict['ref_frames'] = ref_frames_meaning
    topics_dict['ref_frames'] += ["-"]*(15-len(topics_dict['ref_frames']))
    topics_dict['infered_frames'] += ["-"]*(15-len(topics_dict['fval_frames']))
    topics_dict['fval_frames'] += ["-"]*(15-len(topics_dict['fval_frames']))
    topics_dict['pred_sentence'] = " ".join(pred_sentence)

    return topics_dict, real_sentence, next_frames_dict, word_to_frame


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


def run(args, args_dict):
    if args.cuda and torch.cuda.is_available():
        use_cuda = True
    elif args.cuda and not torch.cuda.is_available():
        print("You do not have CUDA, turning cuda off")
        use_cuda = False
    else:
        use_cuda = False

    #Load the vocab
    vocab , _ = du.load_vocab(args.vocab)
    vocab2 = du.load_vocab(args.frame_vocab_address, is_Frame=True)

    args.total_frames = len(vocab2.stoi.keys())

    # Create the model
    models = get_models(args, use_cuda)

    dataset = du.SentenceDataset(path=args.valid_data, path2=args.valid_frames, vocab=vocab, vocab2=vocab2,
                                num_clauses=args.num_clauses, add_eos=False, is_ref=True, obsv_prob=0.0)
    # Batch size during decoding is set to 1
    batches = BatchIter(dataset, 2, sort_key=lambda x:len(x.text), train=True, device=-1, sort_within_batch=True)

    child_per_layer = list(map(int, args.num_of_children.split()))

    iteration_wo_pad = 0
    with open(f'./inferences/infer_exp_num_{args.exp_num}.txt', 'w') as fp:

        for item in args_dict:
            fp.write(f"{item}: {args_dict[item]}\n")
        fp.write('-'*50)
        fp.write("\n\n")

        for iteration, bl in enumerate(tqdm(batches)):
            with torch.no_grad():
                batch, batch_lens = bl.text
                f_vals, f_vals_lens = bl.frame
                f_ref, _ = bl.ref

                if use_cuda:
                    batch = Variable(batch.cuda())
                    f_vals= Variable(f_vals.cuda())
                else:
                    batch = Variable(batch)
                    f_vals= Variable(f_vals)

                for i, model in enumerate(models):
                    frames_infer = None
                    curr_f_vals = f_vals
                    if i > 0:
                        frames_infer = models[i-1].latent_embs
                        curr_f_vals = torch.argmax(models[i-1].latent_gumbels, -1)
                        curr_f_vals = get_frames_for_upper_layers(args, curr_f_vals, vocab2)

                    _, _, _, _, _, _  = model(batch, batch_lens, frames_infer, f_vals=curr_f_vals)
                    topics_dict, real_sentence, next_frames_dict, word_to_frame = show_inference(model, batch, vocab, vocab2, f_vals, f_ref, args, child_per_layer[i])

                    if topics_dict['ref_frames'].count("<pad>") == 0:
                        if i == 0:
                            iteration_wo_pad += 1
                            fp.write(f"Iteration {iteration_wo_pad}:\n")
                            fp.write("Real Text:".ljust(16))
                            fp.write(f"{' '.join(real_sentence)}\n")

                            fp.write("Real Frames:".ljust(16))
                            fp.write(f"{' '.join(topics_dict['ref_frames'][:5])}\n\n")
                            
                            fp.write("Passed Frames:".ljust(16))
                            fp.write(f"{' '.join(topics_dict['fval_frames'][:5])}\n")
                            
                        else:
                            parent_f_vals = curr_f_vals[0,:]
                            parent_f_vals = [vocab2.itos[item.cpu().numpy()] for item in parent_f_vals]
                            fp.write("Passed Frames:".ljust(16))
                            fp.write(f"{' '.join(parent_f_vals)}\n")
                        
                        decimal_2 = lambda x: str(round(x, 2))
                        pred_frames = models[i].latent_embs
                        entropy = Categorical(probs = pred_frames[0, :]).entropy()
                        entropy = list(map(decimal_2, entropy.tolist()))
                        fp.write(f"Level {i} Entropy:".ljust(16))
                        fp.write(f"{' , '.join(entropy)}\n")

                        fp.write(f"Level {i} Frames:".ljust(16))
                        fp.write(f"{' '.join(topics_dict['infered_frames'][:child_per_layer[i]])}\n")
                        fp.write(f"Level {i} Text:".ljust(16))
                        fp.write(f"{topics_dict['pred_sentence']}\n\n")
                        if i == len(models) - 1:
                            fp.write("\n")
                        # print(iteration_wo_pad)
            
            # break
            if iteration_wo_pad >= 100:
                break