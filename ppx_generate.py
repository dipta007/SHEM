###########################################
# Model for generating samples from model
#
###########################################
import torch
import argparse
import pickle
import numpy as np
import os
import random

from cloze_utils import generate, find_file

import torch
torch.nn.Module.dump_patches = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHEM')
    parser.add_argument('--impute_with', type=int, default=0)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size')
    parser.add_argument('--seed', type=int, default=11, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--load', type=str,  default='model.pt',help='path to load the final model')
    parser.add_argument('--latent', type=str, help='A str in form of python list')
    parser.add_argument('--beam_size',  type=int, default=-1, help='Beam size')
    parser.add_argument('-perplexity',  action='store_true')
    parser.add_argument('-schema',  action='store_true')
    parser.add_argument('-nohier',  action='store_true')
    parser.add_argument('-max_len_decode', type=int, default=50, help='Maximum prediction length.')
    parser.add_argument('--n_best', type=int, default=1, help="""outputs the n_best decoded sentences""")
    parser.add_argument('--ranking',  action='store_true', help="""N cloze ranking""")
    parser.add_argument('--max_decode', type=int, default=2000, help="""max sentences to be evaluated/decoded.""")
    parser.add_argument('--num_clauses', type=int,default=5)
    parser.add_argument('--obsv_prob',  type=float, default=0, help='Beam size')
    parser.add_argument('--NYT_Noah_type',type=str,help='Noah val or test? inverse narrative cloze?')
    parser.add_argument('--exp_num', type=str,default=5)
    parser.add_argument('--data_mode',default=None, type=str,help="valid or test?")

    path = os.path.dirname(os.path.realpath(__file__))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)    

    args.frame_max = 500

    args.cuda = True
    args.template = 20
    args.vocab='./data/naacl/vocab_40064_verb_max_13572.pkl'
    args.valid_data = './data/naacl/{}_0.6_TUP.txt'.format(str(args.data_mode))
    args.perplexity=True
    args.batch_size=200

    ''' NOTE: We don't use these frames for validation but the data loader needs it (we replace actual frames with __NOFRAME__'''
    args.valid_frames='./data/naacl/{}_0.6_frame.txt'.format(str(args.data_mode))
    #args.valid_narr = '/p/data/rezaee/event-SSSDV/wiki_6_inverse_cloze/test_0.6_TUP_DIST.txt'
    args.frame_vocab_address = './data/naacl/vocab_frame_'+str(args.frame_max)+'.pkl'

    config_prefix = './saved_configs/'
    model_prefix = './saved_models/'

    config_address = config_prefix + 'chain__emb_size_300_nlayers_2_lr_0.001_batch_size_64_seed_{}_bidir_True_num_latent_values_500_latent_dim_500_dropout_0.0_num_clauses_5_obsv_prob_{}_template_20_exp_num_{}_num_of_models_2_num_of_children_5 3.pkl'.format(str(args.seed),str(args.obsv_prob),str(args.exp_num))
    
    config_postfix = find_file(f'exp_num_{str(args.exp_num)}_', config_prefix)
    config_address = config_prefix + config_postfix
    experiment_name = 'wiki_valid_{}_eps_{}_num_{}_seed_{}'.format('chain_event_',str(args.obsv_prob),str(args.exp_num),str(args.seed))

    print('prob: ', args.obsv_prob)
    print('perplexity_data: ', args.valid_data)
    print('config_address: ', config_address)
    with open(config_address, 'rb') as f:
        args_dict, args_info = pickle.load(f)
        model_postfix = config_postfix[:-3] + 'pt'
    
        args.num_of_models = int(args_dict['num_of_models'])
        args.model_prefix = model_prefix
        args.model_postfix = model_postfix

        ppl = generate(args)
        print('perplexity_data: ', args.valid_data)
        args_dict["WikiTestPPL"] = ppl
        print('prob: ', args.obsv_prob)

