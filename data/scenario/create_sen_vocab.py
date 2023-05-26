import json
import pickle
from collections import Counter

from torchtext.vocab import Vocab
from tqdm import tqdm

UNK_TOKEN = "<unk>"
PAD_TOK = "<pad>"
NOFRAME_TOK = "__NOFRAME__"
SOS_TOK = "<sos>"  # start of sentence
EOS_TOK = "<eos>"  # end of sentence
DUMMY = "<dum>"  # dummy node for zero in degree


def create_vocab(
    max_size=None,
    min_freq=1,
    savefile=None,
    specials=[UNK_TOKEN, PAD_TOK, SOS_TOK, EOS_TOK],
):
    count = Counter()
    print("SZ", max_size)

    with open("./train_0.9_TUP.txt", "r") as f:
        for sen in tqdm(f):
            words = sen.split()
            count.update(words)

    voc = Vocab(count, max_size=max_size, min_freq=min_freq, specials=specials)

    print(voc.itos[:100])

    covered = sum([x[1] for x in count.most_common(max_size)])
    total = sum(count.values())
    print("Coverage: ", covered * 100 / total)
    print("For 100% coverage use vocab size of", len(count.keys()))

    if savefile is not None:
        with open(savefile, "wb") as fi:
            pickle.dump(voc, fi)
        return voc
    else:
        return voc


SZ = 50000
create_vocab(max_size=SZ, savefile="./vocab_sen_50k.pickle")
# SZ 50000
# ['<unk>', '<pad>', '<sos>', '<eos>', '<TUP>', 'and', 'in', 'to', 'for', 'he', 'sold', 'it', 'bought', 'who', 'with', 'on', 'at', 'they', 'from', 'that', '$', 'but', 'passed', 'paid', 'i', 'as', 'which', 'pay', 'we', 'went', 'sell', 'by', 'into', 'buy', 'go', 'cost', 'she', 'you', 'about', 'worked', 'moved', 'after', 'through', 'shot', 'or', 'company', 'have', 'sells', 'going', 'people', 'selling', 'retired', 'like', 'purchased', 'had', 'years', 'of', 'fired', 'made', 'over', 'crossed', 'one', 'costs', 'failed', 'under', 'gone', 'according_to', 'work', 'paying', 'without', 'percent', 'pays', 'resigned', 'before', 'take', 'time', 'buying', 'companies', 'than', 'took', 'working', 'during', 'has', 'house', 'came', 'required', 'emerged', 'what', 'out_of', 'move', 'said', 'them', 'put', 'city', 'including', 'come', 'price', 'since', 'get', 'passes']
# Coverage:  99.5820393391676
# For 100% coverage use vocab size of 59581