import json
import pickle
from collections import Counter

from torchtext.vocab import Vocab

UNK_TOKEN = "<unk>"
PAD_TOK = "<pad>"
NOFRAME_TOK = "__NOFRAME__"
SOS_TOK = "<sos>"  # start of sentence
EOS_TOK = "<eos>"  # end of sentence
DUMMY = "<dum>"  # dummy node for zero in degree


def create_vocab(
    max_size=None, min_freq=1, savefile=None, specials=[UNK_TOKEN, NOFRAME_TOK, PAD_TOK]
):
    count = Counter()
    print("SZ", max_size)

    with open("./train_0.9_frame.txt", "r") as f:
        for sen in f:
            frames = sen.split()
            count.update(frames)

    voc = Vocab(count, max_size=max_size, min_freq=min_freq, specials=specials)

    print(voc.itos[:16])

    # print(len(voc), len(count), voc.itos[:], count.most_common(10))
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


SZ = 500
create_vocab(max_size=SZ, savefile="./vocab_frame_500.pickle")
# Coverage:  99.70029245998926
# For 100% coverage use vocab size of 644