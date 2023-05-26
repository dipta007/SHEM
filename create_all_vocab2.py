from collections import Counter
import nltk
from pprint import pprint
nltk.download('framenet_v17', download_dir='./')
nltk.data.path.append('./')
from nltk.corpus import framenet as fn
from torchtext.vocab import Vocab
from tqdm import tqdm
import pickle



#Reserved Special Tokens
PAD_TOK = "<pad>"
SOS_TOK = "<sos>" #start of sentence
EOS_TOK = "<eos>" #end of sentence
UNK_TOK = "<unk>"
TUP_TOK = "<TUP>"
DIST_TOK = "<DIST>" # distractor token for NC task
NOFRAME_TOK = "__NOFRAME__"

all_relations = [
  'Using',
  'Precedes',
  'Metaphor', 
  'See_also', 
  'Causative_of', 
  'Inchoative_of', 
  'Inheritance', 
  'Perspective_on', 
  'Subframe',
  'ReFraming_Mapping'
]

SZ = 700

def get_parent_child_framenet(relation):
  relations = relation.split("=")
  parent = relations[1].split(" ")[0]
  child = relations[-1][:-1]
  return parent, child

def create_vocab_for_relation(count, frame, relt):
  try:
    relations = list(fn.frame_relations(frame, type=relt))
    # relations = list(fn.frame_relations(frame))
  except:
    relations = []
  
  for relation in relations:
    parent, child = get_parent_child_framenet(str(relation))
  
    if parent == frame:
      count.update([child])
    else:
      count.update([parent])

def create_vocab_for_all(filename, max_size=None, min_freq=1, savefile=None, specials = [NOFRAME_TOK, UNK_TOK, PAD_TOK, EOS_TOK, SOS_TOK]):
  count = Counter()
  print('SZ', SZ)

  with open(filename, 'r') as fp:
    for line in tqdm(fp):
      for tok in line.split(" "):
        frame = tok.rstrip('\n')
        count.update([frame])

        for relt in all_relations:
          create_vocab_for_relation(count, frame, relt)

    voc = Vocab(count, max_size=max_size, min_freq=min_freq, specials=specials)

    print(voc.itos[:16])
  
  # print(len(voc), len(count), voc.itos[:], count.most_common(10))
  covered = sum([x[1] for x in count.most_common(SZ)])
  total = sum(count.values())
  print('Coverage: ', covered * 100 / total)

  if savefile is not None:
    with open(savefile, 'wb') as fi:
        pickle.dump(voc, fi)
    return None
  else:
      return voc
  return voc
  # for i in range(6000):
  #   try:
  #     frame = fn.frame(i)
  #     frame = frame.name
  #     c.update([frame])
  #   except:
  #     continue

  print(len(c))


def load_vocab(filename,is_Frame=False):
  #load vocab from json file
  with open(filename, 'rb') as fi:
    if is_Frame:
      voc = pickle.load(fi)
      return voc
    else:
      voc,verb_max_idx,config = pickle.load(fi)
      return voc,verb_max_idx

create_vocab_for_all('./data/train_0.6_frame.txt', max_size=SZ, savefile=f'./data/vocab_frame_{SZ}.pkl')
voc = load_vocab(f'./data/vocab_frame_{SZ}.pkl', True)
print(len(voc))

# 500 -> Coverage:  95.20882500168005
# 650 -> Coverage:  98.49903024693657
# 700 -> Coverage:  99.01608025344314
# 800 -> Coverage:  99.66757301270236
# 900 -> Coverage:  99.92040947821218
# 1000 -> Coverage:  99.98322214359382