from collections import Counter
import nltk
from pprint import pprint
nltk.download('framenet_v17', download_dir='../../')
nltk.data.path.append('../../')
from nltk.corpus import framenet as fn
from torchtext.vocab import Vocab
from tqdm import tqdm
import pickle
from collections import defaultdict


#Reserved Special Tokens
PAD_TOK = "<pad>"
SOS_TOK = "<sos>" #start of sentence
EOS_TOK = "<eos>" #end of sentence
UNK_TOK = "<unk>"
TUP_TOK = "<TUP>"
DIST_TOK = "<DIST>" # distractor token for NC task
NOFRAME_TOK = "__NOFRAME__"

SZ = 500

scenarios = fn.frames(r'(?i)_scenario')
scenario_dict = {s.name: s for s in scenarios}
frame_to_scenario = defaultdict(set)

# print(scenario_dict)
for scenario_name, scenario_obj in scenario_dict.items():
  for sub in scenario_obj.frameRelations:
    frame_to_scenario[sub['subFrame'].name].add(scenario_obj.name)

def create_vocab_for_scenerio(count, frame):
  for subframe in frame_to_scenario[frame]:
    count.update([subframe])

def create_vocab_for_all(filename, max_size=None, min_freq=1, savefile=None, specials = [NOFRAME_TOK, UNK_TOK, PAD_TOK, EOS_TOK, SOS_TOK]):
  count = Counter()
  print('SZ', SZ)

  with open(filename, "r") as f:
    for line in f:
        
        for tok in line.strip().split(" "):
            frame = tok.strip('\n')
            count.update([frame])

            create_vocab_for_scenerio(count, frame)

    voc = Vocab(count, max_size=max_size, min_freq=min_freq, specials=specials)

    print(voc.itos[:16])
  
  # print(len(voc), len(count), voc.itos[:], count.most_common(10))
  covered = sum([x[1] for x in count.most_common(SZ)])
  total = sum(count.values())
  print("Coverage: ", covered * 100 / total)
  print("For 100% coverage use vocab size of", len(count.keys()))

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

create_vocab_for_all('./train_0.9_frame.txt', max_size=SZ, savefile=f'./vocab_frame_scenerio_{SZ}.pickle')

SZ = 10000
create_vocab_for_all('./train_0.9_frame.txt', max_size=SZ, savefile=f'./vocab_frame_all.pickle')

# SZ 500
# ['__NOFRAME__', '<unk>', '<pad>', '<eos>', '<sos>', 'Commerce_scenario', 'Motion_scenario', 'Commerce_sell', 'Motion', 'Commerce_buy', 'Commerce_pay', 'Traversing', 'Employee_scenario', 'Cycle_of_existence_scenario', 'Result_of_attempt_scenario', 'Expensiveness']
# Coverage:  99.70606996469556
# For 100% coverage use vocab size of 705
# SZ 10000
# ['__NOFRAME__', '<unk>', '<pad>', '<eos>', '<sos>', 'Commerce_scenario', 'Motion_scenario', 'Commerce_sell', 'Motion', 'Commerce_buy', 'Commerce_pay', 'Traversing', 'Employee_scenario', 'Cycle_of_existence_scenario', 'Result_of_attempt_scenario', 'Expensiveness']
# Coverage:  100.0
# For 100% coverage use vocab size of 705