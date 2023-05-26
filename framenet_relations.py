import nltk
from pprint import pprint
nltk.download('framenet_v17', download_dir='./')
nltk.data.path.append('./')
from nltk.corpus import framenet as fn
import torch

frame_relation = "Inheritance"

def get_parent_child_framenet(relation):
  relations = relation.split("=")
  parent = relations[1].split(" ")[0]
  child = relations[-1][:-1]
  return parent, child

def get_parent_child_mapping(frame_max, vocab2):
  mp = {}
  for i in range(frame_max):
    frame = vocab2.itos[i]
    if i not in mp:
      mp[i] = []
    try:
      relations = list(fn.frame_relations(frame, type=frame_relation))
      # relations = list(fn.frame_relations(frame))
    except:
      relations = []

    for relation in relations:
      parent, child = get_parent_child_framenet(str(relation))
      parent, child = vocab2.stoi[parent], vocab2.stoi[child]
      if child not in mp:
        mp[child] = []

      # Remove the __NO_FRAME__ <unk> <pad>
      if parent > 2:
        mp[child].append(parent)

  cnt = 0
  for i in range(frame_max):
    frame = i
    mp[frame] = list(set(mp[frame]))
    if len(mp[frame]) > 0:
      cnt += 1

  print("Present parent frame:", cnt, "for frames:", frame_max)
  return mp

frame_parents = {}

def get_frames_for_upper_layers(args, f_val, vocab2, obsv_prob=0.0):
  global frame_parents
  if len(frame_parents.keys()) == 0:
    frame_parents = get_parent_child_mapping(args.total_frames, vocab2)

  f_val = f_val.tolist()
  mx = 0
  for i in range(len(f_val)):
    # change it to [] for just parents
    ret = f_val[i]
    for v in f_val[i]:
      ret = ret + frame_parents[v]
    
    f_val[i] = ret
    mx = max(mx, len(f_val[i]))

  for i in range(len(f_val)):
    # TODO: as used padding, so use use_packed for upper layers also
    f_val[i] = f_val[i] + [vocab2.stoi['__NO_FRAME__']] * (mx - len(f_val[i]))

  return torch.tensor(f_val).cuda()
  
  # partially observed
  probs = obsv_prob * torch.ones_like(f_val, dtype=torch.float)
  selector = torch.bernoulli(probs)
  selector[selector < 1] = 0
  selector[selector == 1] = 1
  selector = selector.type(torch.LongTensor)
  selector = selector.cuda()
  obs_frames_select = f_val * selector
  return obs_frames_select