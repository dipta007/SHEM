import nltk
from pprint import pprint
nltk.download('framenet_v17', download_dir='./')
nltk.data.path.append('./')
from nltk.corpus import framenet as fn
import torch


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
      relations = list(fn.frame_relations(frame, type="Inheritance"))
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

def get_parent_child_mapping_for_scenerio(frame_max, vocab2):
  from collections import defaultdict

  scenarios = fn.frames(r'(?i)_scenario')
  scenario_dict = {s.name: s for s in scenarios}
  frame_to_scenario = defaultdict(set)

  # print(scenario_dict)
  for scenario_name, scenario_obj in scenario_dict.items():
    for sub in scenario_obj.frameRelations:
      frame_to_scenario[sub['subFrame'].name].add(scenario_obj.name)

  mp = {i: [] for i in range(frame_max)}  
  for key, val in frame_to_scenario.items():
    p_no = vocab2.stoi[key]
    if p_no not in mp:
      mp[p_no] = []
    
    for v in val:
      mp[p_no].append(vocab2.stoi[v])
  
  cnt = 0
  for i in range(frame_max):
    frame = i
    mp[frame] = list(set(mp[frame]))
    if len(mp[frame]) > 0:
      cnt += 1

  print("Present parent frame:", cnt, "for frames:", frame_max)

  return mp

def get_frames_for_upper_layers(args, f_val, vocab2, obsv_prob=0.0):
  global frame_parents
  if len(frame_parents.keys()) == 0:
    frame_parents = get_parent_child_mapping_for_scenerio(3020, vocab2) # the max id in framenet 3020

  f_val = f_val.tolist()
  mx = 0
  for i in range(len(f_val)):
    # change it to [] for just parents
    ret = []
    for v in f_val[i]:
      if len(frame_parents[v]) > 0:
        ret = ret + [frame_parents[v][0]]
      else:
        ret = ret + [v]
      # ret = ret + frame_parents[v]
    
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


# print(fn.frames())

# s = set()
# for f in fn.frames():
#   print(f.ID, f.name)
#   s.add(f.ID)

# print(len(s), max(s))
# for i in range(10000):
#   if i not in s:
#     print(i)
#     break




# from collections import defaultdict

# scenarios = fn.frames(r'(?i)_scenario')
# scenario_dict = {s.name: s for s in scenarios}
# frame_to_scenario = defaultdict(set)

# # print(scenario_dict)
# for scenario_name, scenario_obj in scenario_dict.items():
#   for sub in scenario_obj.frameRelations:
#     frame_to_scenario[sub['subFrame'].name].add(scenario_obj.name)

# with open('./data/train_0.6_frame.txt', 'r') as fp:
#   tot, cnt = 0, 0
#   for line in fp:
#     flg = False
#     for v in line.split()[:]:
#       if len(frame_to_scenario[v]) > 0:
#         flg = True
#         break
#     if flg:
#       cnt += 1
#     tot += 1
  
#   print(cnt, tot)
#   print("Coverage: ", cnt * 100.0 / tot)