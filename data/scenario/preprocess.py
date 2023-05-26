import gzip
from pprint import pprint
import glob
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split


TUP_TOK = " <TUP> "

# get all files
files = glob.glob('/p/work/ferraro/concrete-framenet-subframe/anyt-edoc/maxclauses20/*')


tot = 0
num_clauses = 5
all_sens = []
all_frames = []
for filename in tqdm(files):
    with gzip.open(filename, 'rb') as fp:
        for sentline in fp:
            tot += 1
            objs = json.loads(sentline)

            sens = []
            frames = []
            ords = []
            for obj in objs:
                sen = f"{obj['predicate']} {obj['arg0']} {obj['arg1']} {obj['mod']}"
                sen = sen.strip()
                sen = sen.lower()
                ord = obj['selection_order']

                if obj['frame'] != "__NOFRAME__":
                    ords.append(ord)
                    sens.append((sen, ord))
                    frames.append((obj['frame'], ord))

            ords.sort()
            ords = ords[:num_clauses]

            sens = [sen for sen, ord in sens if ord in ords]
            frames = [frame for frame, ord in frames if ord in ords]

            sens = TUP_TOK.join(sens)
            frames = " ".join(frames)
            all_sens.append(sens)
            all_frames.append(frames)

print(tot)                      # 111152
print(len(all_sens))            # 111152
print(len(all_frames))          # 111152

# split
train_sens, test_sens, train_frames, test_frames = train_test_split(all_sens, all_frames, test_size=0.1, random_state=42)
val_sens, test_sens, val_frames, test_frames = train_test_split(test_sens, test_frames, test_size=0.5, random_state=42)

print(len(train_sens))          # 100036
print(len(test_sens))           # 5558
print(len(val_sens))            # 5558

# save
with open('./train_0.9_TUP.txt', 'w') as fp:
    for sen in train_sens:
        fp.write(f"{sen}\n")
with open('./valid_0.9_TUP.txt', 'w') as fp:
    for sen in val_sens:
        fp.write(f"{sen}\n")
with open('./test_0.9_TUP.txt', 'w') as fp:
    for sen in test_sens:
        fp.write(f"{sen}\n")

with open('./train_0.9_frame.txt', 'w') as fp:
    for sen in train_frames:
        fp.write(f"{sen}\n")
with open('./valid_0.9_frame.txt', 'w') as fp:
    for sen in val_frames:
        fp.write(f"{sen}\n")
with open('./test_0.9_frame.txt', 'w') as fp:
    for sen in test_frames:
        fp.write(f"{sen}\n")