from transformers import AutoTokenizer

from base.base_dataset import BaseDataset

# Reserved Special Tokens
PAD_TOK = "<pad>"
SOS_TOK = "<s>"  # start of sentence
EOS_TOK = "</s>"  # end of sentence
UNK_TOK = "<unk>"
TUP_TOK = " . "
DIST_TOK = "<DIST>"  # distractor token for NC task
NOFRAME_TOK = "__NOFRAME__"

# These are the values that should be used during evalution to keep things consistent
MIN_EVAL_SEQ_LEN = 8
MAX_EVAL_SEQ_LEN = 50


class InverseNarrativeDatasetForBart(BaseDataset):
    def __init__(self, config, paths, vocab, min_src_seq_length, max_src_seq_length):
        super().__init__(paths)
        self.config = config
        self.vocab = vocab
        self.min_src_seq_length = min_src_seq_length
        self.max_src_seq_length = max_src_seq_length

        self.data = self.get_data()

    def get_data(self):
        data = []
        with open(self.paths[0], "r") as f:
            for idx, line in enumerate(f):
                text = line.strip()

                sents = text.split(DIST_TOK)
                assert len(sents) == 6, "Original + 5 distractions"

                sents = [s.replace(".", "") for s in sents]
                sents = [s.replace("<TUP>", ".") for s in sents]
                sents = [s.strip() for s in sents]

                actual = sents[0].strip()
                dists = sents[1:]
                seed = actual.split(TUP_TOK)[0].strip()

                words_in_actual = len(actual.split())
                if (
                    words_in_actual < self.min_src_seq_length
                    or words_in_actual > self.max_src_seq_length
                ):
                    continue

                def tokenize(x):
                    enc = self.vocab(
                        x,
                        padding="max_length",
                        truncation=True,
                        max_length=self.config.data.seq_len,
                        return_tensors="pt",
                    )
                    l = enc.attention_mask.sum()
                    enc = {k: v.squeeze(0) for k, v in enc.items()}
                    return (enc, l)

                curr_data = [seed, actual]
                for i, dist in enumerate(dists):
                    dist = f"{seed} . {dist}"
                    assert (
                        len(dist.split(".")) == 6
                    ), f"All sentences must have 6 events. {i}"

                    curr_data.append(dist)

                curr_data = [tokenize(s) for s in curr_data]

                data.append(curr_data)

        return data
