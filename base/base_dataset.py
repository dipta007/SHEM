import logging

import hydra
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(self, paths, fields=["x", "y"]):
        super().__init__()

        self.paths = paths
        self.fields = fields

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        """
        batch = [
            [(text1, text1_len), (text2, text2_len), ..., target],
            [(text1, text1_len), (text2, text2_len), ..., target],
        ]
        last one must be the target, if there is no pass -1 as target (i.e. unsupervised)
        """
        # ? sorting
        # batch.sort(key=lambda x: x[0][1], reverse=True)

        # #? zipping to collate inside batch
        # values = list(zip(*batch))
        # x = (tuple(zip(*x)) for x in values[:-1])
        # y = values[-1]

        # #? padding and make tensor
        # x = [self.tokenizer(x[0]), torch.tensor(x[1]) for x in x]
        # y = torch.tensor(y)

        # #? make dictionary
        # ret = {x: y for x, y in zip(self.fields, x)}
        # ret['target'] = y

        return batch


def get_data(config, target, dataset_args, dataloader_args):
    dataset = hydra.utils.instantiate(target)(config=config, **dataset_args)
    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, **dataloader_args)
    log.info(f"Data Loaded, length: {len(dataset)}, batch length: {len(dataloader)}")
    return dataset, dataloader
