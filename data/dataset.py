import json
import random

import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader

from hyperparameter import hp


class VctkDataset(Dataset):
    def __init__(self):
        self.hp = hp
        with open(self.hp.metadata, "r") as j:
            self.metadata = json.load(j)['train']

    def __len__(self, ):
        return len(self.metadata)

    def __getitem__(self, idx):
        mel_path, gender = self.metadata[idx]

        mel = torch.load(mel_path)

        if len(mel) < self.hp.seg_len:
            len_pad = self.hp.seg_len - len(mel)
            mel = pad(mel, (0, 0, 0, len_pad), 'constant', 0)
        else:
            start = random.randint(0, max(len(mel) - self.hp.seg_len, 0))
            mel = mel[start: (start + self.hp.seg_len)]

        mel = mel.T

        return mel, gender


def make_training_data_loader():
    data = VctkDataset()
    data_loader = DataLoader(
        dataset=data,
        batch_size=hp.bs,
        shuffle=True,
        drop_last=True
    )

    return data_loader
