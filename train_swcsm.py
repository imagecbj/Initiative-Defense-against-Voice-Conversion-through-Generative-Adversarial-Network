import json
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import SWCSM
from data.utils import melgan, audio2mel, normalizer, denormalizer

import warnings
from hyperparameter import hp

warnings.filterwarnings("ignore")


def prepare_rec_mels():
    mels = Path(hp.vctk_mels)
    rec_mels = Path(hp.vctk_rec_mels)
    for spk in tqdm(mels.iterdir()):
        (rec_mels / spk.name).mkdir(exist_ok=True, parents=True)
        for mel_path in spk.iterdir():
            rec_mel_path = rec_mels / spk.name / mel_path.name
            mel = torch.load(mel_path).to(hp.device)
            wav = melgan.inverse(mel.unsqueeze(dim=0).transpose(-1, -2))
            rec_mel = audio2mel(wav.unsqueeze(dim=0))
            rec_mel = rec_mel.squeeze(dim=0).transpose(-1, -2)
            torch.save(rec_mel.cpu(), rec_mel_path)


class DistortionDataset(Dataset):
    def __init__(self, metadata_tag="train"):
        self.hp = hp
        with open(self.hp.metadata, "r") as j:
            self.metadata = json.load(j)[metadata_tag]

    def __len__(
        self,
    ):
        return len(self.metadata)

    def __getitem__(self, idx):
        mel_path, gender = self.metadata[idx]
        mel_path = Path(mel_path)
        spk = mel_path.parent.name

        rec_mel_path = mel_path.parents[2] / hp.vctk_rec_mels / spk / mel_path.name
        mel = torch.load(mel_path)
        rec_mel = torch.load(rec_mel_path)

        if len(mel) < self.hp.seg_len:
            len_pad = self.hp.seg_len - len(mel)
            mel = pad(mel, (0, 0, 0, len_pad), "constant", 0)
            rec_mel = pad(rec_mel, (0, 0, 0, len_pad), "constant", 0)
        else:
            start = random.randint(0, max(len(mel) - self.hp.seg_len, 0))
            mel = mel[start : (start + self.hp.seg_len)]
            rec_mel = rec_mel[start : (start + self.hp.seg_len)]

        mel = mel.T
        rec_mel = rec_mel.T

        return normalizer(mel), normalizer(rec_mel)


if __name__ == "__main__":
    prepare_rec_mels()
    dataset = DistortionDataset()
    validset = DistortionDataset("valid")
    testset = DistortionDataset("test")
    mse = nn.MSELoss()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    validloader = DataLoader(validset, batch_size=1024, shuffle=False, drop_last=True)
    testloader = DataLoader(testset, batch_size=1024, shuffle=False, drop_last=True)
    g = SWCSM().to(hp.device)
    optimizer = torch.optim.Adam(g.parameters(), lr=1e-4)
    valid_best = 0
    test_best = 0

    valid_loss_per_epochs = []
    test_loss_per_epochs = []
    for epoch in range(500):
        g.train()
        for i, (mel, rec_mel) in tqdm(enumerate(dataloader)):
            mel = mel.to(hp.device)
            rec_mel = rec_mel.to(hp.device)
            optimizer.zero_grad()
            distored_mel = g(mel)
            loss = mse(distored_mel, rec_mel)
            print(f"loss: {loss.item()}\r")
            loss.backward()
            optimizer.step()

        print(f'{"*" * 10} valid and test start {"*" * 10}')
        g.eval()
        with torch.no_grad():
            valid_losses = []
            for i, (mel, rec_mel) in tqdm(enumerate(validloader)):
                mel = mel.to(hp.device)
                rec_mel = rec_mel.to(hp.device)
                valid_losses.append(mse(g(mel), rec_mel))
            valid_loss_per_epochs.append(torch.tensor(valid_losses).mean())
            print(valid_loss_per_epochs[epoch])
            if valid_loss_per_epochs[epoch] < valid_loss_per_epochs[valid_best]:
                valid_best = epoch

            test_losses = []
            for i, (mel, rec_mel) in tqdm(enumerate(testloader)):
                mel = mel.to(hp.device)
                rec_mel = rec_mel.to(hp.device)
                test_losses.append(mse(g(mel), rec_mel))
            test_loss_per_epochs.append(torch.tensor(test_losses).mean())
            print(test_loss_per_epochs[epoch])
            if test_loss_per_epochs[epoch] < test_loss_per_epochs[test_best]:
                test_best = epoch

            print(
                f'{"*" * 10} done, :ValidBest:{valid_best} value: {valid_loss_per_epochs[valid_best]} | \
                TestBEST: {test_best} value: {test_loss_per_epochs[test_best]} {"*" * 10}'
            )

        torch.save(g.state_dict(), f"checkpoints/swcsm/epoch_{epoch}_d_net.pt")
