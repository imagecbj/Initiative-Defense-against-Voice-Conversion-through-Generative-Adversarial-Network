import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from data import make_training_data_loader
from data.utils import melgan, normalizer, denormalizer
from hyperparameter import hp
from model import load_gen_and_dis
from target import vc_model
from model import SWCSM


class Trainer:
    def __init__(self):
        self.netG, self.netD = load_gen_and_dis()
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), hp.lr)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), hp.lr)
        self.data_loader = make_training_data_loader()
        self.vc_model = vc_model()
        self.vocoder = melgan
        self.mse = nn.MSELoss()
        self.distort = self.load_SWCSM()

    def seed_everything(self, seed=42):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_SWCSM(self):
        d = SWCSM().to(hp.device)
        d.load_state_dict(torch.load('d_net.pt'))
        d.eval()
        return d

    def save_model(self, epoch):
        pt_path = Path(hp.saved_pt)
        pt_path.mkdir(exist_ok=True, parents=True)
        torch.save(
            {'netG': self.netG.state_dict()}, pt_path / f"epoch_{epoch}_netG_{hp.pm}"
        )
        torch.save({'netD': self.netD.state_dict()}, pt_path / f"epoch_{epoch}_netD_{hp.pm}")
        torch.save(
            {
                'G': self.optimizer_G.state_dict(),
                'D': self.optimizer_D.state_dict()
            }, pt_path / f"epoch_{epoch}_opt_{hp.pm}"
        )
        print(f"========SAVING PTS IN {pt_path}=========")
        return pt_path / f"epoch_{epoch}_netG_{hp.pm}"

    def train_batch(self, x):
        self.netG.train()
        self.netD.train()

        real_label_factor = round(random.uniform(0.7, 1.2), 2)
        fake_label_factor = round(random.uniform(0.0, 0.3), 2)

        # optimizeation of Ds
        for i in range(1):
            x = normalizer(x)
            per = self.netG(x)
            adv = per * hp.pm + x

            self.optimizer_D.zero_grad()
            pred_real = self.netD(x)

            loss_D_real = F.mse_loss(
                pred_real,
                real_label_factor
                * torch.ones_like(pred_real, device=hp.device),
            )
            pred_fake = self.netD(adv.detach())

            loss_D_fake = F.mse_loss(
                pred_fake,
                fake_label_factor
                + torch.zeros_like(pred_fake, device=hp.device),
            )

            loss_D_GAN = (
                    loss_D_real
                    + loss_D_fake
            )

            loss_D_GAN.backward()
            self.optimizer_D.step()

        # optimization of G

        for i in range(1):
            self.optimizer_G.zero_grad()

            # Gan loss
            pred_fake = self.netD(adv)
            loss_G_GAN = F.mse_loss(
                pred_fake,
                real_label_factor
                * torch.ones_like(pred_fake, device=hp.device),
            )
            adv = self.distort(adv)
            loss_quality = self.mse(adv, x)

            adv = denormalizer(adv)
            x = denormalizer(x)
            src = x.clone().detach()

            adv_l2_loss = -self.vc_model.adv_loss(adv, x, src)

            loss_G = loss_G_GAN + hp.lambda_adv_l2 * adv_l2_loss + hp.lambda_quality * loss_quality

            loss_G.backward()
            self.optimizer_G.step()

        print(
            f"loss_D_GAN: {loss_D_GAN} | loss_G_GAN: {loss_G_GAN} | adv_l2: {adv_l2_loss}  | mse: {loss_quality}\r")

    def train(
            self,
    ):
        self.seed_everything(hp.seed)
        for epoch in range(hp.epochs):
            print(f"training on epoch {epoch}")
            for i, data in tqdm(enumerate(self.data_loader, start=0)):
                mel, gender = data
                mel = mel.to(hp.device)
                self.train_batch(mel)

            pt_path = self.save_model(epoch)
