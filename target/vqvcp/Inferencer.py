from pathlib import Path

import soundfile as sf
import torch.nn as nn

from data.utils import melgan, get_melgan_mel
from hyperparameter import hp
from .vqvcp import load_model


class VQVCPInferencer(object):

    def __init__(self):
        self.name = "vqvcp"
        self.vocoder = melgan
        self.model = load_model(hp.vqvcp_model)
        self.L2 = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.cos = nn.CosineSimilarity()

    def generate_wavs(self, mels):
        return self.vocoder.inverse(mels)

    def adv_loss(self, adv, x, src):
        clean_rec_mel = self.inference_from_mel(tgt_mel=x, src_mel=src)
        adv_rec_mel = self.inference_from_mel(tgt_mel=adv, src_mel=src)
        l2_loss = self.L2(adv_rec_mel, clean_rec_mel)

        return l2_loss

    def inference_from_mel(self, tgt_mel, src_mel):
        src_mel = src_mel[:, :, :src_mel.size(2) // 16 * 16]
        tgt_mel = tgt_mel[:, :, :tgt_mel.size(2) // 16 * 16]

        q_after_block, sp_embedding_block, std_block, _ = self.model.encode(src_mel)
        q_after_block_tg, sp_embedding_block_tg, std_block_tg, _ = self.model.encode(tgt_mel)

        dec = self.model.decode(q_after_block, sp_embedding_block_tg, std_block_tg)

        cvt_mel = dec

        return cvt_mel

    def inference_from_path(self, tgt_path, src_path, save_folder):
        src_path = Path(src_path)
        tgt_path = Path(tgt_path)
        output_path = Path(save_folder) / f"cvt_{self.name}_{tgt_path.stem}_to_{src_path.stem}.wav"

        melgan_src_mel = get_melgan_mel(src_path, is_squeeze=False, is_transposed=False).to(hp.device)
        melgan_tgt_mel = get_melgan_mel(tgt_path, is_squeeze=False, is_transposed=False).to(hp.device)
        cvt_mel = self.inference_from_mel(tgt_mel=melgan_tgt_mel, src_mel=melgan_src_mel)

        wav = self.generate_wavs(cvt_mel)[0].data.cpu().numpy()
        sf.write(output_path, wav, 22050)
        return output_path
