import json
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from tqdm import tqdm
from pathlib import Path

from hyperparameter import hp


class Audio2Mel(nn.Module):
    def __init__(
            self,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            sampling_rate=22050,
            n_mel_channels=80,
            mel_fmin=0.0,
            mel_fmax=None,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec


audio2mel = Audio2Mel().to(hp.device)


def get_melgan_mel(wav_path, is_transposed=True, is_squeeze=True):
    speech, speech_sr = librosa.core.load(wav_path, sr=22050)

    speech_tensor = torch.from_numpy(speech)[None][None]
    speech_tensor = speech_tensor.to(hp.device)

    melgan_mel = audio2mel(speech_tensor)

    if is_squeeze:
        melgan_mel = melgan_mel.squeeze()
    if is_transposed:
        melgan_mel = melgan_mel.transpose(-1, -2)

    return melgan_mel


def load_json(file):
    with open(file, "r") as j:
        obj = json.load(j)
    return obj


def save_json(obj, file):
    with open(file, "w") as j:
        json.dump(obj, j)
    return file


melgan = torch.hub.load("descriptinc/melgan-neurips", "load_melgan")


def save_resampled_22k(wav_path, save_path):
    wav, sr = librosa.core.load(wav_path, sr=22050)
    wav, _ = librosa.effects.trim(wav, top_db=20)
    sf.write(save_path, wav, samplerate=sr)


def normalizer(mels):
    db_mels = 20 * mels
    norm_mels = torch.clamp((db_mels - hp.ref_db + hp.dc_db) / hp.dc_db, min=1e-8, max=1)
    return norm_mels


def denormalizer(norm_mels):
    db_mels = (torch.clamp(norm_mels, min=1e-8, max=1) * hp.dc_db) - hp.dc_db + hp.ref_db
    mels = db_mels / 20
    return mels


def vctk_48_2_22(vctk_48k_path, vctk_22k_path):
    vctk_48k_path = Path(vctk_48k_path)
    vctk_22k_path = Path(vctk_22k_path)
    vctk_22k_path.mkdir(exist_ok=True)

    spk_dirs = vctk_48k_path.iterdir()
    for spk in tqdm(spk_dirs):
        if spk.name == ".DS_Store":
            continue

        wav_files = list(spk.iterdir())
        (vctk_22k_path / spk.name).mkdir(exist_ok=True)
        for wav in wav_files:
            if wav.stem.split("_")[-1] == "mic1":
                stem_without_mic = "_".join(wav.stem.split("_")[:-1])
                speech, speech_sr = librosa.core.load(wav, sr=22050)
                speech, index = librosa.effects.trim(speech, top_db=20)
                soundfile.write(
                    file=vctk_22k_path / spk.name / f'{stem_without_mic}.wav',
                    data=speech,
                    samplerate=hp.sample_rate)
    print("Done")


def gender_info():
    male = []
    female = []
    gender = {}

    with open(hp.vctk_speaker_info, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                line_split = line.split()
                if line_split[2] == "M":
                    male.append(line_split[0])
                elif line_split[2] == "F":
                    female.append(line_split[0])
    gender["male"] = male
    gender["female"] = female
    return gender
