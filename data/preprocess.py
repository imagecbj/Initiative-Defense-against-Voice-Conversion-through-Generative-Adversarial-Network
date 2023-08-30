import json
from pathlib import Path

import torch
from tqdm import tqdm

from data.utils import get_melgan_mel, load_json, gender_info
from hyperparameter import hp


def vctk_processing_wav(
        data_dir, mel_dir
):
    data_dir = Path(data_dir)
    mel_dir = Path(mel_dir)
    mel_dir.mkdir(exist_ok=True)

    spk_dirs = data_dir.iterdir()
    for spk in tqdm(spk_dirs):
        if spk.name == ".DS_Store":
            continue

        wav_files = list(spk.iterdir())
        (mel_dir / spk.name).mkdir(exist_ok=True)
        for wav in wav_files:
            print(f'wav: {wav.stem}')
            mel = get_melgan_mel(wav)
            torch.save(mel.cpu(), mel_dir / spk.name / f"{wav.stem}.pt")
    print("Done")


def metadata(mel_dir):
    mel_dir = Path(mel_dir)
    gender_dict = gender_info()

    meta_data = dict()
    meta_data['train'] = []

    for spk in mel_dir.iterdir():
        if spk.is_dir():
            for mel in spk.iterdir():
                if spk.name in gender_dict["female"]:
                    gender = 0
                else:
                    gender = 1
                meta_data['train'].append((str(mel), gender))

    with open(hp.metadata, "w") as j:
        json.dump(meta_data, j)
