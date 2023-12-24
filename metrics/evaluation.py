from pathlib import Path

import numpy as np
import yaml
from resemblyzer import VoiceEncoder, preprocess_wav
from tqdm import tqdm

from hyperparameter import hp

voiceEncoder = VoiceEncoder()


def compute_sim(target, vc):
    threshold_path = hp.vctk_eer
    info = yaml.safe_load(Path(threshold_path).open())
    emb_a = voiceEncoder.embed_utterance(preprocess_wav(target))
    emb_b = voiceEncoder.embed_utterance(preprocess_wav(vc))
    cosine_similarity = (
        np.inner(emb_a, emb_b) / np.linalg.norm(emb_a) / np.linalg.norm(emb_b)
    )
    return True if cosine_similarity > info["Threshold"] else False, cosine_similarity


def evaluate_sim(root, tag):
    n_accept = 0
    root = Path(root)
    t_ps = sorted(list(root.glob("*/t_*.wav")))
    total = len(t_ps)
    for t_p in tqdm(t_ps):
        compared = list(t_p.parent.glob(tag))[0]
        sim_record = compute_sim(t_p, compared)
        if sim_record[0]:
            n_accept += 1
    return n_accept / total
