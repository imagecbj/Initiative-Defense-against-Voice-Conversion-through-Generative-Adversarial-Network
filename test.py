import yaml

from pathlib import Path
from tqdm import tqdm
from resemblyzer import VoiceEncoder, preprocess_wav
from metrics import evaluate_sim


def evalation(root, alg_tags):
    vcs = ["adainvc", "vqvcp", "againvc", "triaan"]
    with open("test_result.txt", "a") as f:
        f.write("============\n")
        for tag in alg_tags:
            f.write(f"{tag}\n")
            for sub in vcs:
                for target in vcs:
                    f.write(
                        f"sub: {sub} | target: {target} | {evaluate_sim(root, f'cvt_{target}_{tag}_0075_{sub.name}_*')}\n"
                    )
                f.write(f"quality: {evaluate_sim(root, f'{tag}_0075_{sub}_*')}\n")
            f.write("\n")
        f.write("============\n")


if __name__ == "__main__":
    for tag in ["ae", "aew", "huang", "pgd", "fgsm"]:
        evalation(root=f"audios/{tag}", alg_tags=[tag])
