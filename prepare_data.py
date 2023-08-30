from data import metadata, vctk_processing_wav
from data.utils import vctk_48_2_22
from hyperparameter import hp


def main():
    vctk_48_2_22(vctk_48k_path=hp.vctk_48k, vctk_22k_path=hp.vctk_22k)
    vctk_processing_wav(hp.vctk_22k, hp.vctk_mels)
    metadata(hp.vctk_mels)


if __name__ == "__main__":
    main()
