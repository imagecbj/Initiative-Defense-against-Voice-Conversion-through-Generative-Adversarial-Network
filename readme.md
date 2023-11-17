# Initiative Defense against Voice Conversion through Generative Adversarial Network

This is the official implementation of the paper Initiative Defense against Voice Conversion through Generative
Adversarial Network.

## The intention of our work
![vc_model drawio](https://github.com/predawnang/Initiative-Defense-against-Voice-Conversion-through-Generative-Adversarial-Network/assets/37857978/38a0e5f3-d86b-440b-8fe8-0f6073ee3983)
The left part of the figure represents the normal voice conversion process, while the right part illustrates the intention of our work. We introduce perturbations to the mel spectrogram of the target audio to prevent the voice conversion model from generating the intended output.

## VC Models

1. Adain-vc: https://github.com/jjery2243542/adaptive_voice_conversion?utm_source=catalyzex.com
2. Vqvc+: https://github.com/ericwudayi/SkipVQVC
3. Again-vc: https://github.com/KimythAnly/AGAIN-VC
4. Triaan-vc: https://github.com/winddori2002/TriAAN-VC

## vocoder && data preprocessing

Melgan: https://github.com/descriptinc/melgan-neurips

## Speaker verification

Resemblyzer: https://github.com/resemble-ai/Resemblyzer

## Dataset

VCTK: https://datashare.ed.ac.uk/handle/10283/2950

## Training

### Hyper-parameter

hyper-parameter settings are all in the file hyperparameter.py. You can adjust them according to the specific needs.

### Training

#### Setup hyperparameter.py

1. Set the feild sefl.project_root to the path of this repository.
2. Set the feild self.vctk_48k to the path of the VCTK dataset.
3. Set the feild self.vctk_22k to the path where you want to place resampled waveform.
4. Set the feild self.vctk_training_data to the path where you want to store the processed mel.
5. Set the feild self.vctk_speaker_info to the path of the VCTK speaker-info.txt file.
6. Set the feild self.metadata to the path where you place metadata.json.
7. Set the feild self.vctk_rec_mels to the path where you want to place vctk_rec_mels which are mels that reconstruct to
   the waveform and re-extract with melgan preprocessing.
8. Set the feild self.vqvcp_model to the path where you place the weights of the vqvcp.

#### Preprocessing

1. Run prepare_data.py to 1) downsample the VCTK dataset, 2) extract from the VCTK dataset, and 3) obtain the metadata.

#### Train SWCSM

1. Run train_swcsm.py to train the swcsm.

#### Train Framework

1. Run trainer.py to train the framework.

## Acknowledge

Our implementation is hugely influenced by the repositories metioned before and the following repositories as we benefit
a lot from their codes and papers.

- https://github.com/mathcbc/advGAN_pytorch
- https://github.com/cyhuang-tw/attack-vc
