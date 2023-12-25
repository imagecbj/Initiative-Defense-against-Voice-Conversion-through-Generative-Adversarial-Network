import os
import torch.cuda


class HyParams:
    def __init__(self):
        self.project_root = "/home/lab/workspace/works/chanchan/Initiative-Defense-against-Voice-Conversion-through-Generative-Adversarial-Network"
        self.sample_rate = 22050
        self.ref_db = 20.0
        self.dc_db = 100.0
        self.seed = 1240
        self.seg_len = 128
        self.pm = 0.075
        self.epochs = 5
        self.bs = 128
        self.lr = 0.001
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.lambda_adv_l2 = 1
        self.lambda_quality = 10

        self.vctk_speaker_info = self.__path_join(
            self.project_root, "data/VCTK-Corpus-0.92/speaker-info.txt"
        )
        self.vctk_txt = self.__path_join(self.project_root, "data/VCTK-Corpus-0.92/txt")
        self.vctk_spk_txt = self.__path_join(self.project_root, "data/spk_txt.json")
        self.metadata = self.__path_join(self.project_root, "data/metadata.json")
        self.gender_info = self.__path_join(self.project_root, "data/gender_info.json")
        self.vqvcp_model = self.__path_join(self.project_root, "target/vqvcp/gen")
        self.saved_pt = self.__path_join(self.project_root, "checkpoints/netG")
        self.temp_folder = self.__path_join(self.project_root, "temp")
        self.vctk_mels = self.__path_join(self.project_root, "data/mels")
        self.vctk_48k = self.__path_join(
            self.project_root, "data/VCTK-Corpus-0.92/wav48_silence_trimmed"
        )
        self.vctk_22k = self.__path_join(self.project_root, "data/vctk_22k")
        self.vctk_rec_mels = self.__path_join(self.project_root, "data/rec_mels")
        self.vctk_eer = self.__path_join(
            self.project_root,
            "metrics/speaker_verification/equal_error_rate/VCTK_eer.yaml",
        )
        self.d_net = self.__path_join(self.project_root, "checkpoints/swcsm/d_net.pt")
        self.test_set = [
            "p236",
            "p237",
            "p240",
            "p262",
            "p268",
            "p274",
            "p280",
            "p286",
            "p299",
            "p301",
            "p302",
        ]
        self.valid_set = [
            "p246",
            "s5",
            "p362",
            "p253",
            "p263",
            "p267",
            "p248",
            "p306",
            "p276",
            "p252",
            "p264",
        ]

    def __path_join(self, project_root, relative_path):
        root = os.path.join(project_root, relative_path)
        return root


hp = HyParams()

if __name__ == "__main__":
    pass
