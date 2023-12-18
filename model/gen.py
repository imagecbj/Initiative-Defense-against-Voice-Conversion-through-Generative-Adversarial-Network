from .opreations import *


class Generator(nn.Module):
    def __init__(self, in_channels=80, channels=512):
        super().__init__()
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = Conv1dReluBn(512, 256, kernel_size=5, padding=2)
        self.layer3 = Conv1dReluBn(256, 128, kernel_size=5, padding=2)
        self.layer4 = Conv1dReluBn(128, 64, kernel_size=5, padding=2)
        self.layer5 = Conv1dReluBn(64, 32, kernel_size=5, padding=2)
        self.layer6 = Conv1dReluBn(32, 1, kernel_size=5, padding=2)

        self.conv = Conv1dReluBn(1, 1, kernel_size=5, padding=2)

        self.layer7 = ConvTranspose1dReluBn(1, 32, kernel_size=5, padding=2)
        self.layer8 = ConvTranspose1dReluBn(32, 64, kernel_size=5, padding=2)
        self.layer9 = ConvTranspose1dReluBn(64, 128, kernel_size=5, padding=2)
        self.layer10 = ConvTranspose1dReluBn(128, 256, kernel_size=5, padding=2)
        self.layer11 = ConvTranspose1dReluBn(256, 512, kernel_size=5, padding=2)
        self.layer12 = ConvTranspose1dReluBn(
            channels, in_channels, kernel_size=5, padding=2
        )

        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.conv(out)

        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.tanh(out)
        return out
