from .opreations import *


class Discriminator(nn.Module):
    def __init__(self, in_channels=80, channels=512):
        super().__init__()
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8
        )
        self.layer3 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8
        )
        self.layer4 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8
        )
        self.layer5 = Conv1dReluBn(512, 256, kernel_size=5, padding=2)
        self.layer6 = Conv1dReluBn(256, 128, kernel_size=5, padding=2)
        self.layer7 = Conv1dReluBn(128, 64, kernel_size=5, padding=2)
        self.layer8 = Conv1dReluBn(64, 32, kernel_size=5, padding=2)
        self.layer9 = Conv1dReluBn(32, 1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.sigmoid(out)
        return out
