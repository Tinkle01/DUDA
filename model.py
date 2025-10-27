import torch
from torch import nn
import torch.nn.functional as F
import pdb


class MSNet(nn.Module):
    def __init__(self, input_channel=3):
        super(MSNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(input_channel),
            nn.Conv2d(input_channel, 32, 5, padding=2),
            nn.SELU(),
        )
        self.pool1 = nn.MaxPool2d((4, 1), return_indices=True)

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(32), nn.Conv2d(32, 64, 5, padding=2), nn.SELU()
        )
        self.pool2 = nn.MaxPool2d((4, 1), return_indices=True)

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(64), nn.Conv2d(64, 128, 5, padding=2), nn.SELU()
        )
        self.pool3 = nn.MaxPool2d((4, 1), return_indices=True)
        self.bottom = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1, 5, padding=(0, 2)),
            nn.SELU(),
        )

        self.up_pool3 = nn.MaxUnpool2d((4, 1))
        self.up_conv3 = nn.Sequential(
            nn.BatchNorm2d(128), nn.Conv2d(128, 64, 5, padding=2), nn.SELU()
        )
        self.up_pool2 = nn.MaxUnpool2d((4, 1))
        self.up_conv2 = nn.Sequential(
            nn.BatchNorm2d(64), nn.Conv2d(64, 32, 5, padding=2), nn.SELU()
        )

        self.up_pool1 = nn.MaxUnpool2d((4, 1))
        self.up_conv1 = nn.Sequential(
            nn.BatchNorm2d(32), nn.Conv2d(32, 1, 5, padding=2), nn.SELU()
        )

        self.final_linear = nn.Sequential(
            nn.Conv1d(
                7+13+321,
                320, 5, padding=2),
            nn.SELU()
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, octave=None, tone=None, return_feat=False):
        x = x.to(next(self.parameters()).device)
        c1, ind1 = self.pool1(self.conv1(x))
        c2, ind2 = self.pool2(self.conv2(c1))
        c3, ind3 = self.pool3(self.conv3(c2))
        bm = self.bottom(c3)
        u3 = self.up_conv3(self.up_pool3(c3, ind3))
        u2 = self.up_conv2(self.up_pool2(u3, ind2))
        u1 = self.up_conv1(self.up_pool1(u2, ind1))
        feat = torch.cat((bm, u1), dim=2).squeeze(dim=1)

        if octave is not None:
            feat = torch.cat((feat, octave, tone), dim=1)
            feat = self.final_linear(feat)
            feat = torch.cat((bm.squeeze(dim=1), feat), dim=1)
        output = self.softmax(feat)
        if return_feat:
            return output, feat
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))


if __name__ == "__main__":
    x = torch.randn(1, 256, 64, 64)
    net = MSNet()
    print(net(x).shape)
