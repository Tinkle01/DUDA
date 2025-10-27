import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Softmax



class MSNet(nn.Module):
    def __init__(self):
        super(MSNet, self).__init__()
        self.oct_num = 6
        self.tone_num = 12
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, 5, padding=2),
            nn.SELU()
        )
        self.pool1 = nn.MaxPool2d((4, 1), return_indices=True)

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.SELU()
        )
        self.pool2 = nn.MaxPool2d((4, 1), return_indices=True)

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.SELU()
        )
        self.pool3 = nn.MaxPool2d((4, 1), return_indices=True)
        self.bottom = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1, 5, padding=(0, 2)),
            nn.SELU()
        )

        self.up_pool3 = nn.MaxUnpool2d((4, 1))
        self.up_conv3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 5, padding=2),
            nn.SELU()
        )
        self.up_pool2 = nn.MaxUnpool2d((4, 1))
        self.up_conv2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.SELU()
        )

        self.up_pool1 = nn.MaxUnpool2d((4, 1))
        self.up_conv1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 5, padding=2),
            nn.SELU()
        )

        self.octave_linear2 = nn.Sequential(
            nn.Linear(320, 256),
            nn.Dropout(p=0.2),
            nn.SELU(),
            nn.Linear(256, 64),
            nn.Dropout(p=0.2),
            nn.SELU(),
            nn.Linear(64, self.oct_num),
            nn.Dropout(p=0.2),
            nn.SELU()
        )
        self.tone_linear2 = nn.Sequential(
            nn.Linear(320, 512),
            nn.Dropout(p=0.2),
            nn.SELU(),
            nn.Linear(512, 128),
            nn.Dropout(p=0.2),
            nn.SELU(),
            nn.Linear(128, self.tone_num),
            nn.Dropout(p=0.2),
            nn.SELU()
        )

        self.tone_bm = nn.Sequential(
            nn.Linear(1, 1),
            nn.SELU()
        )

        self.octave_bm = nn.Sequential(
            nn.Linear(1, 1),
            nn.SELU()
        )

        self.softmax = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)


    def forward(self, x):

        c1, ind1 = self.pool1(self.conv1(x))
        #print('1222', c1.shape)
        c2, ind2 = self.pool2(self.conv2(c1))
        c3, ind3 = self.pool3(self.conv3(c2))
        bm = self.bottom(c3)
        u3 = self.up_conv3(self.up_pool3(c3, ind3))
        u2 = self.up_conv2(self.up_pool2(u3, ind2))
        u1 = self.up_conv1(self.up_pool1(u2, ind1))
        feat = torch.cat((bm, u1), dim=2)
        output = self.softmax(torch.cat((bm, u1), dim=2))
        # ********  octave-tone  **********
        octave = self.octave_linear2(u1.squeeze(dim=1).permute(0, 2, 1)).permute(0,2,1)
        bm = bm.squeeze(dim=1).permute(0,2,1) #(bs, 64, 1)
        bm_oct = self.octave_bm(bm).permute(0,2,1)
        tone = self.tone_linear2(u1.squeeze(dim=1).permute(0, 2, 1)).permute(0,2,1)
        bm_tone = self.tone_bm(bm).permute(0, 2, 1)

        feat_octave = torch.cat((bm_oct, octave), dim=1)
        feat_tone = torch.cat((bm_tone, tone), dim=1)

        octave = self.softmax2(feat_octave)
        tone = self.softmax2(feat_tone)

        return output.squeeze(dim=1), octave, tone,  feat_octave, feat_tone

if __name__ == '__main__':
    x = torch.randn(1, 3, 320, 64)
    Net = MSNet()
