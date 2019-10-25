import torch.nn as nn
import torch
import torch.nn.functional as F


class Occlusion(nn.Module):
    def __init__(self):
        super(Occlusion, self).__init__()

        self.convbranch1 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1)
        self.convbranch2 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1)
        self.convbranch3 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1)
        self.convmerge = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1)


    def forward(self,x):
        x1 = torch.sigmoid(self.convbranch1(x)).transpose(-2,-1)
        x2 = torch.sigmoid(self.convbranch2(x))
        x3 = torch.sigmoid(self.convbranch2(x))
        x12 = torch.matmul(x2,x1)
        x13 = torch.matmul(x12,x3)
        occlusion = F.sigmoid(self.convmerge(x13))
        return occlusion

