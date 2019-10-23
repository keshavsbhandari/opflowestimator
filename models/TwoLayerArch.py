import torch.nn as nn
import torch.nn.functional as F
import torch
class Twolayer(nn.Module):
    def __init__(self):
        super(Twolayer, self).__init__()
        self.conv1 = nn.Conv2d(kernel_size=3,in_channels=2,out_channels=8, stride=1, padding=1)
        self.conv2 = nn.Conv2d(kernel_size=3,in_channels=8,out_channels=28, stride=1, padding=1)
    def forward(self,x):
        return F.leaky_relu(self.conv2(F.leaky_relu(self.conv1(x))))