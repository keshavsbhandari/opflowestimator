import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionFlow(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim = 8, activation='leaky_relu'):
        super(SelfAttentionFlow, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.predictflow = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        print(out.shape)

        occact = torch.bmm(out.view(m_batchsize,C,-1),attention).view(m_batchsize,C,width,height)
        _, occlusion = F.softmax(occact, 1).max(1)
        return self.predictflow(out), 1 - occlusion.unsqueeze(1)