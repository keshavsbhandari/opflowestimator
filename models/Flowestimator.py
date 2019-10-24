import torch.nn as nn
import torch
from models.DenseNet import DenseNet
from models.Unet import UNet
from models.Occlusion import Occlusion
from models.TwoLayerArch import Twolayer
from utils.masktransform import WDTransformer
from utils.censustransform import censustransform


class FlowEstimator(nn.Module):
    def __init__(self, **wdtargs):
        """
        :param wdtargs: {shape_in = (256,256), use_l2 = True, channel_in = 3, stride = 1, kernel_size = 2, use_cst = True}
        :type wdtargs:
        """
        super(FlowEstimator, self).__init__()
        self.unet = UNet(8, 8)
        self.predictflow = nn.Conv2d(kernel_size=3, stride=1, in_channels=8, padding=1, out_channels=2)
        self.occlusion = Occlusion()

        """Another Options"""
        self.init_wdt(wdtargs)

    def init_wdt(self, wdtargs):
        self.wdt = WDTransformer(**wdtargs).cuda()
    def forward(self, frame1, frame2):
            x = self.wdt(frame1, frame2)
            frame1 = censustransform(frame1)
            frame2 = censustransform(frame2)
            x = torch.cat([x,frame1,frame2],1)
            flow = self.predictflow(self.unet(x))
            occ = 1 - self.occlusion(torch.sigmoid(flow))
            return flow, occ
