import torch.nn as nn
import torch
from models.DenseNet import DenseNet
from models.PyramidalUNet import UNet
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
        self.pyramid_occlusion1 = Occlusion()
        self.pyramid_occlusion2 = Occlusion()
        self.pyramid_occlusion3 = Occlusion()
        self.pyramid_occlusion4 = Occlusion()
        self.pyramid_occlusion5 = Occlusion()
        self.pyramid_occlusion6 = Occlusion()

        """Another Options"""
        self.init_wdt(wdtargs)

    def init_wdt(self, wdtargs):
        self.wdt = WDTransformer(**wdtargs).cuda()

    def forward(self, frame1, frame2):
        x = self.wdt(frame1, frame2)
        # frame1 = censustransform(frame1)
        # frame2 = censustransform(frame2)
        x = torch.cat([x, frame1, frame2], 1)
        # x = torch.cat([frame1, frame2], 1)
        if self.training:
            flow1, flow2, flow3, flow4, flow5, flow6, out = self.unet(x)
            occ1 = self.occlusion1(torch.sigmoid(flow1))
            occ2 = self.occlusion1(torch.sigmoid(flow2))
            occ3 = self.occlusion1(torch.sigmoid(flow3))
            occ4 = self.occlusion1(torch.sigmoid(flow4))
            occ5 = self.occlusion1(torch.sigmoid(flow5))
            occ6 = self.occlusion1(torch.sigmoid(flow6))

            flow = self.predictflow(out)

            occ = self.occlusion(torch.sigmoid(flow))

            return flow1, flow2, flow3, flow4, flow5, flow6, flow, occ1, occ2, occ3, occ4, occ5, occ6, occ
        else:
            out = self.unet(x)
            flow = self.predictflow(out)
            return flow, out
