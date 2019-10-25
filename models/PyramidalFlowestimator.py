import torch.nn as nn
import torch
from models.DenseNet import DenseNet
from models.PyramidalUNet import UNet
# from models.Occlusion import Occlusion
from models.Attention import SelfAttentionFlow
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

        self.predictflow_and_occlusion = SelfAttentionFlow()


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
            # flow1, flow2, flow3, flow4, flow5, flow6, out = self.unet(x)
            # flow4, flow5, flow6, out = self.unet(x)
            out = self.unet(x)
            # flow1, occ1 = self.predictflow(flow1)
            # flow2, occ2 = self.predictflow(flow2)
            # flow3, occ3 = self.predictflow(flow3)
            # flow4, occ4 = self.predictflow_and_occlusion(flow4)
            # flow4 = self.predictflow(flow4)
            # flow5, occ5 = self.predictflow(flow5)
            # flow6, occ6 = self.predictflow(flow6)
            flow, occ = self.predictflow_and_occlusion(out)
            # flow = self.predictflow(flow)
            return flow, occ
            # return flow4, flow5, flow6, flow, occ4, occ5, occ6, occ
            # return flow1, flow2, flow3, flow4, flow5, flow6, flow, occ1, occ2, occ3, occ4, occ5, occ6, occ

        else:
            out = self.unet(x)
            flow, occ = self.predictflow_and_occlusion(out)
            # flow = self.predictflow(flow)
            return flow, occ
