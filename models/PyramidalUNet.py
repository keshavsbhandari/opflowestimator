import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from models.Spectral import SpectralNorm


class UNet(nn.Module):

    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            SpectralNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels)),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(out_channels),
            SpectralNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels)),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def expansive_final(self, in_channels, mid_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            SpectralNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channels[0])),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(mid_channels[0]),
            SpectralNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels[0], out_channels=mid_channels[1])),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(mid_channels[1]),
            SpectralNorm(torch.nn.ConvTranspose2d(in_channels=mid_channels[1], out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1))
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            SpectralNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel)),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            SpectralNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel)),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            SpectralNorm(torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1))
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            SpectralNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel)),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            SpectralNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel)),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            SpectralNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1)),
            torch.nn.LeakyReLU(),
        )
        return block

    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        # Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(64, 128)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(128, 256)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1,
                                     output_padding=1)
        )
        # Decode
        self.conv_decode3 = self.expansive_block(512, 256, 128)
        self.conv_decode2 = self.expansive_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)
        self.expansive_final_block = self.expansive_final(out_channel, [32, 8], out_channels=2)

        # self.pyramid_feature1 = nn.Sequential(SpectralNorm(nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1)),
        #                                       torch.nn.LeakyReLU(),
        #                                       torch.nn.BatchNorm2d(8),
        #                                       )
        # self.pyramid_feature2 = nn.Sequential(SpectralNorm(nn.Conv2d(in_channels=128, out_channels=8, kernel_size=3, stride=1, padding=1)),
        #                                       torch.nn.LeakyReLU(),
        #                                       torch.nn.BatchNorm2d(8),
        #                                       )
        # self.pyramid_feature3 = nn.Sequential(SpectralNorm(nn.Conv2d(in_channels=256, out_channels=8, kernel_size=3, stride=1, padding=1)),
        #                                       torch.nn.LeakyReLU(),
        #                                       torch.nn.BatchNorm2d(8),
        #                                       )
        # self.pyramid_feature4 = nn.Sequential(SpectralNorm(nn.Conv2d(in_channels=256, out_channels=8, kernel_size=3, stride=1, padding=1)),
        #                                       torch.nn.LeakyReLU(),
        #                                       torch.nn.BatchNorm2d(8),
        #                                       )
        # self.pyramid_feature5 = nn.Sequential(SpectralNorm(nn.Conv2d(in_channels=128, out_channels=8, kernel_size=3, stride=1, padding=1)),
        #                                       torch.nn.LeakyReLU(),
        #                                       torch.nn.BatchNorm2d(8),
        #                                       )
        # self.pyramid_feature6 = nn.Sequential(SpectralNorm(nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1)),
        #                                       torch.nn.LeakyReLU(),
        #                                       torch.nn.BatchNorm2d(8),
        #                                       )

    def crop_and_concat(self, upsampled, bypass, crop=True):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            d = bypass.size(2) - upsampled.size(2) - c
            bypass = bypass[:, :, c:-d, c:-d]
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):

        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)  # go to pyramid1, [3, 64, 126, 126]

        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)  # go to pyramid2, [3, 128, 61, 61]

        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)  # go to pyramid3,[ 3, 256, 28, 28]

        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)  # go to pyramid4, [3, 256, 48, 48]

        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)  # go to pyramid 5, [3, 128, 88, 88]

        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)  # go to pyramid 6, [3, 64, 168, 168]

        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)

        if self.training:
            # flow1 = self.pyramid_feature1(encode_pool1)
            # flow2 = self.pyramid_feature2(encode_pool2)
            # flow3 = self.pyramid_feature3(encode_pool3)
            # flow4 = self.pyramid_feature4(bottleneck1)
            # flow5 = self.pyramid_feature5(cat_layer2)
            # flow6 = self.pyramid_feature6(cat_layer1)

            # return flow4, flow5, flow6, final_layer

            return  final_layer

            # return flow1, flow2, flow3, flow4, flow5, flow6, final_layer
        else:
            return final_layer