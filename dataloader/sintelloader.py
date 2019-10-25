from __future__ import print_function, division
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.getsintelpath import getSintelPairFrame
from utils.masktransform import WDTransformer
from utils.censustransform import censustransform
import torch
from utils.flowread import getflow
from utils.flow2rgb import flow2rgb
from torchvision.transforms import ToPILImage, ToTensor
from utils.warper import warper

USE_CUT_OFF = 10
import random


class SintelDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, nsample = None, test = False, visualize = False, shape=(256, 256), use_l2=True, channel_in=3, stride=1, kernel_size=2, transform=None,
                 usecst=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pairframe = getSintelPairFrame(root_dir, nsample, test)
        # if USE_CUT_OFF: self.pairframe = random.sample(self.pairframe, USE_CUT_OFF)
        self.transform = transform
        self.usecst = usecst
        self.cst = lambda x: censustransform(x) if self.usecst else x
        self.flow = WDTransformer(shape=shape, use_l2=use_l2, channel_in=channel_in, stride=stride,
                                  kernel_size=kernel_size)
        self.nsample = nsample
        self.test = test
        self.visualize = visualize

        self.transformunet = transforms.Compose([transforms.Resize((164,164)),
                                             transforms.ToTensor()])

    def __len__(self):
        return len(self.pairframe)

    def __getitem__(self, idx):
        instance = self.pairframe[idx]
        frame1, frame2 = [*map(lambda x: Image.open(x),instance['frame'])]
        sample = {}
        if self.transform:
            sample['frame1'] = self.transform(frame1)
            sample['frame2'] = self.transform(frame2)

            sample['frame1Unet'] = self.transformunet(frame1)
            sample['frame2Unet'] = self.transformunet(frame2)

            if self.visualize and (not self.test):
                flow = torch.tensor(getflow(instance['flow'])).permute(2,0,1).unsqueeze(0)
                frame1Unet_ = self.transformunet(ToPILImage()(warper(flow, ToTensor()(frame2).unsqueeze(0), scaled = False, nocuda = True)[0]))
                flow = flow2rgb(flow)[0]
                sample['flow'] = self.transform(ToPILImage()(flow))
                sample['occlusion'] = self.transform(Image.open(instance['occlusion']))

                sample['flowUnet'] = self.transformunet(ToPILImage()(flow))
                sample['occlusionUnet'] = self.transformunet(Image.open(instance['occlusion']))
                sample['frame1Unet_'] = frame1Unet_
        return sample


class SintelLoader(object):
    def __init__(self, sintel_root="/data/keshav/sintel/training/final", nsample = None, test = False, visualize = False, transform_resize = (256,256),**loaderconfig):
        self.transform = transforms.Compose([transforms.Resize(transform_resize),
                                             transforms.ToTensor()])
        self.sinteldataset = SintelDataset(root_dir=sintel_root, transform=self.transform, nsample=nsample, test=test, visualize=visualize)
        self.loaderconfig = loaderconfig

        if nsample:
            self.loaderconfig.update({'batch_size':nsample})

    def load(self):
        return DataLoader(self.sinteldataset, **self.loaderconfig)
