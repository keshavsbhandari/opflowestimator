from utils.censustransform import censustransform
import torch
from utils.warper import warper
from torchvision.transforms import Normalize
import torch.nn.functional as F


def normalize(x, mean=0, std=0.1):
    return Normalize(mean, std)(x)

def photometricloss(I, I_, I1, occ):
    ssimloss = ssim(I, I_)
    losslimit = reconloss(I,I_,occ)
    realloss = reconloss(I,I1,occ)
    photomax = torch.min(losslimit, realloss)
    return ssimloss + photomax



def reconloss(I, I_, occ, eps=1e-2, q=4e-1):
    error =  torch.abs(I-I_)
    error = torch.pow(error + eps, q)
    occ = 0.01 * torch.abs(occ) + 0.5
    error = error * occ
    occsum = occ.view(occ.size(0), -1).sum(-1).unsqueeze(-1)
    error = error.view(error.size(0), -1) / occsum
    error = error.view(error.size(0), -1) / occsum
    reconstruction_loss = error.sum() / I.size(0)
    return reconstruction_loss

def ssim(x, y):
    # x = x * occ
    # y = y * occ

    mse = F.mse_loss(x, y)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)

    sigma_x = F.avg_pool2d(x ** 2, 3, 1)
    sigma_y = F.avg_pool2d(y ** 2, 3, 1)
    sigma_xy = F.avg_pool2d(x * y, 3, 1)

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    loss = mse + torch.clamp((1 - SSIM) / 2, 0, 1).mean()
    return loss
