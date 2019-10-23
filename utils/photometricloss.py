import torch
from utils.warper import warper
from torchvision.transforms import Normalize
import torch.nn.functional as F

def normalize(x, mean = 0, std = 0.1):
    return Normalize(mean,std)(x)

def photometricloss(I, I_, occ, eps=1e-2, q=4e-1):
    ssimloss = ssim(I, I_)
    # I = I/I.norm()
    # I_ = I_ / I_.norm()
    error = torch.pow(torch.abs(I - I_) + eps, q)
    error = error  * occ
    # error = torch.pow(torch.abs(I-I_))*occ
    occsum = occ.view(occ.size(0), -1).sum(-1).unsqueeze(-1)
    error = error.view(error.size(0), -1) / occsum
    reconstruction_loss = error.sum()/I.size(0)
    reconstruction_loss +=ssimloss
    assert ((reconstruction_loss == reconstruction_loss).item() == 1)
    return reconstruction_loss

def ssim(x, y):
    # x = x * occ
    # y = y * occ

    mse = F.mse_loss(x,y)
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

    loss = mse+ torch.clamp((1 - SSIM) / 2, 0, 1).mean()
    return  loss


