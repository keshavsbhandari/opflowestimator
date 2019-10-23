import torch.nn.functional as F

import torch


def warper(flow, img, scaled=True):
    flow = flow.permute(0,2,3,1)
    b, h, w, c = flow.size()
    if not scaled: flow = flow / torch.tensor([h, w]).float()

    meshgrid = torch.cat([torch.linspace(-1.0, 1.0, w).view(1, 1, w, 1).expand(b, h, w, 1),
                          torch.linspace(-1.0, 1.0, h).view(1, h, 1, 1).expand(b, h, w, 1)], -1)

    grid = (meshgrid.cuda() + flow)
    warped = F.grid_sample(input=img, grid=grid, mode='bilinear', padding_mode='border', align_corners=True)

    return warped
