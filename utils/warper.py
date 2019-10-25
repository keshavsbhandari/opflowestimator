import torch.nn.functional as F

import torch


def warper(flow, img, scaled=True,nocuda = False):
    flow = flow.permute(0,2,3,1)
    b, h, w, c = flow.size()
    if not scaled:
        if nocuda:
            flow = flow / torch.tensor([h, w])
        else:
            flow = flow / torch.tensor([h, w]).cuda()


    meshgrid = torch.cat([torch.linspace(-1.0, 1.0, w).view(1, 1, w, 1).expand(b, h, w, 1),
                          torch.linspace(-1.0, 1.0, h).view(1, h, 1, 1).expand(b, h, w, 1)], -1)

    if nocuda:
        grid = (meshgrid + flow)
    else:
        grid = (meshgrid.cuda() + flow)
    warped = F.grid_sample(input=img, grid=grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped


# tensorFlow = uv/torch.tensor([436,1024]).float()
#
# b,h,w,c = tensorFlow.size()
#
# torchHorizontal = torch.linspace(-1.0, 1.0, w).view(1, 1, w, 1).expand(b, h, w, 1)
# torchVertical = torch.linspace(-1.0, 1.0, h).view(1, h, 1, 1).expand(b, h, w, 1)
#
# meshgrid = torch.cat([torch.linspace(-1.0, 1.0, w).view(1, 1, w, 1).expand(b, h, w, 1),
#                       torch.linspace(-1.0, 1.0, h).view(1, h, 1, 1).expand(b, h, w, 1)],
#                      -1)
#
# grid=(meshgrid + tensorFlow)
# outim = F.grid_sample(input=img, grid=grid, mode='bilinear', padding_mode='border')
# toimage(outim[0])