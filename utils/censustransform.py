import torch


def censustransform(img):
    """
    :param img: pytorch tensor with any format [h,w],[c,h,w],[b,c,h,w]
    :return:
    """
    size = img.size()
    if img.dim() == 2:
        """
        Assuming image with one channel and only h,w
        """
        img = img[(None,) * 2]
    elif img.dim() == 3:
        """
        Assuming single batch image with c,h,w
        """
        img.unsqueeze_(0)

    elif img.dim() == 4:
        """
        Assuming image with batches b,c,h,w
        """
        pass
    else:
        raise Exception("Only supported 2<=img.dim()>=4, supported format [h,w],[c,h,w],[b,c,h,w]")

    B, C, H, W = img.size()
    unfold = torch.nn.Unfold(3, padding=1)
    img = unfold(img).view(B, C, 9, -1).permute(0, 1, 3, 2)
    mid = img[:, :, :, 4].unsqueeze(-1)
    #NEW WAY START
    encoding = (img >= mid).float().view(-1, 9) * torch.tensor([128., 64., 32., 16., 0., 8., 4., 2., 1.]).cuda()
    return encoding.sum(-1).view(*size)/255
