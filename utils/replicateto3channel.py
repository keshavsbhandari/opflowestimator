def replicatechannel(x):
    """
    :param x: [B,1,H,W]
    :type x: an image with channel 1
    :return: [B,3,H,W]
    :rtype: an image with replicated first channel
    """
    return x.view(1, -1).repeat(3, 1).view(3, x.size(0), x.size(2), x.size(3)).permute(1, 0, 2, 3)