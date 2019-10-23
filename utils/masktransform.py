import torch
from itertools import chain
from utils.censustransform import censustransform

"""
Window Displacement Transformer : WDTransformer
"""


class WDTransformer(torch.nn.Module):
    def __init__(self, shape=(436, 1024), channel_in=3, use_l2=False, use_cst=True, **unfold_args):
        """
        @param shape:contains shape of an input image in width,height order
        @type shape: tuple
        @param channel_in: number of input channel
        @type channel_in: int
        @param use_l2: use l2 distance if True else use cosine similarity as measurement metrics between two pixel
        @type use_l2: bool
        @param unfold_args: contains unfold arguments mainly kernel_size(int), padding(int), stride(int) and dilation(int)
        @type unfold_args: dict
        """
        super(WDTransformer, self).__init__()
        self.cst = lambda x: censustransform(x) if use_cst else x
        self.use_l2 = use_l2
        self.channel_in = channel_in
        self.shape_in = shape
        self.kernel_size = unfold_args.get('kernel_size', 2)
        self.padding = unfold_args.get('padding', 0)
        self.stride = unfold_args.get('stride', 1)
        self.dilation = unfold_args.get('dilation', 1)
        self.shape_out = self.get_shape()
        self.unfold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)
        self.fold = torch.nn.Fold(output_size=self.shape_in, kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x, y):
        """
        @param x:first frame
        @type x: tensor with size [b,c,h,w]
        @param y: second frame
        @type y: tensor with size [b,c,h,w]
        @return: window wise approximated displacement
        @rtype: tensor with size [b,2,h,w]
        """

        x, y = self.cst(x), self.cst(y)

        batch_size, _, _, _ = x.shape
        if self.use_l2:
            x = self.unfold(x).permute(0, 2, 1).contiguous().view(-1, self.kernel_size ** 2, 1)
            y = self.unfold(y).permute(0, 2, 1).contiguous().view(-1, 1, self.kernel_size ** 2)
            xy = torch.abs(x - y).view(batch_size, *self.shape_out, self.channel_in, self.kernel_size ** 2,
                                       self.kernel_size ** 2)
            xy = xy.pow(2).sum(-3).pow(0.5)
        else:
            x = self.unfold(x).permute(0, 2, 1).contiguous().view(-1, self.kernel_size ** 2)
            y = self.unfold(y).permute(0, 2, 1).contiguous().view(-1, self.kernel_size ** 2)
            x = x.unsqueeze(-1) - torch.zeros_like(x).unsqueeze(-2)
            y = torch.zeros_like(y).unsqueeze(-1) - y.unsqueeze(-2)
            x = x.view(batch_size, *self.shape_out, self.channel_in, self.kernel_size ** 2, self.kernel_size ** 2)
            y = y.view(batch_size, *self.shape_out, self.channel_in, self.kernel_size ** 2, self.kernel_size ** 2)
            xy = torch.nn.functional.cosine_similarity(x, y, -3)

        xy = xy.view(batch_size, *self.shape_out, self.kernel_size ** 2, self.kernel_size, self.kernel_size)
        x = xy.min(-1)[1].max(-1)[0]
        y = xy.min(-2)[1].max(-1)[0]

        x = x - torch.tensor(list(range(self.kernel_size)) * self.kernel_size).float().cuda()
        y = y - torch.tensor([*chain.from_iterable([[i] * self.kernel_size for i in range(self.kernel_size)])]).float().cuda()

        x = x.view(batch_size, -1, self.kernel_size ** 2).transpose(2, 1)
        y = y.view(batch_size, -1, self.kernel_size ** 2).transpose(2, 1)

        xy = torch.cat((x, y), 1)
        xy = self.fold(xy.float().cuda()) # this is an approx flow per kernel
        # xy = xy  # standardize by max value
        h, w = self.shape_in
        grid = torch.stack(torch.meshgrid([torch.linspace(-1, 1, w), torch.linspace(-1, 1, h)])).permute(2, 1,
                                                                                                         0).unsqueeze(0).float().cuda()
        xy = xy.permute(0, 2, 3, 1) / torch.tensor([436., 1024.]).float().cuda() + grid
        return xy.permute(0,3,1,2)

    def get_shape(self):
        return [*map(lambda x: ((x + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride) + 1,
                     self.shape_in)]
