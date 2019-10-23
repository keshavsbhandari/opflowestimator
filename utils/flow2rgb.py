import torch


# PURE PYTORCH

def flow2rgb(flow_map):
    # EXPECTED : B,C,H,W
    B, C, H, W = flow_map.size()
    condition = (flow_map.select(1, 0) == 0) & (flow_map.select(1, 1) == 0)
    condition = condition.view(B, -1).repeat(1, 2).view(*flow_map.size())
    fill = torch.ones_like(flow_map) * -1e9
    flow_map = torch.where(condition, fill, flow_map)
    flow_map = flow_map / flow_map.max()
    rgb_map = torch.ones(B, 3, H, W)
    r = flow_map.select(1, 0)
    g = -0.5 * (flow_map.select(1, 1) + r)
    b = flow_map.select(1, 1)
    rgb = torch.stack([r, g, b], 1)
    rgb_map = rgb_map + rgb
    return rgb_map.clamp(0, 1)
