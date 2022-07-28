import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random

def warping(tensor):
    identity = torch.Tensor([[[1., 0, 0], [0, 1, 0], [0, 0, 1]]]).cuda()
    mask = torch.Tensor([[[1., 1, 1], [1, 1, 1], [0, 0, 0]]]).cuda()
    noise = torch.randn((tensor.shape[0], 3, 3)).cuda()

    forward = identity + .075 * noise * mask


    forward_grid = F.affine_grid(forward[:, :2], tensor[:, :3].shape)
    
    warped_input = F.grid_sample(tensor, forward_grid)
    
    return warped_input

def apply_warped(net, tensor):
    identity = torch.Tensor([[[1., 0, 0], [0, 1, 0], [0, 0, 1]]]).cuda()
    mask = torch.Tensor([[[1., 1, 1], [1, 1, 1], [0, 0, 0]]]).cuda()
    noise = torch.randn((tensor.shape[0], 3, 3)).cuda()

    forward = identity + .075 * noise * mask

    backward = torch.inverse(forward)
    
    if random.random() < .5:
        forward, backward = backward, forward

    forward_grid = F.affine_grid(forward[:, :2], tensor[:, :3].shape)
   
    
    warped_input = F.grid_sample(tensor, forward_grid)
    
    warped_output = net(warped_input)
    
    backward_grid = F.affine_grid(backward[:, :2], warped_output.shape)
    
    unwarped_output = F.grid_sample(warped_output, backward_grid)
    
    return unwarped_output

class UNet(nn.Module):
    def __init__(self, num_layers, channels, dimension):
        super().__init__()

        self.num_layers = num_layers
        down_channels = np.array(channels[0])
        up_channels_out = np.array(channels[1][1:])
        up_channels_in = down_channels[1:] + np.concatenate([up_channels_out[1:], [0]])
        self.downConvs = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])

        for depth in range(self.num_layers):
            self.downConvs.append(
                nn.Conv2d(
                    down_channels[depth],
                    down_channels[depth + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
            self.upConvs.append(
                nn.ConvTranspose2d(
                    up_channels_in[depth],
                    up_channels_out[depth],
                    kernel_size=4,
                    padding=1,
                    stride=2,
                )
            )

        self.lastConv = nn.Conv2d(17, channels[1][0], kernel_size=3, padding=1)

    def forward(self, x):
        skips = []
        for depth in range(self.num_layers):
            skips.append(x)
            x = F.gelu(self.downConvs[depth](x))
        for depth in reversed(range(self.num_layers)):
            x = F.gelu(self.upConvs[depth](x))

            x = x[:, :, : skips[depth].size()[2], : skips[depth].size()[3]]
            x = torch.cat([x, skips[depth]], 1)
        x = self.lastConv(x)
        return x

class WarpedNetwork(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        return apply_warped(self.net, x)
