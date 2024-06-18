import torch
import torch.nn as nn
import numpy as np
from .slice import bilateral_slice


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias, padding_mode='replicate')
        self.activation = activation() if activation else None

        if use_bias:
            self.conv.bias.data.fill_(0.00)
        torch.nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        return x


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        bilateral_grid = bilateral_grid.permute(0, 3, 4, 2, 1)
        guidemap = guidemap.squeeze(1)
        coeffs = bilateral_slice(bilateral_grid, guidemap).permute(0, 3, 1, 2)
        return coeffs


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, coeff, ref_image):
        out = ref_image * coeff[:, 0:1, :, :] + coeff[:, 1:2, :, :]
        return out


class Coeffs(nn.Module):

    def __init__(self, nin=2, nout=1, args=None):
        super(Coeffs, self).__init__()
        self.args = args
        self.nin = nin
        self.nout = nout

        lb = args.luma_bins
        num_down_layers = args.num_down_layers
        feature_layers = args.feature_layers
        cm = args.channel_multiplier

        self.down_layers = nn.ModuleList()
        prev_ch = 3
        low_res_channels = cm * (2 ** (num_down_layers - 1)) * lb
        for i in range(num_down_layers):
            cur_ch = cm * (2 ** i) * lb
            self.down_layers.append(ConvBlock(prev_ch, cur_ch, 3, stride=2))
            prev_ch = cur_ch

        self.features = nn.ModuleList()
        for i in range(feature_layers):
            self.features.append(ConvBlock(low_res_channels, low_res_channels, 3))
        self.features.append(ConvBlock(low_res_channels, low_res_channels, 3, use_bias=False))

        self.conv_out = ConvBlock(low_res_channels, lb * nout * nin, 1, padding=0, activation=None)

    def forward(self, x):
        for layer in self.down_layers:
            x = layer(x)

        for layer in self.features:
            x = layer(x)

        x = self.conv_out(x)
        y = torch.stack(torch.split(x, self.nin * self.nout, 1), 2)
        return y

class DGNet(nn.Module):

    def __init__(self, args):
        super(DGNet, self).__init__()
        self.coeffs = Coeffs(args=args)
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

    def forward(self, ref_image, masked_image, mask):
        net_input = torch.cat([ref_image, masked_image, 1 - mask], dim=1)
        coeffs = self.coeffs(net_input)
        slice_coeffs = self.slice(coeffs, ref_image)
        out = self.apply_coeffs(slice_coeffs, ref_image)
        return out