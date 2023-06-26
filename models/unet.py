import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import zarr
import dask
import dask.array as da
import numpy as np


class DownsamplingUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 batch_norm=True,
                 downsample_op=nn.MaxPool2d):
        super(DownsamplingUnit, self).__init__()
        if downsample_op is None:
            downsample_op = nn.Identity

        self._dwn_sample = downsample_op(kernel_size=2, stride=2, padding=0)
        self._c1 = nn.Conv2d(in_channels, out_channels,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size//2,
                             bias=False)

        if batch_norm:
            self._bn1 = nn.GroupNorm(num_groups=out_channels,
                                     num_channels=out_channels)
        else:
            self._bn1 = nn.Identity()

        self._c2 = nn.Conv2d(out_channels, out_channels,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size//2,
                             bias=False)

        if batch_norm:
            self._bn2 = nn.GroupNorm(num_groups=out_channels,
                                     num_channels=out_channels)
        else:
            self._bn2 = nn.Identity()

        self._relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fx = self._dwn_sample(x)
        fx = self._c1(fx)
        fx = self._bn1(fx)
        fx = self._relu(fx)
        fx = self._c2(fx)
        fx = self._bn2(fx)
        fx = self._relu(fx)
    
        return fx


class UpsamplingUnit(nn.Module):
    def __init__(self, in_channels, unit_channels, out_channels, kernel_size=3,
                 batch_norm=True,
                 upsample_op=nn.ConvTranspose2d):
        super(UpsamplingUnit, self).__init__()
        if upsample_op is None:
            upsample_op = nn.Identity

        self._c1 = nn.Conv2d(in_channels, unit_channels,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size//2,
                             bias=False)
        if batch_norm:
            self._bn1 = nn.GroupNorm(num_groups=unit_channels,
                                     num_channels=unit_channels)
        else:
            self._bn1 = nn.Identity()

        self._c2 = nn.Conv2d(unit_channels, unit_channels,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size//2,
                             bias=False)
        if batch_norm:
            self._bn2 = nn.GroupNorm(num_groups=unit_channels,
                                     num_channels=unit_channels)
        else:
            self._bn2 = nn.Identity()

        self._relu = nn.ReLU(inplace=True)
        self._up_sample = upsample_op(unit_channels, out_channels,
                                      kernel_size=2,
                                      stride=2,
                                      padding=0,
                                      output_padding=0,
                                      bias=True)

    def forward(self, x):
        fx = self._c1(x)
        fx = self._bn1(fx)
        fx = self._relu(fx)
        fx = self._c2(fx)
        fx = self._bn2(fx)
        fx = self._relu(fx)
        fx = self._up_sample(fx)
        return fx


class BottleneckUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 batch_norm=True):
        super(BottleneckUnit, self).__init__()
        self._dwn_sample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self._c1 = nn.Conv2d(in_channels, out_channels,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size//2,
                             bias=False)

        if batch_norm:
            self._bn1 = nn.GroupNorm(num_groups=out_channels,
                                     num_channels=out_channels)
        else:
            self._bn1 = nn.Identity()

        self._c2 = nn.Conv2d(out_channels, out_channels,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size//2,
                             bias=False)

        if batch_norm:
            self._bn2 = nn.GroupNorm(num_groups=out_channels,
                                     num_channels=out_channels)
        else:
            self._bn2 = nn.Identity()

        self._relu = nn.ReLU(inplace=True)
        self._up_sample = nn.ConvTranspose2d(out_channels, in_channels,
                                             kernel_size=2, stride=2,
                                             padding=0,
                                             output_padding=0,
                                             bias=True)

    def forward(self, x):
        fx = self._dwn_sample(x)
        fx = self._c1(fx)
        fx = self._bn1(fx)
        fx = self._relu(fx)
        fx = self._c2(fx)
        fx = self._bn2(fx)
        fx = self._relu(fx)
        fx = self._up_sample(fx)
        return fx


class UNet(nn.Module):
    def __init__(self, **kwargs):
        super(UNet, self).__init__()

        self.analysis_track = nn.ModuleList([
            DownsamplingUnit(in_channels=3,
                             out_channels=64,
                             kernel_size=3,
                             downsample_op=None,
                             batch_norm=True),                             
            DownsamplingUnit(in_channels=64,
                             out_channels=128,
                             kernel_size=3,
                             downsample_op=nn.MaxPool2d,
                             batch_norm=True),
            DownsamplingUnit(in_channels=128,
                             out_channels=256,
                             kernel_size=3,
                             downsample_op=nn.MaxPool2d,
                             batch_norm=True),
            DownsamplingUnit(in_channels=256,
                             out_channels=512,
                             kernel_size=3,
                             downsample_op=nn.MaxPool2d,
                             batch_norm=True),

        ])

        self.bottleneck = BottleneckUnit(in_channels=512, out_channels=1024,
                                         kernel_size=3,
                                         batch_norm=True)

        self.synthesis_track = nn.ModuleList([
            UpsamplingUnit(in_channels=1024, unit_channels=512,
                           out_channels=256,
                           kernel_size=3,
                           upsample_op=nn.ConvTranspose2d,
                           batch_norm=True),
            UpsamplingUnit(in_channels=512, unit_channels=256,
                           out_channels=128,
                           kernel_size=3,
                           upsample_op=nn.ConvTranspose2d,
                           batch_norm=True),
            UpsamplingUnit(in_channels=256, unit_channels=128,
                           out_channels=64,
                           kernel_size=3,
                           upsample_op=nn.ConvTranspose2d,
                           batch_norm=True),
            UpsamplingUnit(in_channels=128, unit_channels=64,
                           out_channels=64,
                           kernel_size=3,
                           upsample_op=None,
                           batch_norm=True)
        ])

        # Classifier
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True)

    def forward(self, x):
        fx = x
        fx_brg = []

        for layer in self.analysis_track:
            fx = layer(fx)
            fx_brg.insert(0, fx)

        # Bottleneck
        fx = self.bottleneck(fx)

        for fx_b, layer in zip(fx_brg, self.synthesis_track):
            fx = torch.cat((fx_b, fx), dim=1)
            fx = layer(fx)

        # Pixel-wise class prediction
        y = self.fc(fx)
        return y
