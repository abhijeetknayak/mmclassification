# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer

from ..builder import BACKBONES
from .resnet import ResNet

class SoftGateI(nn.Module):
    """This module has the same structure as FFGate-I.
    In training, adopt continuous gate output. In inference phase,
    use discrete gate outputs"""
    def __init__(self, pool_size=5, channel=10):
        super(SoftGateI, self).__init__()
        self.pool_size = pool_size
        self.channel = channel

        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = conv3x3(channel, channel)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)

        # adding another conv layer
        self.conv2 = conv3x3(channel, channel, stride=2)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU(inplace=True)

        pool_size = math.floor(pool_size/2)  # for max pooling
        pool_size = math.floor(pool_size/2 + 0.5)  # for conv stride = 2

        self.avg_layer = nn.AvgPool2d(pool_size)
        self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        self.logprob = nn.LogSoftmax()

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x).squeeze()
        softmax = self.prob_layer(x)
        logprob = self.logprob(x)

        x = softmax[:, 1].contiguous()
        x = x.view(x.size(0), 1, 1, 1)

        if not self.training:
            x = (x > 0.5).float()
        return x, logprob



@BACKBONES.register_module()
class Dyn1(ResNet):
    """ResNet backbone for CIFAR.

    Compared to standard ResNet, it uses `kernel_size=3` and `stride=1` in
    conv1, and does not apply MaxPoolinng after stem. It has been proven to
    be more efficient than standard ResNet in other public codebase, e.g.,
    `https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py`.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): This network has specific designed stem, thus it is
            asserted to be False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
    """

    def __init__(self, depth, deep_stem=False, **kwargs):
        super(Dyn1, self).__init__(
            depth, deep_stem=deep_stem, **kwargs)
        assert not self.deep_stem, 'ResNet_CIFAR do not support deep_stem'
        self.gating_network1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d()

        )

    def _make_stem_layer(self, in_channels, base_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, base_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            outs.append(x)
        
        return tuple(outs)
