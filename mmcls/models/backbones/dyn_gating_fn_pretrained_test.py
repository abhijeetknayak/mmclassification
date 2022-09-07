# Copyright (c) OpenMMLab. All rights reserved.
from pyexpat import model
from tracemalloc import start
import torch.nn as nn
from mmcls.models.heads.cls_head import ClsHead
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
import torch
import ipdb
import math
import time
from .base_backbone import BaseBackbone

from ..builder import BACKBONES
from .resnet import ResLayer, ResNet

skipped = [0, 0, 0, 0]
total_time = 0.0

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SoftGateII(nn.Module):
    """ Soft gating version of FFGate-II"""
    def __init__(self, pool_size=5, channel=10):
        super(SoftGateII, self).__init__()
        self.mp = nn.MaxPool2d(2, 2)
        self.pool_size = pool_size
        self.channel = channel

        self.conv1 = conv3x3(channel, 2, stride=2)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu1 = nn.ReLU(inplace=True)

        self.avg_layer = nn.AvgPool2d(pool_size)
        # self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=2,
        #                               kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        self.eps = 1.0
        self.eps_decay = 0.99

    def forward(self, x):
        # ipdb.set_trace()
        # x = self.mp(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.avg_layer(x)
        # x = self.linear_layer(x)
        x = torch.flatten(x, start_dim=1)

        softmax = self.prob_layer(x)

        x = softmax[:, 1].contiguous()
        x = x.view(x.size(0), 1, 1, 1)
        # print(x)
        if True: #not self.training:
            x = (x > 0.5).float()
        return x

@BACKBONES.register_module()
class GatingFnNewPretrainedTest(BaseBackbone):
    def __init__(self, is_train=True, init_cfg=None, **kwargs):
        super(GatingFnNewPretrainedTest, self).__init__(init_cfg)
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # for params in self.resnet.parameters():
        #     params.requires_grad = False

        self.is_train = is_train

        # Number of channels after two convolution layers
        self.channels = [64, 64, 128, 256]
        # self.pool_size = [8, 8, 8, 4, 4, 2, 2, 1] # With SoftGateI
        self.pool_size = [8, 8, 4, 2]
        self.pool_size = [16, 16, 8, 4]

        for idx, channel in enumerate(self.channels):
            exec(f"self.gating_fn{idx} = SoftGateII(pool_size={self.pool_size[idx]}, channel={channel})")

    def forward(self, x):
        global total_time
        start_time = time.process_time()
        # ipdb.set_trace()
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        gating_idx = 0

        for i in range(1, 5):
            # gate_out = eval(f"self.gating_fn{gating_idx}")(x)
            gate_out = torch.Tensor([1.0]).to('cuda')
            if gate_out == 0.0:
                skipped[gating_idx] += 1

            identity = x

            if gate_out == 1.0:
                x = eval(f"self.resnet.layer{i}")(x)
                # x = gate_out.expand_as(x) * x + \
                #     (1 - gate_out).expand_as(identity) * identity
            else:
                if eval(f"self.resnet.layer{i}[0]").downsample is not None:
                    identity = eval(f"self.resnet.layer{i}[0].downsample")(identity)
                x = identity

            gating_idx += 1
            
        end_time = time.process_time()
        print(f"This took {end_time - start_time} seconds")
        total_time += (end_time - start_time)
        print(f"Total time: {total_time}")

        print(skipped)
        return x