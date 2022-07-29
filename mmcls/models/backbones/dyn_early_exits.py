# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcls.models.heads.cls_head import ClsHead
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
import torch
import ipdb
import time

from ..builder import BACKBONES
from .resnet import ResNet

exit_info = [0, 0, 0, 0, 0]
total_time = 0.0

class GAP(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dim=2, img_size=4):
        super(GAP, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'
        self.img_size = img_size
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(self.img_size)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((self.img_size, self.img_size))
        else:
            self.gap = nn.AdaptiveAvgPool3d((self.img_size, self.img_size, self.img_size))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs

class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes, load_file) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        s = torch.load(load_file)["state_dict"]
        weight_dict = {}

        for name, val in s.items():
            if "head" in name:
                weight_dict[name] = val

        for idx, in_c in enumerate(self.in_channels):
            exec(f"self.fc{idx} = nn.Linear(in_c, self.num_classes)")
            with torch.no_grad():
                exec(f"self.fc{idx}.weight.copy_(weight_dict['head.fc{idx}.weight'])")
                exec(f"self.fc{idx}.bias.copy_(weight_dict['head.fc{idx}.bias'])")

        # ipdb.set_trace()
        # print(f"Weight Dict: {weight_dict['head.fc0.weight']}")
        # print(f"Stored: {self.fc0.weight}")
        
    def forward(self, x, idx, softmax=True):
        x = eval(f"self.fc{idx}")(x)

        if softmax:
            pred = F.softmax(x, dim=1)
        
        return pred

@BACKBONES.register_module()
class EarlyExitNet(ResNet):
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

    def __init__(self, depth, deep_stem=False, is_train=True, load_file="", **kwargs):
        super(EarlyExitNet, self).__init__(
            depth, deep_stem=deep_stem, **kwargs)
        assert not self.deep_stem, 'ResNet_CIFAR do not support deep_stem'
        self.is_train = is_train

        # Image size after Global Average Pooling
        self.gis = 1
        self.gis2 = self.gis ** 2
        self.filters = [64, 64, 128, 256, 512]

        in_channels = [self.gis2 * filter for filter in self.filters]

        self.gap = GAP(img_size=self.gis)

        if not self.is_train:
            self.cls = Classifier(in_channels=in_channels, num_classes=10, load_file=load_file)

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
        # ipdb.set_trace()
        global total_time
        start_time = time.process_time()
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        # outs = [x]

        # if not self.is_train:
        preds = self.cls.forward(self.gap(x), 0)
        end_time = time.process_time()
        if torch.max(preds) >= 0.8:                
            exit_info[0] += 1
            print(exit_info) 
            print(f"Processing Time: {end_time - start_time}")
            total_time += (end_time - start_time)
            print(f"Total elapsed time: {total_time}")
            return preds
            
        # print(0, "------------------->", x.shape)

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            # if not self.is_train:
            preds = self.cls.forward(self.gap(x), i + 1)
            end_time = time.process_time()
            if torch.max(preds) >= 0.8 or i == len(self.res_layers) - 1:                    
                exit_info[i + 1] += 1
                print(exit_info)
                print(f"Processing Time: {end_time - start_time}")
                total_time += (end_time - start_time)
                print(f"Total elapsed time: {total_time}")
                return preds
            # else:
            #     outs.append(x)  
        # preds = self.cls.forward(self.gap(x), 4) 
        # end_time = time.process_time()
        # print(f"Processing Time: {end_time - start_time}")
        # total_time += (end_time - start_time)
        # print(f"Total elapsed time: {total_time}")
        
        
        # return preds
        
        # return self.gap(tuple(outs))
