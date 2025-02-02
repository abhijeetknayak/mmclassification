# Copyright (c) OpenMMLab. All rights reserved.
from torch import gt
import torch.nn as nn
import torch.nn.functional as F
import torch

from ..builder import HEADS
from .cls_head import ClsHead
import ipdb

# skipped = [0, 0, 0, 0, 0]


@HEADS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 is_train=True,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(LinearClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.is_train = is_train

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')
            
        # print("Reached Here!!!!!!!!!!!!!!!!!!!!!!!!!1")
        print(self.in_channels)

        # Instantiate all linear classifiers
        for idx, in_c in enumerate(self.in_channels):
            exec(f"self.fc{idx} = nn.Linear(in_c, self.num_classes)")

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def all_outputs(self, inputs):
        out = []
        for idx, x in enumerate(inputs):
            exec(f"out.append(self.fc{idx}(x))")
        
        return tuple(out)  

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        if self.is_train:
            cls_scores = self.all_outputs(x)
            for idx, score in enumerate(cls_scores):
                if softmax:
                    pred = F.softmax(score, dim=1) if score is not None else None
                else:
                    pred = score
        else:
            pred = x

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label, **kwargs):
        cls_scores = self.all_outputs(x)
        total_loss = torch.tensor([0.0], requires_grad=True)
        total_len = len(cls_scores)
        losses = []
        for idx, score in enumerate(cls_scores):
            losses.append(self.loss(score, gt_label, **kwargs)['loss'])

        weights = [1/(total_len - idx) for idx in range(total_len)]
        # print(f"Losses: {losses}")
        for idx in range(total_len):
            losses[idx] *= weights[idx]
        # print(f"Weighted Losses: {losses}")        
        
        total_loss = sum(losses)
        # print(f"Total Loss: {total_loss}")
        
        return {'loss': total_loss}
