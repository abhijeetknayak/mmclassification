# Copyright (c) OpenMMLab. All rights reserved.
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales
from .dyn1_neck import GAP

__all__ = ['GlobalAveragePooling', 'GeneralizedMeanPooling', 'HRFuseScales', 'GAP']
