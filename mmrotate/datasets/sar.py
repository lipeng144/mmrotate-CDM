# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.registry import DATASETS
from .hrsc import HRSCDataset


@DATASETS.register_module()
class SARDataset(HRSCDataset):
    """SAR ship dataset for detection (Support RSSDD and HRSID)."""
    CLASSES = ('ship', )
    PALETTE = [
        (0, 255, 0),
    ]
