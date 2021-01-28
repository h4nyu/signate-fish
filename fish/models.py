import torch
from typing import *
from torch import nn, Tensor
from object_detection.models.centernet import CenterNet, NetOutput, Head
from object_detection.models.backbones.effnet import (
    EfficientNetBackbone,
)
from object_detection.models.modules import (
    MaxPool2dStaticSamePadding,
    SeparableConvBR2d,
    MemoryEfficientSwish,
)
from object_detection.entities import ImageBatch
from fish import config


class FilterCenterNet(nn.Module):
    def __init__(
        self,
        channels: int,
        box_depth: int = 1,
        cls_depth: int = 1,
        fpn_depth: int = 1,
        out_idx: int = 4,
        backbone_id: int = 3,
        num_classes: int = config.num_classes,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        backbone = EfficientNetBackbone(3, out_channels=channels, pretrained=True)
        self.net = CenterNet(
            channels=channels,
            num_classes=num_classes,
            backbone=backbone,
            box_depth=box_depth,
            cls_depth=cls_depth,
            fpn_depth=fpn_depth,
            out_idx=out_idx,
        )
        self.filter_head = nn.Sequential(
            SeparableConvBR2d(in_channels=channels),
            MemoryEfficientSwish(),
            MaxPool2dStaticSamePadding(3, 2),
            SeparableConvBR2d(in_channels=channels, out_channels=channels * 2),
            MemoryEfficientSwish(),
            MaxPool2dStaticSamePadding(3, 2),
            SeparableConvBR2d(in_channels=channels * 2, out_channels=num_classes),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid(),
        )

    def __call__(self, x: ImageBatch) -> Tuple[NetOutput, Tensor]:
        (heat_maps, box_maps, anchor), fpn = self.net(x)
        fh = self.filter_head(fpn[-1])
        heat_maps = fh * heat_maps
        return (heat_maps, box_maps, anchor), fh.view(-1, self.num_classes)
