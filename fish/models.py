import torch
from torch import nn, Tensor
from object_detection.models.centernet import CenterNet, NetOutput
from object_detection.models.backbones.resnet import ResNetBackbone
from object_detection.entities import ImageBatch
import torchvision.models as models


class FrameCenterNet(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        box_depth: int = 1,
        cls_depth: int = 1,
        fpn_depth: int = 1,
        out_idx: int = 4,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=6,
            out_channels=3,
            kernel_size=1,
        )
        backbone = ResNetBackbone("resnet50", out_channels=channels)
        self.net = CenterNet(
            channels=channels,
            num_classes=num_classes,
            backbone=backbone,
            box_depth=box_depth,
            cls_depth=cls_depth,
            fpn_depth=fpn_depth,
            out_idx=out_idx,
        )

    def __call__(self, image0: ImageBatch, image1: ImageBatch) -> NetOutput:
        dualImage = torch.cat([image0, image1], dim=1)
        image = self.conv(dualImage)
        return self.net(image)
