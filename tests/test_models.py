import torch
from fish.models import FrameCenterNet
from object_detection.entities import ImageBatch
from object_detection.models.backbones.effnet import (
    EfficientNetBackbone,
)


def test_frame_center_net() -> None:
    channels = 128

    image_batch0 = ImageBatch(torch.rand(1, 3, 256, 256))
    image_batch1 = ImageBatch(torch.rand(1, 3, 256, 256))
    net = FrameCenterNet(
        num_classes=2,
        channels=channels,
    )
    netout = net(image_batch0, image_batch1)
