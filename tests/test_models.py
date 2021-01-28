import torch
from fish.models import FilterCenterNet
from object_detection.entities import ImageBatch


def test_filter_center_net() -> None:
    channels = 128
    num_classes = 2
    image_batch = ImageBatch(torch.rand(1, 3, 256, 256))
    net = FilterCenterNet(
        num_classes=num_classes,
        channels=channels,
    )
    netout, weight = net(image_batch)
    assert weight.shape == (1, num_classes)
