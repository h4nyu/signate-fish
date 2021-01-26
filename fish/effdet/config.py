from typing import *
from object_detection.model_loader import WatchMode
from fish.config import *


pseudo_threshold = 0.10

# train
use_amp = True

# data
confidence_threshold = 0.01
iou_threshold = 0.2
batch_size = 7

# model
channels = 64
box_depth = 1
fpn_depth = 1
lr = 5e-4
out_ids: List[int] = [5, 6, 7]

metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

# criterion
topk = 13
box_weight = 8
cls_weight = 1

anchor_ratios = [1.0, 1.3, 1.9]
anchor_scales = [1.0, 1.44]
num_anchors = len(anchor_ratios) * len(anchor_scales)
anchor_size = 3

out_dir = f"/store/efficientdet-{num_anchors}-{channels}"
