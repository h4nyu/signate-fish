from typing import *
from object_detection.model_loader import WatchMode
from fish.config import *


pseudo_threshold = 0.10

# train
use_amp = True

# data
confidence_threshold = 0.03
iou_threshold = 0.15
batch_size = 3

# model
channels = 32
box_depth = 1
fpn_depth = 1
lr = 1e-3
out_ids: List[int] = [5, 6, 7]

metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

# criterion
topk = 39
box_weight = 1
cls_weight = 2

anchor_ratios = [2 / 3, 1.0, 3 / 2]
anchor_scales = [1.0, 1.44]
num_anchors = len(anchor_ratios) * len(anchor_scales)
anchor_size = 8

out_dir = f"/store/efficientdet-{num_anchors}-{channels}"
