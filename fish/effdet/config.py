from typing import *
from object_detection.model_loader import WatchMode
from fish.config import *


pseudo_threshold = 0.15

# train
use_amp = True

# data
confidence_threshold = 0.05
iou_threshold = 0.15
batch_size = 3

# model
channels = 96
box_depth = 1
fpn_depth = 1
lr = 5e-4
out_ids: List[int] = [4, 5, 6, 7]

metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

# criterion
topk = 39
box_weight = 3
cls_weight = 1

anchor_ratios = [1.3, 1.9, 2.9]
anchor_scales = [1.0, 1.44]
anchor_size = 2

out_dir = f"/store/efficientdet4-{channels}"
