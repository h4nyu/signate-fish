from typing import *
from object_detection.model_loader import WatchMode
from fish.config import *


pseudo_threshold = 0.10

# train
use_amp = True

# data
confidence_threshold = 0.001
iou_threshold = 0.45
batch_size = 4

# model
channels = 32
box_depth = 1
fpn_depth = 1
lr = 1e-4
out_ids: List[int] = [6, 7]

metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

# criterion
topk = 50
box_weight = 1
cls_weight = 1

anchor_ratios = [1.0]
anchor_scales = [1.0, 1.44]
num_anchors = len(anchor_ratios) * len(anchor_scales)
anchor_size = 4

out_dir = f"/store/efficientdet-{num_anchors}-{channels}"
