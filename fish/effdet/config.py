from typing import *
from object_detection.model_loader import WatchMode
from fish.config import *


out_dir = "/store/efficientdet3"

# train
use_amp = True

# data
confidence_threshold = 0.2
iou_threshold = 0.20
batch_size = 4

# model
channels = 64
box_depth = 1
fpn_depth = 1
lr = 5e-4
out_ids: List[int] = [6, 7]

metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

# criterion
topk = 39
box_weight = 10
cls_weight = 1

anchor_ratios = [1.3, 1.9, 2.9]
anchor_scales = [1.0, 1.44]
anchor_size = 1
