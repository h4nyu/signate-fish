from typing import *
from object_detection.model_loader import WatchMode
from fish.config import *


pseudo_threshold = 0.05
pseudo_iou_threshold = 0.35

# train
use_amp = True

# data
confidence_threshold = 0.01
iou_threshold = 0.3
batch_size = 2

# model
backbone_id = 6
channels = 96
box_depth = 1
fpn_depth = 1
lr = 1e-4
out_ids: List[int] = [5, 6, 7]

metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True


# criterion
topk = 13
box_weight = 1
cls_weight = 1

anchor_ratios = [1.0]
anchor_scales = [1.0]
num_anchors = len(anchor_ratios) * len(anchor_scales)
anchor_size = 3

out_dir = f"/store/efficientdet-{backbone_id}-{num_anchors}-{channels}-{''.join([str(i) for i in out_ids])}"
