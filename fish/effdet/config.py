from typing import *
from object_detection.model_loader import WatchMode
from fish.config import *


pseudo_threshold = 0.05
pseudo_iou_threshold = 0.35

# train
use_amp = True

# data
batch_size = 2

# model
backbone_id = 7
channels = 64
box_depth = 1
fpn_depth = 1
lr = 1e-4
out_ids: List[int] = [5, 6, 7]

metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

# criterion
topk = 50
box_weight = 1
cls_weight = 3
confidence_threshold = 0.0001
iou_threshold = 0.31
pre_box_limit = 10000

anchor_ratios = [1.0]
anchor_scales = [1.0]
num_anchors = len(anchor_ratios) * len(anchor_scales)
anchor_size = 3

out_dir = f"/store/efficientdet-{backbone_id}-{num_anchors}-{channels}-{''.join([str(i) for i in out_ids])}-{image_width}"
