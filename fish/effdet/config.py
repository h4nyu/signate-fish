from typing import *
from object_detection.model_loader import WatchMode


out_dir = "/store/efficientdet"

# data
image_size = 1024
num_classes = 2
batch_size = 3

confidence_threshold = 0.3
iou_threshold = 0.50
batch_size = 8

# model
backbone_id = 1
channels = 64
box_depth = 2
lr = 1e-3
out_ids: List[int] = [4, 5, 6]

metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

# criterion
topk = 39
box_weight = 4
cls_weight = 1

anchor_ratios = [1.0]
anchor_scales = [1.0, 1.25, 1.5, 1.75]
anchor_size = 2
