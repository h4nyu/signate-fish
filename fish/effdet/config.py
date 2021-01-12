from typing import *
from object_detection.model_loader import WatchMode


out_dir = "/store/efficientdet-0"

# data
image_size = 1024 + 512
num_classes = 2
original_width = 3840
original_height = 2160

confidence_threshold = 0.3
iou_threshold = 0.50
batch_size = 3

# model
backbone_id = 1
channels = 96
box_depth = 1
lr = 1e-3
out_ids: List[int] = [6, 7]

metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

# criterion
topk = 39
box_weight = 2
cls_weight = 1

anchor_ratios = [1.0]
anchor_scales = [1.0, 1.5]
anchor_size = 4