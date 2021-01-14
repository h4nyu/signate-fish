from typing import *
from object_detection.model_loader import WatchMode


out_dir = "/store/efficientdet1"

# train
use_amp = True
n_splits = 6

# data
num_classes = 2
original_width = 3840
original_height = 2160

confidence_threshold = 0.1
iou_threshold = 0.50
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
topk = 9
box_weight = 1
cls_weight = 1

anchor_ratios = [1.0]
anchor_scales = [1.0, 1.33, 1.66]
anchor_size = 8
