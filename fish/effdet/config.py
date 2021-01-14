from typing import *
from object_detection.model_loader import WatchMode


out_dir = "/store/efficientdet-0"

# train
use_amp = True
n_splits = 6

# data
image_size = 1024 + 1024
num_classes = 2
original_width = 3840
original_height = 2160

confidence_threshold = 0.1
iou_threshold = 0.50
batch_size = 5

# model
channels = 64
box_depth = 1
fpn_depth = 2
lr = 5e-4
out_ids: List[int] = [6, 7]

metric: Tuple[str, WatchMode] = ("test_label", "min")
pretrained = True

# criterion
topk = 5
box_weight = 2
cls_weight = 1

anchor_ratios = [1.0]
anchor_scales = [1.0, 1.33, 1.66]
anchor_size = 8
