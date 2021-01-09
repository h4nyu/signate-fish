from typing import *
from object_detection.model_loader import WatchMode

out_dir = "store/centernet-0"

image_size = 1024 + 512
num_classes = 2
batch_size = 6

# model
backbone_idx = 1
channels = 64
out_idx = 6
cls_depth = 2

# criterion
lr = 1e-3
box_weight = 1.0
heatmap_weight = 1.0
sigma = 5
metric: Tuple[str, WatchMode] = ("score", "max")

# to_boxes
to_boxes_threshold = 0.3
iou_threshold = 0.4

# train
use_amp = True
