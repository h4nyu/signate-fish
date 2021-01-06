from typing import *
from object_detection.model_loader import WatchMode

out_dir = "store/1"

image_size = 1024 + 512
num_classes = 2
batch_size = 3

# model
backbone_idx = 3
channels = 64
out_idx = 4
box_depth = 1

# criterion
lr = 1e-3
box_weight = 1.0
heatmap_weight = 1.0
sigma = 20
metric: Tuple[str, WatchMode] = ("score", "max")

# to_boxes
to_boxes_threshold = 0.3
iou_threshold = 0.4
