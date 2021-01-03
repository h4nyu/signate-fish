from typing import *
from object_detection.model_loader import WatchMode

out_dir = "store/0"

image_size = 1024
num_classes = 2
batch_size = 2

# model
backbone_idx = 1
channels = 128
out_idx = 5
box_depth = 2

# criterion
lr = 1e-3
box_weight = 5.0
heatmap_weight = 1.0
metric: Tuple[str, WatchMode] = ("score", "max")

# to_boxes
to_boxes_threshold = 0.3
