from typing import *
from object_detection.model_loader import WatchMode
from object_detection.models.mkmaps import GaussianMapMode

out_dir = "store/centernet-1"

image_size = 1024 + 512
original_width = 3840
original_height = 2160
num_classes = 2
batch_size = 4
mk_map_mode: GaussianMapMode = "length"

# model
backbone_idx = 1
channels = 128
out_idx = 6
cls_depth = 1
box_depth = 2

# criterion
lr = 1e-3
box_weight = 1.0
heatmap_weight = 1.0
sigma = 6
metric: Tuple[str, WatchMode] = ("score", "max")

# to_boxes
to_boxes_threshold = 0.3
iou_threshold = 0.5

# train
use_amp = True
