from typing import *
from fish.config import *
from object_detection.model_loader import WatchMode
from object_detection.models.mkmaps import GaussianMapMode

out_dir = "store/centernet3"

batch_size = 8
mk_map_mode: GaussianMapMode = "length"

# model
channels = 32
out_idx = 5
cls_depth = 1
box_depth = 1

# criterion
lr = 4e-4
box_weight = 2.0
heatmap_weight = 1.0
sigma = 15
metric: Tuple[str, WatchMode] = ("score", "max")

# to_boxes
to_boxes_threshold = 0.03
iou_threshold = 0.2
to_boxes_kernel_size = 7

pseudo_threshold = 0.15


# train
use_amp = True
