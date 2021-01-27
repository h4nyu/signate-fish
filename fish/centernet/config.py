from typing import *
from fish.config import *
from object_detection.model_loader import WatchMode
from object_detection.models.mkmaps import GaussianMapMode


batch_size = 4
mk_map_mode: GaussianMapMode = "length"

# model
channels = 64
out_idx = 7
cls_depth = 1
box_depth = 1

# criterion
lr = 1e-3
box_weight = 1.5
heatmap_weight = 1.0
sigma = 15
metric: Tuple[str, WatchMode] = ("score", "max")

# to_boxes
to_boxes_threshold = 0.1
iou_threshold = 0.1
to_boxes_kernel_size = 3

pseudo_threshold = 0.15

# train
use_amp = True

out_dir = f"store/centernet-{channels}"
