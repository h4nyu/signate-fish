from typing import *
from object_detection.model_loader import WatchMode
from object_detection.models.mkmaps import GaussianMapMode

out_dir = "store/centernet-2"

original_width = 3840
original_height = 2160
num_classes = 2
batch_size = 3
mk_map_mode: GaussianMapMode = "aspect"

# model
channels = 64
out_idx = 6
cls_depth = 1
box_depth = 1

# criterion
lr = 1e-3
box_weight = 2.0
heatmap_weight = 1.0
sigma = 15
metric: Tuple[str, WatchMode] = ("score", "max")

# to_boxes
to_boxes_threshold = 0.20
iou_threshold = 0.2
to_boxes_kernel_size = 3


# train
use_amp = True
