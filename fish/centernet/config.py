from typing import *
from fish.config import *
from object_detection.model_loader import WatchMode
from object_detection.models.mkmaps import GaussianMapMode


batch_size = 2
mk_map_mode: GaussianMapMode = "length"

# model
backbone_id = 5
channels = 128
out_idx = 6
cls_depth = 1
box_depth = 1

# criterion
lr = 5e-4
box_weight = 1.5
heatmap_weight = 1.0
sigma = 20
metric: Tuple[str, WatchMode] = ("score", "max")

# to_boxes
to_boxes_threshold = 0.001
iou_threshold = 0.45
to_boxes_kernel_size = 3

pseudo_threshold = 0.2

# train
use_amp = True

out_dir = f"store/centernet-{backbone_id}-{channels}"
