import numpy as np, torch
from typing import *
from typing import Tuple, List, Dict
from torchvision.ops.boxes import box_iou
from object_detection.metrics.average_precision import auc
from object_detection.entities import PascalBoxes, Labels, Confidences
from fish import config


class Metrics:
    def __init__(
        self, iou_threshold: float = config.metrics_iou_threshold, eps: float = 1e-8
    ) -> None:
        self.iou_threshold = iou_threshold
        self.scores: List[float] = []

    def reset(self) -> None:
        self.scores = []

    @torch.no_grad()
    def add(
        self,
        boxes: PascalBoxes,
        labels: Labels,
        gt_boxes: PascalBoxes,
        gt_labels: Labels,
        confidences: Optional[Confidences] = None,
    ) -> float:
        unique_gt_labels = set(np.unique(gt_labels.to("cpu").numpy()))
        if len(unique_gt_labels) == 0:
            self.scores.append(0.0)
            return 0.0
        scores = np.zeros(len(unique_gt_labels))
        for i, k in enumerate(unique_gt_labels):
            c_boxes = boxes[labels == k]
            c_gt_boxes = gt_boxes[gt_labels == k]
            if(len(c_boxes)== 0 or len(gt_boxes) == 0):
                continue
            iou_m = box_iou(c_boxes, c_gt_boxes) > self.iou_threshold
            detected_indecies, matched_gt_box_indices = iou_m.max(dim=1)
            detected_boxes = c_boxes[detected_indecies]
            matched_gt_box_indices=matched_gt_box_indices[detected_indecies]
            tp = np.zeros(len(detected_boxes))
            matched:Set[int] = set()
            for box_id, gt_box_id in enumerate(matched_gt_box_indices.to("cpu").numpy()):
                if(gt_box_id not in matched):
                    tp[box_id] = 1
                    matched.add(gt_box_id)
            tpc = tp.cumsum()
            precision = tpc / np.arange(1, len(tp) + 1)
            div = np.ones(len(tp)) * min(len(detected_boxes), len(c_boxes))
            scores[i] = (precision / div).sum()
        score = scores.mean()
        self.scores.append(score)
        return score

    @torch.no_grad()
    def __call__(self) -> float:
        return np.array(self.scores, dtype=float).mean()
