import numpy as np, torch
from typing import Tuple, List, Dict
from object_detection.metrics.average_precision import AveragePrecision
from object_detection.entities import PascalBoxes, Labels, Confidences


class Metrics:
    def __init__(self, iou_threshold: float, eps: float = 1e-8) -> None:
        self.iou_threshold = iou_threshold
        self.scores: List[float] = []

    def reset(self) -> None:
        self.scores = []

    @torch.no_grad()
    def add(
        self,
        boxes: PascalBoxes,
        confidences: Confidences,
        labels: Labels,
        gt_boxes: PascalBoxes,
        gt_labels: Labels,
    ) -> None:
        unique_gt_labels = set(np.unique(gt_labels.to("cpu").numpy()))
        unique_labels = set(np.unique(labels.to("cpu").numpy()))
        if len(unique_gt_labels) == len(unique_labels) == 0:
            self.scores.append(1.0)
            return

        category_scores = []
        for k in unique_gt_labels | unique_labels:
            ap = AveragePrecision(self.iou_threshold)
            ap.add(
                boxes=PascalBoxes(boxes[labels == k]),
                confidences=Confidences(confidences[labels == k]),
                gt_boxes=PascalBoxes(gt_boxes[gt_labels == k]),
            )
            category_scores.append(ap())
        self.scores.append(np.array(category_scores).mean())
        return

    @torch.no_grad()
    def __call__(self) -> float:
        return np.array(self.scores, dtype=float).mean()
