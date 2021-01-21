from typing import *
import numpy as np, torch
from typing import Tuple, List, Dict
from object_detection.entities import PascalBoxes, Labels, Confidences
from torchvision.ops.boxes import box_iou

class AveragePrecision:
    def __init__(
        self,
        iou_threshold: float,
        eps: float = 1e-8,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.eps = eps
        self.tp_list: List[Any] = []
        self.confidence_list: List[Any] = []
        self.n_gt_box = 0

    def reset(self) -> None:
        ...
        # self.n_gt_box = 0
        # self.confidence_list = []
        # self.tp_list = []

    def __call__(
        self,
        boxes: PascalBoxes,
        confidences: Confidences,
        gt_boxes: PascalBoxes,
    ) -> float:
        n_gt_box = len(gt_boxes)
        n_box = len(boxes)
        if n_gt_box == 0:
            if n_box == 0:
                return 1.0
            else:
                return 0.0
        tp = np.zeros((n_box, ))
        sort_indices = confidences.argsort(descending=True)
        iou_matrix = box_iou(boxes[sort_indices], gt_boxes)
        ious, matched_indices = torch.max(iou_matrix, dim=1)
        matched: Set = set()
        n_correct = (ious > self.iou_threshold).sum().item()
        print(n_correct)
        for box_id, gt_id in enumerate(matched_indices.to("cpu").numpy()):
            if ious[box_id] > self.iou_threshold and gt_id not in matched:
                tp[box_id] = 1
                matched.add(gt_id)

        tpc = tp.cumsum()
        fpc = (1 - tp).cumsum()
        count = (tpc + fpc)
        precision = tpc / count
        div = np.min(np.stack([np.ones(n_box) * n_correct, count]), axis=0)
        print(precision)
        print(div)
        return np.sum(precision / div)

# class MeanAveragePrecision:
#     def __init__(
#         self, num_classes: int, iou_threshold: float, eps: float = 1e-8
#     ) -> None:
#         self.ap = AveragePrecision(iou_threshold, eps)
#         self.aps = {k: AveragePrecision(iou_threshold, eps) for k in range(num_classes)}
#         self.eps = eps

#     def reset(self) -> None:
#         for v in self.aps.values():
#             v.reset()

#     @torch.no_grad()
#     def add(
#         self,
#         boxes: PascalBoxes,
#         confidences: Confidences,
#         labels: Labels,
#         gt_boxes: PascalBoxes,
#         gt_labels: Labels,
#     ) -> None:
#         for k in np.unique(gt_labels.to("cpu").numpy()):
#             ap = self.aps[k]
#             ap.add(
#                 boxes=PascalBoxes(boxes[labels == k]),
#                 confidences=Confidences(confidences[labels == k]),
#                 gt_boxes=PascalBoxes(gt_boxes[gt_labels == k]),
#             )

#     @torch.no_grad()
#     def __call__(self) -> Tuple[float, Dict[int, float]]:
#         aps = {k: v() for k, v in self.aps.items()}
#         return np.fromiter(aps.values(), dtype=float).mean(), aps
