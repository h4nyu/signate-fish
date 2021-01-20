import torch
from fish.metrics import AveragePrecision
from object_detection.entities.box import PascalBoxes, Labels, Confidences


def test_metrics() -> None:
    metrics = AveragePrecision(
        iou_threshold=0.3
    )
    boxes = PascalBoxes(torch.tensor([
        [0, 0, 10, 10],
        [2, 2, 10, 10],
        [10, 10, 12, 13],
    ]))

    gt_boxes = PascalBoxes(torch.tensor([
        [0, 0, 10, 10],
        [20, 20, 30, 30],
    ]))
    confidences = Confidences(torch.tensor([0.4, 0.3, 0.2]))
    res = metrics(boxes=boxes, gt_boxes=gt_boxes, confidences=confidences)
    print(res)
