import torch
from fish.metrics import Metrics
from object_detection.entities import PascalBoxes, Labels, Confidences


def test_default() -> None:
    metrics = Metrics()
    boxes = PascalBoxes(
        torch.tensor(
            [
                [0, 0, 15, 15],
                [10, 10, 20, 20],
                [10, 10, 20, 20],
            ]
        )
    )
    labels = Labels(torch.tensor([0, 0, 0]))
    gt_boxes = PascalBoxes(torch.tensor([[0, 0, 10, 10], [5, 5, 10, 10]]))
    gt_labels = Labels(torch.tensor([0, 1]))
    score = metrics.add(
        boxes=boxes,
        labels=labels,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
    )
    print(score)
