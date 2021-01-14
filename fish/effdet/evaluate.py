import glob, tqdm, torch, json
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader
from object_detection.utils import DetectionPlot
from object_detection.metrics import MeanAveragePrecision
from object_detection.entities.box import (
    PascalBoxes,
    Confidences,
    Labels,
    resize,
)
from fish.data import read_train_rows, FileDataset, test_transforms, kfold
from fish.effdet import config
from fish.effdet.train import model, model_loader, to_boxes, collate_fn
from ensemble_boxes import weighted_boxes_fusion
from torchvision.transforms.functional import hflip, vflip

from logging import (
    getLogger,
)

logger = getLogger(config.out_dir)


@torch.no_grad()
def predict(device: str) -> None:
    annotations = read_train_rows("/store")
    out_dir = Path("/store/evaluate")
    out_dir.mkdir(exist_ok=True)
    dataset = FileDataset(
        rows=annotations, transforms=test_transforms(config.image_size)
    )
    net = model_loader.load_if_needed(model).to(device).eval()
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    weights = [1]
    metrics = MeanAveragePrecision(iou_threshold=0.3, num_classes=config.num_classes)
    for image_batch, gt_box_batch, gt_label_batch, ids in tqdm.tqdm(loader):
        image_batch = image_batch.to(device)
        _, _, h, w = image_batch.shape
        gt_box_batch = [x.to(device) for x in gt_box_batch]
        gt_label_batch = [x.to(device) for x in gt_label_batch]
        box_batch, confidence_batch, label_batch = to_boxes(net(image_batch))
        for id, img, boxes, gt_boxes, labels, gt_labels, confidences in zip(
            ids,
            image_batch,
            box_batch,
            gt_box_batch,
            label_batch,
            gt_label_batch,
            confidence_batch,
        ):
            metrics.add(
                boxes=boxes,
                confidences=confidences,
                labels=labels,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
            )
            plot = DetectionPlot(img)
            plot.draw_boxes(boxes=boxes, labels=labels, confidences=confidences)
            plot.draw_boxes(boxes=gt_boxes, labels=gt_labels, color="red")
            plot.save(out_dir.joinpath(f"{id}.jpg"))
    score, scores = metrics()
    logger.info(f"{score=}, {scores=}")


if __name__ == "__main__":
    predict("cuda")
