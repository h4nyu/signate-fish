import glob, tqdm, torch, json, shutil
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader
from object_detection.utils import DetectionPlot
from fish.metrics import Metrics
from object_detection.entities.box import (
    yolo_to_pascal,
    yolo_vflip,
    PascalBoxes,
    Confidences,
    Labels,
    resize,
    filter_size,
    resize,
)
from fish.data import (
    read_train_rows,
    FileDataset,
    test_transforms,
    kfold,
    inv_normalize,
)
from object_detection.entities.box import (
    PascalBoxes,
    Confidences,
    Labels,
    resize,
    box_hflip,
)
from fish.effdet import config
from fish.effdet.train import model, model_loader, to_boxes, collate_fn
from ensemble_boxes import weighted_boxes_fusion
from torchvision.transforms.functional import hflip, vflip

from logging import (
    getLogger,
)

logger = getLogger(__name__)


@torch.no_grad()
def predict(device: str) -> None:
    rows = read_train_rows("/store")
    out_dir = Path(config.out_dir).joinpath("evaluate")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)
    dataset = FileDataset(rows=rows, transforms=test_transforms)
    net = model_loader.load_if_needed(model).to(device).eval()
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size * 4,
        num_workers=config.batch_size * 4,
        shuffle=False,
        drop_last=False,
    )
    weights = [1]
    metrics = Metrics(iou_threshold=config.metrics_iou_threshold)
    for image_batch, gt_box_batch, gt_label_batch, ids, _ in tqdm.tqdm(loader):
        image_batch = image_batch.to(device)
        _, _, h, w = image_batch.shape
        gt_box_batch = [x.to(device) for x in gt_box_batch]
        gt_label_batch = [x.to(device) for x in gt_label_batch]
        box_batch, confidence_batch, label_batch = to_boxes(net(image_batch))
        for (
            id,
            img,
            boxes,
            confidences,
            labels,
            gt_boxes,
            gt_labels,
        ) in zip(
            ids,
            image_batch,
            box_batch,
            confidence_batch,
            label_batch,
            gt_box_batch,
            gt_label_batch,
        ):

            metrics.add(
                boxes=boxes,
                confidences=confidences,
                labels=labels,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
            )
        print(metrics())

    score = metrics()
    logger.info(f"{score=}")


if __name__ == "__main__":
    predict("cuda")
