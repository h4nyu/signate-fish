from typing import List, Any, Tuple
from adabelief_pytorch import AdaBelief
import torch
from typing import Dict
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from object_detection.meters import MeanMeter
from object_detection.models.centernet import (
    CenterNet,
    Visualize,
    Criterion,
    ToBoxes,
    collate_fn,
)
from fish.models import FrameCenterNet
from object_detection.models.mkmaps import (
    MkGaussianMaps,
    MkCenterBoxMaps,
)
from object_detection.utils import DetectionPlot
from object_detection.models.backbones.resnet import (
    ResNetBackbone,
)
from object_detection.model_loader import (
    ModelLoader,
    BestWatcher,
)
from object_detection.metrics import MeanAveragePrecision
from object_detection.entities.box import (
    yolo_to_pascal,
    pascal_to_yolo,
    PascalBoxes,
    Labels,
    YoloBoxes,
)
from object_detection.entities import ImageId, ImageBatch
from fish.data import (
    FileDataset,
    FrameDataset,
    kfold,
    train_transforms,
    test_transforms,
    read_train_rows,
    inv_normalize,
)
from object_detection.metrics import MeanPrecition
from fish.centernet import config
from logging import (
    getLogger,
)

logger = getLogger(config.out_dir)


device = torch.device("cuda")
backbone = ResNetBackbone("resnet50", out_channels=config.channels)
model = CenterNet(
    backbone=backbone,
    num_classes=config.num_classes,
    channels=config.channels,
    out_idx=config.out_idx,
    cls_depth=config.cls_depth,
    box_depth=config.box_depth,
).to(device)
to_boxes = ToBoxes(
    threshold=config.to_boxes_threshold, kernel_size=config.to_boxes_kernel_size
)
model_loader = ModelLoader(
    out_dir=config.out_dir,
    key=config.metric[0],
    best_watcher=BestWatcher(mode=config.metric[1]),
)


def train(epochs: int) -> None:
    annotations = read_train_rows("/store")
    train_rows, test_rows = kfold(annotations)
    train_dataset = FileDataset(
        rows=train_rows,
        transforms=train_transforms,
    )
    test_dataset = FileDataset(
        rows=test_rows,
        transforms=test_transforms,
    )
    criterion = Criterion(
        box_weight=config.box_weight,
        heatmap_weight=config.heatmap_weight,
        mk_hmmaps=MkGaussianMaps(
            num_classes=config.num_classes,
            sigma=config.sigma,
            mode=config.mk_map_mode,
        ),
        mk_boxmaps=MkCenterBoxMaps(),
    )
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=config.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size * 2,
        num_workers=config.batch_size,
        shuffle=False,
    )
    optimizer = AdaBelief(
        model.parameters(),
        lr=config.lr,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decouple=False,
        rectify=True,
    )
    visualize = Visualize(config.out_dir, "test", limit=config.batch_size, transforms=inv_normalize)

    get_score = MeanPrecition(iou_thresholds=[0.3])
    logs: Dict[str, float] = {}
    scaler = GradScaler()

    def train_step() -> None:
        model.train()
        loss_meter = MeanMeter()
        label_loss_meter = MeanMeter()
        box_loss_meter = MeanMeter()
        for ids, image_batch, gt_box_batch, gt_label_batch in tqdm(train_loader):
            image_batch = image_batch.to(device)
            gt_box_batch = [x.to(device) for x in gt_box_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            optimizer.zero_grad()
            with autocast(enabled=config.use_amp):
                netout = model(image_batch)
                loss, label_loss, box_loss, _ = criterion(
                    image_batch, netout, gt_box_batch, gt_label_batch
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item())
            box_loss_meter.update(box_loss.item())
            label_loss_meter.update(label_loss.item())
        logs["train_loss"] = loss_meter.get_value()
        logs["train_box"] = box_loss_meter.get_value()
        logs["train_label"] = label_loss_meter.get_value()

    @torch.no_grad()
    def eval_step() -> None:
        model.eval()
        loss_meter = MeanMeter()
        box_loss_meter = MeanMeter()
        label_loss_meter = MeanMeter()
        metrics = MeanAveragePrecision(
            iou_threshold=0.3, num_classes=config.num_classes
        )

        for ids, image_batch, gt_box_batch, gt_label_batch in tqdm(test_loader):
            image_batch = image_batch.to(device)
            gt_box_batch = [x.to(device) for x in gt_box_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            _, _, h, w = image_batch.shape
            netout = model(image_batch)
            loss, label_loss, box_loss, gt_hms = criterion(
                image_batch, netout, gt_box_batch, gt_label_batch
            )
            loss_meter.update(loss.item())
            box_loss_meter.update(box_loss.item())
            label_loss_meter.update(label_loss.item())
            box_batch, confidence_batch, label_batch = to_boxes(netout)
            for boxes, gt_boxes, labels, gt_labels, confidences in zip(
                box_batch, gt_box_batch, label_batch, gt_label_batch, confidence_batch
            ):
                metrics.add(
                    boxes=yolo_to_pascal(boxes, (w, h)),
                    confidences=confidences,
                    labels=labels,
                    gt_boxes=yolo_to_pascal(gt_boxes, (w, h)),
                    gt_labels=gt_labels,
                )

        visualize(
            netout,
            box_batch,
            confidence_batch,
            label_batch,
            gt_box_batch,
            gt_label_batch,
            image_batch,
            gt_hms,
        )
        score, scores = metrics()
        logs["test_loss"] = loss_meter.get_value()
        logs["test_box"] = box_loss_meter.get_value()
        logs["test_label"] = label_loss_meter.get_value()
        logs["score"] = score
        for k, v in scores.items():
            logs[f"score-{k}"] = v
        model_loader.save_if_needed(
            model,
            score,
        )

    def log() -> None:
        logger.info(",".join([f"{k}={v:.3f}" for k, v in logs.items()]))

    model_loader.load_if_needed(model)
    for _ in range(epochs):
        train_step()
        eval_step()
        log()


if __name__ == "__main__":
    train(1000)
