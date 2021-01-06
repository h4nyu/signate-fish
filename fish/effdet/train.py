from adabelief_pytorch import AdaBelief
import torch
from torch.utils.data import DataLoader
from object_detection.models.backbones.effnet import (
    EfficientNetBackbone,
)
from object_detection.models.effidet import (
    collate_fn,
    EfficientDet,
    Trainer,
    Criterion,
    Visualize,
    ToBoxes,
    Anchors,
)
from object_detection.model_loader import (
    ModelLoader,
    BestWatcher,
)
from object_detection.metrics import MeanPrecition
from fish.data import (
    FileDataset,
    kfold,
    train_transforms,
    test_transforms,
    read_annotations,
)
from fish.effdet import config


def train(epochs: int) -> None:
    annotations = read_annotations("/store")
    train_rows, test_rows = kfold(annotations)
    train_dataset = FileDataset(
        rows=train_rows,
        transforms=train_transforms(config.image_size),
    )
    test_dataset = FileDataset(
        rows=test_rows,
        transforms=test_transforms(config.image_size),
    )
    backbone = EfficientNetBackbone(
        config.backbone_id,
        out_channels=config.channels,
        pretrained=True,
    )
    anchors = Anchors(
        size=config.anchor_size,
        ratios=config.anchor_ratios,
        scales=config.anchor_scales,
    )
    model = EfficientDet(
        num_classes=config.num_classes,
        out_ids=config.out_ids,
        channels=config.channels,
        backbone=backbone,
        anchors=anchors,
        box_depth=config.box_depth,
    )
    model_loader = ModelLoader(
        out_dir=config.out_dir,
        key=config.metric[0],
        best_watcher=BestWatcher(mode=config.metric[1]),
    )
    criterion = Criterion(
        topk=config.topk,
        box_weight=config.box_weight,
        cls_weight=config.cls_weight,
    )
    optimizer = AdaBelief(
        model.parameters(),
        lr=config.lr,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decouple=False,
        rectify=True,
    )

    visualize = Visualize("/store/efficientdet", "test", limit=config.batch_size)
    get_score = MeanPrecition(iou_thresholds=[0.3])
    to_boxes = ToBoxes(
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
    )
    trainer = Trainer(
        model,
        DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            batch_size=config.batch_size,
            num_workers=config.batch_size,
            shuffle=True,
        ),
        DataLoader(
            test_dataset,
            collate_fn=collate_fn,
            batch_size=config.batch_size * 2,
            num_workers=config.batch_size,
            shuffle=True,
        ),
        model_loader=model_loader,
        optimizer=optimizer,
        visualize=visualize,
        criterion=criterion,
        get_score=get_score,
        device="cuda",
        to_boxes=to_boxes,
    )
    trainer(epochs)


if __name__ == "__main__":
    train(10000)
