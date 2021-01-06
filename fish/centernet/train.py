from adabelief_pytorch import AdaBelief
import torch
from torch.utils.data import DataLoader
from object_detection.models.centernet import (
    collate_fn,
    CenterNet,
    Visualize,
    Trainer,
    Criterion,
    ToBoxes,
)
from object_detection.models.mkmaps import (
    MkGaussianMaps,
    MkCenterBoxMaps,
)
from object_detection.utils import DetectionPlot
from object_detection.models.backbones.effnet import (
    EfficientNetBackbone,
)
from object_detection.model_loader import (
    ModelLoader,
    BestWatcher,
)
from fish.data import (
    FileDataset,
    kfold,
    train_transforms,
    test_transforms,
    read_annotations,
)
from object_detection.metrics import MeanPrecition
from fish.centernet import config


backbone = EfficientNetBackbone(
    config.backbone_idx, out_channels=config.channels, pretrained=True
)
model = CenterNet(
    num_classes=2,
    channels=config.channels,
    backbone=backbone,
    out_idx=config.out_idx,
    depth=config.box_depth,
)
to_boxes = ToBoxes(threshold=config.to_boxes_threshold)
model_loader = ModelLoader(
    out_dir=config.out_dir,
    key=config.metric[0],
    best_watcher=BestWatcher(mode=config.metric[1]),
)


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
    criterion = Criterion(
        box_weight=config.box_weight,
        heatmap_weight=config.heatmap_weight,
        mk_hmmaps=MkGaussianMaps(
            num_classes=config.num_classes,
            sigma=config.sigma,
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
        shuffle=True,
    )
    optimizer = AdaBelief(
        model.parameters(),
        lr=config.lr,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decouple=False,
        rectify=True,
    )
    visualize = Visualize(config.out_dir, "test", limit=config.batch_size)

    get_score = MeanPrecition(iou_thresholds=[0.3])
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        model_loader=model_loader,
        optimizer=optimizer,
        visualize=visualize,
        criterion=criterion,
        device="cuda",
        get_score=get_score,
        to_boxes=to_boxes,
    )
    trainer(epochs)


if __name__ == "__main__":
    train(1000)
