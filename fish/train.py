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
from object_detection.models.backbones.effnet import (
    EfficientNetBackbone,
)
from object_detection.model_loader import (
    ModelLoader,
    BestWatcher,
)
from fish.data import FileDataset, kfold
from object_detection.metrics import MeanPrecition
from fish.store import ImageStore
from fish import config


def train(epochs: int) -> None:
    store = ImageStore("/store")
    annotations = store.read()
    train_rows, test_rows = kfold(annotations)
    train_dataset = FileDataset(
        rows=train_rows,
    )
    test_dataset = FileDataset(
        rows=test_rows,
    )
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
    criterion = Criterion(
        box_weight=config.box_weight,
        heatmap_weight=config.heatmap_weight,
        mk_hmmaps=MkGaussianMaps(num_classes=config.num_classes),
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    visualize = Visualize(config.out_dir, "test", limit=2)

    model_loader = ModelLoader(
        out_dir=config.out_dir,
        key=config.metric[0],
        best_watcher=BestWatcher(mode=config.metric[1]),
    )
    to_boxes = ToBoxes(threshold=config.to_boxes_threshold)
    get_score = MeanPrecition()
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
