from adabelief_pytorch import AdaBelief
from tqdm import tqdm
from typing import *
import torch
from torch import Tensor
from toolz.curried import keyfilter, filter, pipe, map, valfilter, take
from typing import Dict, Any
from torch.utils.data import DataLoader, ConcatDataset
from object_detection.models.backbones.effnet import (
    EfficientNetBackbone,
)
from object_detection.entities import ImageId, ImageBatch
from object_detection.entities.box import (
    yolo_to_pascal,
    pascal_to_yolo,
    PascalBoxes,
    Labels,
    YoloBoxes,
)
import torch_optimizer as optim
from torch.cuda.amp import GradScaler, autocast
from object_detection.meters import MeanMeter
from object_detection.metrics import MeanAveragePrecision
from object_detection.models.effidet import (
    EfficientDet,
    Criterion,
    Visualize,
    ToBoxes,
    Anchors,
)
from object_detection.model_loader import (
    ModelLoader,
    BestWatcher,
)
from fish.metrics import Metrics
from object_detection.metrics import MeanPrecition
from fish.data import (
    FileDataset,
    LabeledDataset,
    kfold,
    train_transforms,
    test_transforms,
    read_train_rows,
    read_test_rows,
    inv_normalize,
    ResizeMixDataset,
    filter_limit,
    sort_by_size,
)
from fish.store import StoreApi
from fish.effdet import config
from logging import (
    getLogger,
)

logger = getLogger(config.out_dir)


device = torch.device("cuda")
anchors = Anchors(
    size=config.anchor_size,
    ratios=config.anchor_ratios,
    scales=config.anchor_scales,
)
backbone = EfficientNetBackbone(
    config.backbone_id, out_channels=config.channels, pretrained=True
)
model = EfficientDet(
    num_classes=config.num_classes,
    out_ids=config.out_ids,
    channels=config.channels,
    backbone=backbone,
    anchors=anchors,
    box_depth=config.box_depth,
    fpn_depth=config.fpn_depth,
).to(device)
model_loader = ModelLoader(
    out_dir=config.out_dir,
    key=config.metric[0],
    best_watcher=BestWatcher(mode=config.metric[1]),
)

to_boxes = ToBoxes(
    confidence_threshold=config.confidence_threshold,
    iou_threshold=config.iou_threshold,
    limit=config.pre_box_limit,
)

criterion = Criterion(
    topk=config.topk,
    box_weight=config.box_weight,
    cls_weight=config.cls_weight,
)


def collate_fn(
    batch: List[Any],
) -> Tuple[ImageBatch, List[PascalBoxes], List[Labels], List[ImageId], Tensor]:
    images: List[Any] = []
    id_batch: List[ImageId] = []
    box_batch: List[PascalBoxes] = []
    label_batch: List[Labels] = []
    weight_batch: List[Tensor] = []
    for id, img, boxes, labels, ws in batch:
        c, h, w = img.shape
        images.append(img)
        box_batch.append(boxes)
        id_batch.append(id)
        label_batch.append(labels)
        weight_batch.append(ws)
    return (
        ImageBatch(torch.stack(images)),
        box_batch,
        label_batch,
        id_batch,
        torch.stack(weight_batch),
    )


def train(epochs: int) -> None:
    annotations = read_train_rows("/store")
    test_annotations = read_test_rows("/store")
    api = StoreApi()
    fixed_rows = api.filter()
    fixed_rows = pipe(fixed_rows, filter(lambda x: len(x["boxes"]) > 0), list)
    fixed_keys = pipe(fixed_rows, map(lambda x: x["id"]), set)
    annotations = valfilter(lambda x: len(x["boxes"]) > 0 or x["id"] not in fixed_keys)(
        annotations
    )
    train_rows = valfilter(lambda x: x["sequence_id"] not in config.test_seq_ids)(
        annotations
    )
    test_rows = valfilter(lambda x: x["sequence_id"] in config.test_seq_ids)(
        annotations
    )
    test_keys = set(test_rows.keys())
    train_keys = set(train_rows.keys())
    train_fixed_rows = pipe(
        fixed_rows, filter(lambda x: x["id"] not in test_keys), list
    )
    test_fixed_rows = pipe(
        fixed_rows, filter(lambda x: x["id"] not in train_keys), list
    )
    fixed_keys = set(x["id"] for x in fixed_rows)
    train_rows = keyfilter(lambda x: x not in fixed_keys, train_rows)
    test_rows = keyfilter(lambda x: x not in fixed_keys, test_rows)

    train_dataset: Any = ConcatDataset(
        [
            FileDataset(
                rows=annotations,
                transforms=train_transforms,
            ),
            LabeledDataset(
                rows=fixed_rows,
                transforms=train_transforms,
            ),
            # ResizeMixDataset(
            #     rows=annotations,
            #     transforms=train_transforms,
            # ),
        ]
    )
    test_dataset: Any = ConcatDataset(
        [
            LabeledDataset(
                rows=fixed_rows,
                transforms=test_transforms,
            ),
            FileDataset(
                rows=test_rows,
                transforms=test_transforms,
            ),
        ]
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
        batch_size=config.batch_size * 3,
        num_workers=config.batch_size * 3,
        shuffle=True,
        drop_last=True,
    )
    optimizer = optim.RAdam(
        model.parameters(),
        lr=config.lr,
        eps=1e-8,
        betas=(0.9, 0.999),
    )
    visualize = Visualize(
        config.out_dir, "test", limit=config.batch_size * 2, transforms=inv_normalize
    )
    scaler = GradScaler()
    logs: Dict[str, float] = {}

    def train_step() -> None:
        loss_meter = MeanMeter()
        box_loss_meter = MeanMeter()
        label_loss_meter = MeanMeter()
        for i, (
            image_batch,
            gt_box_batch,
            gt_label_batch,
            _,
            _,
        ) in tqdm(enumerate(train_loader)):
            model.train()
            image_batch = image_batch.to(device)
            gt_box_batch = [x.to(device) for x in gt_box_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            optimizer.zero_grad()
            with autocast(enabled=config.use_amp):
                netout = model(image_batch)
                loss, box_loss, label_loss = criterion(
                    image_batch,
                    netout,
                    gt_box_batch,
                    gt_label_batch,
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
            if i % 100 == 99:
                eval_step()
                log()

    @torch.no_grad()
    def eval_step() -> None:
        model.eval()
        loss_meter = MeanMeter()
        box_loss_meter = MeanMeter()
        label_loss_meter = MeanMeter()
        metrics = Metrics(iou_threshold=0.3)
        for image_batch, gt_box_batch, gt_label_batch, ids, _ in tqdm(test_loader):
            image_batch = image_batch.to(device)
            gt_box_batch = [x.to(device) for x in gt_box_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            netout = model(image_batch)
            loss, box_loss, label_loss = criterion(
                image_batch, netout, gt_box_batch, gt_label_batch
            )
            box_batch, confidence_batch, label_batch = to_boxes(netout)

            loss_meter.update(loss.item())
            box_loss_meter.update(box_loss.item())
            label_loss_meter.update(label_loss.item())

            for boxes, gt_boxes, labels, gt_labels, confidences in zip(
                box_batch, gt_box_batch, label_batch, gt_label_batch, confidence_batch
            ):
                boxes, labels, confidences = filter_limit(boxes, labels, confidences)
                metrics.add(
                    boxes=boxes,
                    confidences=confidences,
                    labels=labels,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                )

        logs["test_loss"] = loss_meter.get_value()
        logs["test_box"] = box_loss_meter.get_value()
        logs["test_label"] = label_loss_meter.get_value()
        logs["score"] = metrics()
        print(ids)
        visualize(
            image_batch,
            (box_batch, confidence_batch, label_batch),
            (gt_box_batch, gt_label_batch),
        )
        model_loader.save_if_needed(
            model,
            logs[model_loader.key],
        )

    def log() -> None:
        logger.info(",".join([f"{k}={v:.3f}" for k, v in logs.items()]))

    model_loader.load_if_needed(model)
    for _ in range(epochs):
        train_step()


if __name__ == "__main__":
    train(10000)
