from typing import List, Any, Tuple
from adabelief_pytorch import AdaBelief
import torch
from torch import nn
from typing import Dict
from toolz.curried import keyfilter, filter, pipe, map, valfilter
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
from object_detection.meters import MeanMeter
from object_detection.models.centernet import (
    Visualize,
    Criterion,
    ToBoxes,
)
from fish.store import StoreApi
from fish.models import FilterCenterNet
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
    LabeledDataset,
    FrameDataset,
    LabeledDataset,
    train_transforms,
    test_transforms,
    read_train_rows,
    read_test_rows,
    inv_normalize,
    ResizeMixDataset,
    NegativeDataset,
)
from fish.store import StoreApi
from fish.centernet.train import collate_fn
from object_detection.metrics import MeanPrecition
from fish.wfilter import config
from logging import (
    getLogger,
)

logger = getLogger(config.out_dir)


device = torch.device("cuda")
model = FilterCenterNet(
    backbone_id=3,
    num_classes=config.num_classes,
    channels=config.channels,
    out_idx=config.out_idx,
    cls_depth=config.cls_depth,
    box_depth=config.box_depth,
).to(device)
to_boxes = ToBoxes(
    threshold=config.to_boxes_threshold,
    iou_threshold=config.iou_threshold,
    kernel_size=config.to_boxes_kernel_size,
    limit=config.to_box_limit,
)
model_loader = ModelLoader(
    out_dir=config.out_dir,
    key=config.metric[0],
    best_watcher=BestWatcher(mode=config.metric[1]),
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


def train(epochs: int) -> None:
    annotations = read_train_rows("/store")
    test_annotations = read_test_rows("/store")
    api = StoreApi()
    fixed_rows = api.filter()
    fixed_keys = pipe(fixed_rows, map(lambda x: x["id"]), set)
    annotations = valfilter(
        lambda x: x["sequence_id"] not in config.ignore_seq_ids
        or x["id"] not in fixed_keys
    )(annotations)
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

    neg_rows = pipe(
        test_annotations.values(),
        filter(
            lambda x: x["sequence_id"] in config.negative_seq_ids and "test" in x["id"]
        ),
        list,
    )
    train_neg_rows = neg_rows
    test_neg_rows = neg_rows[: int(len(test_rows) // config.pos_neg)]
    train_dataset: Any = ConcatDataset(
        [
            FileDataset(
                rows=train_rows,
                transforms=train_transforms,
            ),
            NegativeDataset(rows=train_neg_rows, transforms=train_transforms),
            LabeledDataset(
                rows=train_fixed_rows,
                transforms=train_transforms,
            ),
            ResizeMixDataset(
                rows=train_rows,
                transforms=train_transforms,
            ),
        ]
    )
    test_dataset: Any = ConcatDataset(
        [
            FileDataset(
                rows=test_rows,
                transforms=test_transforms,
            ),
            NegativeDataset(
                rows=test_neg_rows,
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
        batch_size=config.batch_size * 2,
        num_workers=config.batch_size * 2,
        shuffle=True,
        drop_last=True,
    )
    optimizer = AdaBelief(
        model.parameters(),
        lr=config.lr,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decouple=False,
        rectify=True,
    )
    visualize = Visualize(
        config.out_dir, "test", limit=config.batch_size, transforms=inv_normalize
    )

    get_score = MeanPrecition(iou_thresholds=[0.3])
    logs: Dict[str, float] = {}
    scaler = GradScaler()
    mse = nn.MSELoss()

    def train_step() -> None:
        loss_meter = MeanMeter()
        label_loss_meter = MeanMeter()
        box_loss_meter = MeanMeter()
        weight_loss_meter = MeanMeter()
        for i, (
            ids,
            image_batch,
            gt_box_batch,
            gt_label_batch,
            gt_weight_batch,
        ) in tqdm(enumerate(train_loader)):
            model.train()
            image_batch = image_batch.to(device)
            gt_weight_batch = gt_weight_batch.to(device)
            gt_box_batch = [x.to(device) for x in gt_box_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            optimizer.zero_grad()
            with autocast(enabled=config.use_amp):
                netout, weight_batch = model(image_batch)
                loss, label_loss, box_loss, _ = criterion(
                    image_batch, netout, gt_box_batch, gt_label_batch
                )
                weight_loss = mse(weight_batch, gt_weight_batch)
                loss = loss + weight_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item())
            box_loss_meter.update(box_loss.item())
            label_loss_meter.update(label_loss.item())
            weight_loss_meter.update(weight_loss.item())
            logs["train_loss"] = loss_meter.get_value()
            logs["train_box"] = box_loss_meter.get_value()
            logs["train_label"] = label_loss_meter.get_value()
            logs["train_weight"] = weight_loss_meter.get_value()

            if i % 100 == 99:
                eval_step()
                log()

    @torch.no_grad()
    def eval_step() -> None:
        model.eval()
        loss_meter = MeanMeter()
        box_loss_meter = MeanMeter()
        label_loss_meter = MeanMeter()
        weight_loss_meter = MeanMeter()
        score_meter = MeanMeter()
        metrics = MeanAveragePrecision(
            iou_threshold=0.3, num_classes=config.num_classes
        )

        for ids, image_batch, gt_box_batch, gt_label_batch, gt_weight_batch in tqdm(
            test_loader
        ):
            image_batch = image_batch.to(device)
            gt_box_batch = [x.to(device) for x in gt_box_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            gt_weight_batch = gt_weight_batch.to(device)
            _, _, h, w = image_batch.shape
            netout, weight_batch = model(image_batch)
            loss, label_loss, box_loss, gt_hms = criterion(
                image_batch, netout, gt_box_batch, gt_label_batch
            )
            weight_loss = mse(weight_batch, gt_weight_batch)
            loss = loss + weight_loss
            loss_meter.update(loss.item())
            box_loss_meter.update(box_loss.item())
            label_loss_meter.update(label_loss.item())
            weight_loss_meter.update(weight_loss.item())
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
                score_meter.update(metrics()[0])
                metrics.reset()

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
        logs["test_loss"] = loss_meter.get_value()
        logs["test_box"] = box_loss_meter.get_value()
        logs["test_label"] = label_loss_meter.get_value()
        logs["test_weight"] = weight_loss_meter.get_value()
        logs["score"] = score_meter.get_value()
        print(ids)
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
    train(1000)
