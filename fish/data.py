import torch, typing, numpy as np, cv2
from typing import *
from torch import Tensor
from torch.utils.data import Dataset
from skimage.io import imread
from toolz.curried import map, pipe, keyfilter, valfilter, filter, sorted
from io import BytesIO
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.transforms import Normalize
from PIL import Image as PILImage
import os
from object_detection.entities import (
    TrainSample,
    PredictionSample,
    ImageId,
    Image,
    PascalBoxes,
    Labels,
)
from object_detection.entities.box import filter_size, resize
import albumentations as albm
from sklearn.model_selection import GroupKFold, StratifiedKFold
import glob, typing, json, re
from pathlib import Path
from fish import config
from fish.store import StoreApi, Rows


Annotation = typing.TypedDict(
    "Annotation",
    {
        "boxes": typing.List[typing.List[int]],
        "labels": typing.List[int],
        "image_path": str,
        "sequence_id": int,
        "frame_id": int,
    },
)
Annotations = typing.Dict[str, Annotation]

TestRow = typing.TypedDict(
    "TestRow",
    {
        "image_path": str,
        "sequence_id": int,
        "frame_id": int,
    },
)
TestRows = typing.Dict[str, TestRow]


Submission = Dict[str, Any]


def add_submission(
    submission: Submission, id: str, boxes: PascalBoxes, labels: Labels
) -> None:
    row = {}
    row["Jumper School"] = boxes[labels == 0].to("cpu").tolist()
    row["Breezer School"] = boxes[labels == 1].to("cpu").tolist()
    submission[f"{id}.jpg"] = row


def cutmix(rows: Annotations) -> None:
    ...


def parse_label(value: str) -> typing.Optional[int]:
    if value == "Jumper School":
        return 0
    if value == "Breezer School":
        return 1
    return None


def read_train_rows(dataset_dir: str) -> Annotations:
    annotations: Annotations = {}
    _dataset_dir = Path(dataset_dir)
    annotation_dir = _dataset_dir.joinpath("train_annotations")
    image_dir = _dataset_dir.joinpath("train_images")
    for p in glob.glob(f"{annotation_dir}/*.json"):
        path = Path(p)
        id = path.stem
        boxes: typing.List[typing.List[int]] = []
        labels: typing.List[int] = []
        with path.open("r") as f:
            row = json.load(f)
        sequence_id = row["attributes"]["sequence_id"]
        frame_id = row["attributes"]["frame_id"]
        for k, v in row["labels"].items():
            label = parse_label(k)
            if label is None:
                continue
            labels += [label] * len(v)
            boxes += v

        image_path = str(image_dir.joinpath(f"{id}.jpg"))
        annotations[id] = dict(
            boxes=boxes,
            labels=labels,
            image_path=image_path,
            sequence_id=sequence_id,
            frame_id=frame_id,
        )
    return annotations


def read_test_rows(dataset_dir: str) -> TestRows:
    rows: TestRows = {}
    _dataset_dir = Path(dataset_dir)
    annotation_dir = _dataset_dir.joinpath("test_annotations")
    image_dir = _dataset_dir.joinpath("test_images")
    for p in glob.glob(f"{annotation_dir}/*.json"):
        path = Path(p)
        id = path.stem
        with path.open("r") as f:
            row = json.load(f)
        sequence_id = row["attributes"]["sequence_id"]
        frame_id = row["attributes"]["frame_id"]
        image_path = str(image_dir.joinpath(id if "jpg" in id else f"{id}.jpg"))
        rows[id] = dict(
            image_path=image_path,
            sequence_id=sequence_id,
            frame_id=frame_id,
        )
    return rows


def kfold(
    rows: Annotations, n_splits: int = config.n_splits
) -> typing.Tuple[Annotations, Annotations]:
    kf = StratifiedKFold(n_splits)
    x = list(rows.keys())
    y = [len(i["boxes"]) for i in rows.values()]
    train_ids, test_ids = next(kf.split(x, y))
    train_keys = set([x[i] for i in train_ids])
    test_keys = set([x[i] for i in test_ids])
    return keyfilter(lambda k: k in train_keys, rows), keyfilter(
        lambda k: k in test_keys, rows
    )


bbox_params = {"format": "pascal_voc", "label_fields": ["labels"]}
test_transforms = albm.Compose(
    [
        albm.LongestMaxSize(max_size=config.image_size),
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)

prediction_transforms = albm.Compose(
    [
        albm.LongestMaxSize(max_size=config.image_size),
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ToTensorV2(),
    ],
)
inv_normalize = Normalize(
    mean=[-m / s for m, s in zip(config.normalize_mean, config.normalize_std)],
    std=[1 / s for s in config.normalize_std],
)

train_transforms = albm.Compose(
    [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=(-0.2, 0.2),
            rotate_limit=10,
            p=1.0,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.OneOf(
            [
                A.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.3
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.9
                ),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=1.0),
                A.Blur(blur_limit=5, p=1.0),
                A.Blur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ],
            p=0.3,
        ),
        A.OneOf(
            [
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ],
            p=0.2,
        ),
        A.LongestMaxSize(max_size=config.image_size),
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)


def filter_boxes(
    boxes: PascalBoxes, min_width: int, min_height: int, min_area: int
) -> Tuple[PascalBoxes, Tensor]:
    if len(boxes) == 0:
        return boxes, torch.zeros(0, dtype=torch.bool)
    x0, y0, x1, y1 = boxes.unbind(-1)
    w = x1 - x0
    h = y1 - y0
    area = w * h
    filter_indices = (w > min_width) & (h > min_height) & (area > min_area)
    return PascalBoxes(boxes[filter_indices]), filter_indices


class FileDataset(Dataset):
    def __init__(
        self,
        rows: Annotations,
        transforms: typing.Any,
    ) -> None:
        self.rows = list(rows.items())
        self.transforms = transforms

    def __getitem__(self, idx: int) -> TrainSample:
        id, annot = self.rows[idx]
        image = imread(annot["image_path"])
        boxes = annot["boxes"]
        labels = annot["labels"]
        transed = self.transforms(image=image, bboxes=boxes, labels=labels)
        return (
            ImageId(id),
            Image(transed["image"]),
            PascalBoxes(torch.tensor(transed["bboxes"])),
            Labels(torch.tensor(transed["labels"])),
        )

    def __len__(self) -> int:
        return len(self.rows)


class LabeledDataset(Dataset):
    def __init__(
        self, rows: Rows, transforms: typing.Any, image_dir: str = "/store/images"
    ) -> None:
        self.rows = rows
        self.transforms = transforms
        self.image_dir = Path(image_dir)

    def __getitem__(self, idx: int) -> TrainSample:
        row = self.rows[idx]
        id = row["id"]
        path = self.image_dir.joinpath(f"{id}.jpg")
        image = imread(path)
        h, w, _ = image.shape
        boxes = PascalBoxes(
            torch.tensor(
                [
                    [
                        b["x0"],
                        b["y0"],
                        b["x1"],
                        b["y1"],
                    ]
                    for b in row["boxes"]
                ]
            )
        )
        boxes = resize(boxes, (w, h))
        labels = [int(float(b["label"])) for b in row["boxes"]]
        transed = self.transforms(image=image, bboxes=boxes, labels=labels)
        return (
            ImageId(id),
            Image(transed["image"]),
            PascalBoxes(torch.tensor(transed["bboxes"])),
            Labels(torch.tensor(transed["labels"])),
        )

    def __len__(self) -> int:
        return len(self.rows)


def find_prev_frame(
    rows: Annotations, sequence_id: int, frame_id: int
) -> Optional[Annotation]:
    return pipe(
        rows.values(),
        filter(lambda x: x["sequence_id"] == sequence_id and x["frame_id"] < frame_id),
        sorted(key=lambda x: x["frame_id"], reverse=True),
        iter,
        lambda x: next(x, None),
    )


class FrameDataset(Dataset):
    def __init__(
        self,
        rows: Annotations,
        transforms: typing.Any,
    ) -> None:
        self.rows = rows
        self.keys = list(rows.keys())
        self.transforms = A.Compose(
            [transforms],
            additional_targets={
                "image0": "image",
            },
        )

    def __getitem__(
        self, idx: int
    ) -> Tuple[ImageId, Image, Image, PascalBoxes, Labels]:
        id = self.keys[idx]
        annot = self.rows[id]
        image = imread(annot["image_path"])
        prev_row = find_prev_frame(
            self.rows, sequence_id=annot["sequence_id"], frame_id=annot["frame_id"]
        )
        image0: Any = imread(prev_row["image_path"]) if prev_row is not None else image
        boxes = annot["boxes"]
        labels = annot["labels"]
        transed = self.transforms(
            image=image, image0=image0, bboxes=boxes, labels=labels
        )
        return (
            ImageId(id),
            Image(transed["image"].float()),
            Image(transed["image0"].float()),
            PascalBoxes(torch.tensor(transed["bboxes"])),
            Labels(torch.tensor(transed["labels"])),
        )

    def __len__(self) -> int:
        return len(self.rows)


class TestDataset(Dataset):
    def __init__(
        self,
        rows: TestRows,
        transforms: typing.Any,
    ) -> None:
        self.rows = list(rows.items())
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple[ImageId, Image]:
        id, row = self.rows[idx]
        image = imread(row["image_path"])
        transed = self.transforms(image=image)
        return (
            ImageId(id),
            Image(transed["image"].float()),
        )

    def __len__(self) -> int:
        return len(self.rows)
