import torch, typing, numpy as np, cv2
from typing import *
from torch import Tensor
from torch.utils.data import Dataset
from skimage.io import imread
from toolz import map, pipe, keyfilter
from io import BytesIO
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
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
from object_detection.entities.box import filter_size
import albumentations as albm
from sklearn.model_selection import GroupKFold
import glob, typing, json, re
from pathlib import Path


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
        image_path = str(image_dir.joinpath(f"{id}.jpg"))
        rows[id] = dict(
            image_path=image_path,
            sequence_id=sequence_id,
            frame_id=frame_id,
        )
    return rows


def kfold(
    rows: Annotations, n_splits: int = 5, seed: int = 7
) -> typing.Tuple[Annotations, Annotations]:
    kf = GroupKFold(n_splits)
    x = list(rows.keys())
    y = [len(i["boxes"]) for i in rows.values()]
    groups = [i["sequence_id"] for i in rows.values()]

    train_ids, test_ids = next(kf.split(x, y, groups))
    train_keys = set([x[i] for i in train_ids])
    test_keys = set([x[i] for i in test_ids])
    return keyfilter(lambda k: k in train_keys, rows), keyfilter(
        lambda k: k in test_keys, rows
    )


bbox_params = {"format": "pascal_voc", "label_fields": ["labels"]}
test_transforms = lambda size: albm.Compose(
    [
        albm.LongestMaxSize(max_size=size),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)

prediction_transforms = lambda size: albm.Compose(
    [
        albm.LongestMaxSize(max_size=size),
        ToTensorV2(),
    ],
)

train_transforms = lambda size: albm.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=20,
            p=1.0,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        albm.LongestMaxSize(max_size=size),
        A.OneOf(
            [
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.9
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.9
                ),
            ],
            p=0.9,
        ),
        A.OneOf(
            [A.Blur(blur_limit=3, p=1.0), A.MedianBlur(blur_limit=3, p=1.0)], p=0.2
        ),
        A.OneOf(
            [
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ],
            p=0.2,
        ),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)


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
        boxes, indices = filter_size(
            PascalBoxes(torch.tensor(annot["boxes"])), lambda area: area > 36
        )
        labels = Labels(torch.tensor(annot["labels"])[indices])
        transed = self.transforms(image=image, bboxes=boxes, labels=labels)
        return (
            ImageId(id),
            Image(transed["image"].float()),
            PascalBoxes(transed["bboxes"]),
            Labels(transed["labels"]),
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
