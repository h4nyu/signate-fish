import torch, typing, numpy as np
from typing import *
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
import albumentations as albm
from sklearn.model_selection import StratifiedKFold
import glob, typing, json, re
from pathlib import Path


Annotation = typing.TypedDict(
    "Annotation",
    {
        "boxes": typing.List[typing.List[int]],
        "labels": typing.List[int],
        "image_path": str,
    },
)
Annotations = typing.Dict[str, Annotation]

TestRow = typing.TypedDict(
    "TestRow",
    {
        "image_path": str,
    },
)
TestRows = typing.Dict[str, TestRow]


def parse_label(value: str) -> typing.Optional[int]:
    if value == "Jumper School":
        return 0
    if value == "Breezer School":
        return 1
    return None


def read_annotations(dataset_dir: str) -> Annotations:
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
            rows = json.load(f)["labels"]
        for k, v in rows.items():
            label = parse_label(k)
            if label is None:
                continue
            labels += [label] * len(v)
            boxes += v

        image_path = str(image_dir.joinpath(f"{id}.jpg"))
        annotations[id] = dict(boxes=boxes, labels=labels, image_path=image_path)
    return annotations


def read_test_rows(dataset_dir: str) -> TestRows:
    rows: TestRows = {}
    _dataset_dir = Path(dataset_dir)
    image_dir = _dataset_dir.joinpath("test_images")
    for p in glob.glob(f"{image_dir}/*.jpg"):
        path = Path(p)
        id = path.stem
        image_path = str(image_dir.joinpath(path))
        rows[id] = dict(image_path=image_path)
    return rows


def kfold(
    rows: Annotations, n_splits: int = 5, seed: int = 7
) -> typing.Tuple[Annotations, Annotations]:
    skf = StratifiedKFold(n_splits)
    x = list(rows.keys())
    y = [len(i["boxes"]) for i in rows.values()]
    train_ids, test_ids = next(skf.split(x, y))
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
        albm.LongestMaxSize(max_size=size),
        A.OneOf(
            [
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ],
            p=0.2,
        ),
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.2),
        albm.RandomBrightnessContrast(),
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
        boxes = annot["boxes"]
        labels = annot["labels"]
        transed = self.transforms(image=image, bboxes=boxes, labels=labels)
        return (
            ImageId(id),
            Image(transed["image"].float()),
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
        print(row["image_path"])
        image = imread(row["image_path"])
        transed = self.transforms(image=image)
        return (
            ImageId(id),
            Image(transed["image"].float()),
        )

    def __len__(self) -> int:
        return len(self.rows)
