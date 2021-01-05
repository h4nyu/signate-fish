import torch, typing, numpy as np
from torch.utils.data import Dataset
from skimage.io import imread
from fish import config
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
from fish.store import Annotations
from sklearn.model_selection import StratifiedKFold


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
        A.VerticalFlip(p=0.5),
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
        mode: typing.Literal["test", "train"] = "train",
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
