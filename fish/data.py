import torch, typing, numpy as np
from torch.utils.data import Dataset
from skimage.io import imread
from fish import config
from toolz import map
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


bbox_params = {"format": "pascal_voc", "label_fields": ["labels"]}
test_transforms = albm.Compose(
    [
        albm.LongestMaxSize(max_size=config.image_size),
        albm.PadIfNeeded(
            min_width=config.image_size,
            min_height=config.image_size,
        ),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)

train_transforms = albm.Compose(
    [
        albm.LongestMaxSize(max_size=config.image_size),
        albm.PadIfNeeded(
            min_width=config.image_size,
            min_height=config.image_size,
        ),
        A.OneOf(
            [
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ],
            p=0.2,
        ),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.2),
        A.Cutout(),
        albm.RandomBrightnessContrast(),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)


class FileDataset(Dataset):
    def __init__(
        self,
        rows: Annotations,
        mode: typing.Literal["test", "train"] = "train",
    ) -> None:
        self.rows = list(rows.items())
        self.transforms = train_transforms if mode == "train" else test_transforms

    def __getitem__(self, idx: int) -> TrainSample:
        id, annot = self.rows[idx]
        image = imread(annot["image_path"])
        boxes = PascalBoxes(torch.tensor(annot["boxes"]))
        labels = Labels(torch.tensor(annot["labels"]))
        transed = self.transforms(image=image, bboxes=boxes, labels=labels)
        return (
            ImageId(id),
            Image(transed["image"]),
            PascalBoxes(torch.tensor(transed["bboxes"])),
            Labels(torch.tensor(transed["labels"])),
        )

    def __len__(self) -> int:
        return len(self.rows)
