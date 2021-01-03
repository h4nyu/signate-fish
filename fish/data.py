import torch, typing, numpy as np
from torch.utils.data import Dataset
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
    YoloBoxes,
    Labels,
)
import cv2
import albumentations as albm

bbox_params = {"format": "yolo", "label_fields": ["labels"]}
test_transforms = albm.Compose(
    [
        albm.LongestMaxSize(max_size=config.image_size),
        albm.PadIfNeeded(
            min_width=config.image_size,
            min_height=config.image_size,
            border_mode=cv2.BORDER_CONSTANT,
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
            border_mode=cv2.BORDER_CONSTANT,
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
        A.ColorJitter(p=0.2),
        albm.RandomBrightnessContrast(),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)

# class TrainDataset(Dataset):
#     def __init__(
#         self,
#         repo: ImageRepository,
#         rows: Rows,
#         mode: typing.Literal["test", "train"] = "train",
#     ) -> None:
#         self.repo = repo
#         self.rows = rows
#         self.transforms = train_transforms if mode == "train" else test_transforms

#     def __getitem__(self, idx: int) -> TrainSample:
#         id = self.rows[idx]["id"]
#         res = self.repo.find(id)
#         image = np.array(
#             PILImage.open(BytesIO(base64.b64decode(res["data"]))).convert("RGB")
#         )
#         boxes = YoloBoxes(
#             torch.tensor(
#                 [
#                     [
#                         (b["x0"] + b["x1"]) / 2,
#                         (b["y1"] + b["y0"]) / 2,
#                         b["x1"] - b["x0"],
#                         b["y1"] - b["y0"],
#                     ]
#                     for b in res["boxes"]
#                 ]
#             ).clamp(max=1.0 - 1e-2, min=0.0 + 1e-2)
#         )
#         labels = Labels(torch.tensor([0 for b in boxes]))
#         transed = self.transforms(image=image, bboxes=boxes, labels=labels)
#         return (
#             ImageId(id),
#             Image(transed["image"] / 255),
#             YoloBoxes(torch.tensor(transed["bboxes"])),
#             Labels(torch.tensor(transed["labels"])),
#         )

#     def __len__(self) -> int:
#         return len(self.rows)
