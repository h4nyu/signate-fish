import torch, typing, numpy as np, cv2, random
from typing import *
from torch import Tensor
from torch.utils.data import Dataset
from skimage.io import imread
from toolz.curried import map, pipe, keyfilter, valfilter, filter, sorted
from io import BytesIO
import albumentations as A
from torchvision.ops.boxes import box_iou, box_area, clip_boxes_to_image
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
    box_in_area,
)
from object_detection.entities.box import filter_size, resize, shift, Nummber
import albumentations as albm
from sklearn.model_selection import GroupKFold, StratifiedKFold
import glob, typing, json, re
from pathlib import Path
from fish import config
from fish.store import StoreApi, Rows, Box


Annotation = typing.TypedDict(
    "Annotation",
    {
        "id": str,
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
        "id": str,
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
    for key, label in [
        (
            "Jumper School",
            0,
        ),
        ("Breezer School", 1),
    ]:
        indices = labels == label
        if indices.sum() == 0:
            continue
        row[key] = boxes[indices][: config.to_box_limit].to("cpu").tolist()
    submission[f"{id}.jpg"] = row


def annot_to_tuple(row: Annotation) -> Tuple[Any, PascalBoxes, Labels]:
    img = PILImage.open(row["image_path"])
    boxes = PascalBoxes(torch.tensor(row["boxes"]))
    labels = Labels(torch.tensor(row["labels"]))
    return img, boxes, labels


def resize_mix(
    base: Tuple[Any, PascalBoxes, Labels],
    other: Tuple[Any, PascalBoxes, Labels],
    scale: float = 0.8,
) -> Tuple[Any, PascalBoxes, Labels]:
    base_img, base_boxes, base_labels = base
    other_img, other_boxes, other_labels = other
    base_w, base_h = base_img.size
    resized_w, resized_h = int(base_w * scale), int(base_h * scale)
    start_x, start_y = random.randint(0, base_w - resized_w), random.randint(
        0, base_h - resized_h
    )
    indices = ~box_in_area(
        base_boxes,
        torch.tensor([start_x, start_y, start_x + resized_w, start_y + resized_h]),
        min_fill=0.2,
    )
    base_boxes = PascalBoxes(base_boxes[indices])
    base_labels = Labels(base_labels[indices])

    other_boxes = resize(other_boxes, (scale, scale))
    other_boxes = shift(other_boxes, (start_x, start_y))

    other_img = other_img.resize((resized_w, resized_h))
    base_img.paste(other_img, (start_x, start_y))
    boxes = clip_boxes_to_image(torch.cat([other_boxes, base_boxes], dim=0), (base_h, base_w))
    labels = torch.cat([base_labels, other_labels], dim=0)
    return base_img, PascalBoxes(boxes), Labels(labels)


def kfold(
    rows: Annotations, n_splits: int = config.n_splits, seed: int = 7
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


def parse_label(value: str) -> typing.Optional[int]:
    if value == "Jumper School":
        return 0
    if value == "Breezer School":
        return 1
    return None


def to_label_filter(labels: Labels, num_classes: int = config.num_classes) -> Tensor:
    out = torch.zeros((num_classes,))
    if len(labels) == 0:
        return out

    unique_labels = torch.unique(labels.long())
    out[unique_labels] = 1
    return out


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
            id=id,
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
            id=id,
            image_path=image_path,
            sequence_id=sequence_id,
            frame_id=frame_id,
        )
    return rows


bbox_params = dict(format="pascal_voc", label_fields=["labels"], min_area=15)
test_transforms = A.Compose(
    [
        albm.LongestMaxSize(max_size=config.image_width),
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)

inv_normalize = Normalize(
    mean=[-m / s for m, s in zip(config.normalize_mean, config.normalize_std)],
    std=[1 / s for s in config.normalize_std],
)

train_transforms = albm.Compose(
    [
        A.OneOf(
            [
                A.Rotate(limit=(88, 92), p=0.5, border_mode=0),
                A.Rotate(limit=(-92, -88), p=0.5, border_mode=0),
                A.HorizontalFlip(p=0.5),
            ],
            p=1.0,
        ),
        A.PadIfNeeded(
            min_height=config.original_height, min_width=config.original_width, p=1
        ),
        A.RandomSizedCrop(
            (
                config.original_height - config.original_height * 0.05,
                config.original_height,
            ),
            height=config.original_height,
            width=config.original_width,
            p=0.8,
        ),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf(
            [
                A.Blur(blur_limit=17, p=0.5),
                A.MedianBlur(blur_limit=17, p=0.5),
                A.MotionBlur(blur_limit=13, p=0.5),
                A.MotionBlur(blur_limit=15, p=0.5),
            ],
            p=0.4,
        ),
        A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            ],
            p=0.3
        ),
        A.HueSaturationValue(p=0.3, hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15,),
        A.Cutout(
            max_w_size=32,
            max_h_size=32,
        ),
        A.LongestMaxSize(
            max_size=config.image_width,
        ),
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
        num_classes: int = config.num_classes,
    ) -> None:
        self.rows = list(rows.items())
        self.transforms = transforms

    def __getitem__(
        self, idx: int
    ) -> Tuple[ImageId, Image, PascalBoxes, Labels, Tensor]:
        id, annot = self.rows[idx]
        image = imread(annot["image_path"])
        boxes = annot["boxes"]
        labels = annot["labels"]
        transed = self.transforms(image=image, bboxes=boxes, labels=labels)
        t_labels = Labels(torch.tensor(transed["labels"]))
        lf = to_label_filter(t_labels)
        return (
            ImageId(id),
            Image(transed["image"]),
            PascalBoxes(torch.tensor(transed["bboxes"])),
            t_labels,
            lf,
        )

    def __len__(self) -> int:
        return len(self.rows)


class LabeledDataset(Dataset):
    def __init__(
        self,
        rows: Rows,
        transforms: typing.Any,
        image_dir: str = "/store/images",
        num_classes: int = config.num_classes,
    ) -> None:
        self.rows = rows
        self.transforms = transforms
        self.image_dir = Path(image_dir)
        self.num_classes = num_classes

    def __getitem__(
        self, idx: int
    ) -> Tuple[ImageId, Image, PascalBoxes, Labels, Tensor]:
        row = self.rows[idx]
        id = row["id"]
        path = self.image_dir.joinpath(id if "jpg" in id else f"{id}.jpg")
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
            ).clamp(min=0, max=1)
        )
        boxes = resize(boxes, (w, h))
        labels = [int(float(b["label"])) for b in row["boxes"]]
        transed = self.transforms(image=image, bboxes=boxes, labels=labels)
        t_labels = Labels(torch.tensor(transed["labels"]))
        lf = to_label_filter(t_labels)
        return (
            ImageId(id),
            Image(transed["image"]),
            PascalBoxes(torch.tensor(transed["bboxes"])),
            t_labels,
            lf,
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


class NegativeDataset(Dataset):
    def __init__(
        self,
        rows: List[Annotation],
        transforms: typing.Any,
        image_dir: str = "/store/images",
    ) -> None:
        self.transforms = transforms
        self.rows = rows
        self.image_dir = Path(image_dir)

    def __getitem__(
        self, idx: int
    ) -> Tuple[ImageId, Image, PascalBoxes, Labels, Tensor]:
        row = self.rows[idx]
        id = row["id"]
        path = self.image_dir.joinpath(f"{id}.jpg")
        image = imread(path)
        transed = self.transforms(image=image, bboxes=[], labels=[])
        t_labels = Labels(torch.tensor(transed["labels"]))
        lf = to_label_filter(t_labels)
        return (
            ImageId(id),
            Image(transed["image"]),
            PascalBoxes(torch.tensor(transed["bboxes"])),
            t_labels,
            lf,
        )

    def __len__(self) -> int:
        return len(self.rows)


class ResizeMixDataset(Dataset):
    def __init__(
        self,
        rows: Annotations,
        transforms: typing.Any,
    ) -> None:
        self.rows = list(rows.items())
        self.indices = list(range(len(self.rows)))
        self.transforms = transforms

    def __getitem__(
        self, idx: int
    ) -> Tuple[ImageId, Image, PascalBoxes, Labels, Tensor]:
        id, base = self.rows[idx]
        _, other = self.rows[random.choice(self.indices)]
        scale = random.uniform(0.5, 0.7)
        image, boxes, labels = resize_mix(annot_to_tuple(base), annot_to_tuple(other), scale=scale)
        transed = self.transforms(image=np.array(image), bboxes=boxes, labels=labels)
        t_labels = Labels(torch.tensor(transed["labels"]))
        lf = to_label_filter(t_labels)
        return (
            ImageId(id),
            Image(transed["image"]),
            PascalBoxes(torch.tensor(transed["bboxes"])),
            t_labels,
            lf,
        )

    def __len__(self) -> int:
        return len(self.rows)


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
        transed = self.transforms(image=image, labels=[], bboxes=[])
        return (
            ImageId(id),
            Image(transed["image"].float()),
        )

    def __len__(self) -> int:
        return len(self.rows)


def annotate(store: StoreApi, id: str, boxes: PascalBoxes, labels: Labels) -> None:
    payload_boxes: List[Box] = [
        dict(
            x0=float(b[0]),
            y0=float(b[1]),
            x1=float(b[2]),
            y1=float(b[3]),
            label=str(l),
        )
        for b, l in zip(boxes.tolist(), labels.tolist())
    ]
    store.annotate(id, payload_boxes)


def pseudo_predict(
    store: StoreApi,
    id: str,
    boxes: PascalBoxes,
    labels: Labels,
    loss: Optional[float] = None,
) -> None:
    payload_boxes: List[Box] = [
        dict(
            x0=float(b[0]),
            y0=float(b[1]),
            x1=float(b[2]),
            y1=float(b[3]),
            label=str(l),
        )
        for b, l in zip(boxes.tolist(), labels.tolist())
    ]
    store.predict(id, payload_boxes, loss)
