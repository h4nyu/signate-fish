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


def parse_label(value: str) -> typing.Optional[int]:
    if value == "Jumper School":
        return 0
    if value == "Breezer School":
        return 0
    return None


class ImageStore:
    def __init__(self, dataset_dir: str) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.annotation_files: typing.List[Path] = []
        self.annotations: Annotations = {}

    def read(self) -> Annotations:
        annotation_dir = self.dataset_dir.joinpath("train_annotations")
        image_dir = self.dataset_dir.joinpath("train_images")
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
            self.annotations[id] = dict(
                boxes=boxes, labels=labels, image_path=image_path
            )
        return self.annotations
