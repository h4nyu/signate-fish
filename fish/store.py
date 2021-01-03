import glob, typing, json, re
from pathlib import Path


Annotation = typing.TypedDict(
    "Annotation",
    {
        "boxes": typing.List[typing.List[int]],
        "labels": typing.List[str],
        "image_path": str,
    },
)
Annotations = typing.Dict[str, Annotation]


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
            labels: typing.List[str] = []
            with path.open("r") as f:
                rows = json.load(f)["labels"]
            for k, v in rows.items():
                boxes += v
                labels += [k] * len(v)
            image_path = str(image_dir.joinpath(f"{id}.jpg"))
            self.annotations[id] = dict(
                boxes=boxes, labels=labels, image_path=image_path
            )
        return self.annotations
