import glob, typing, json, re
from pathlib import Path


Annotation = typing.TypedDict(
    "Annotation", {"boxes": typing.List[typing.List[int]], "labels": typing.List[str]}
)
Annotations = typing.Dict[str, Annotation]


class ImageStore:
    def __init__(self, dataset_dir: str) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.annotation_files: typing.List[Path] = []
        self.annotations: Annotations = {}

    def read(self) -> Annotations:
        annotation_dir = self.dataset_dir.joinpath("train_annotations")
        for p in glob.glob(f"{annotation_dir}/*.json"):
            path = Path(p)
            id = re.findall("[0-9]+", path.stem)[0]
            boxes: typing.List[typing.List[int]] = []
            labels: typing.List[str] = []
            with path.open("r") as f:
                rows = json.load(f)["labels"]
            for k, v in rows.items():
                boxes += v
                labels += [k] * len(v)
            self.annotations[id] = {"boxes": boxes, "labels": labels}
        return self.annotations
