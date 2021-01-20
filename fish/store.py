import base64
import requests
from urllib.parse import urljoin
from toolz.curried import groupby
from typing import List
import os
import typing
from typing import *

Box = typing.TypedDict(
    "Box", {"x0": float, "y0": float, "x1": float, "y1": float, "label": str}
)
Row = typing.TypedDict(
    "Row",
    {
        "id": str,
        "data": str,
        "hasBox": typing.Optional[bool],
        "boxes": typing.List[Box],
    },
)
Rows = typing.List[Row]


class StoreApi:
    def __init__(self, url: str = os.getenv("STORE_URL", "")) -> None:
        self.url = url
        self.cache: typing.Dict[str, Row] = {}

    def create(self, id: str, data: bytes) -> None:
        encoded = base64.b64encode(data)
        res = requests.post(
            urljoin(self.url, "/api/v1/image/create"),
            json={
                "id": id,
                "data": encoded.decode("ascii"),
            },
        )
        res.raise_for_status()

    def filter(self, state: str = "Done") -> Rows:
        img_res = requests.post(
            urljoin(self.url, "/api/v1/image/filter"),
            json=dict(state=state),
        )
        img_res.raise_for_status()
        boxes_res = requests.post(
            urljoin(self.url, "/api/v1/box/filter"),
            json={},
        )
        boxes_res.raise_for_status()
        boxes = groupby(lambda x: x["imageId"])(boxes_res.json())
        rows = img_res.json()
        for row in rows:
            row["boxes"] = boxes.get(row["id"]) or []
        return rows

    def predict(self, id: str, boxes: typing.List[Box], loss:Optional[float]=None) -> None:
        res = requests.post(
            urljoin(self.url, "/api/v1/box/predict"),
            json={"imageId": id, "boxes": boxes, "loss":loss},
        )
        res.raise_for_status()

    def annotate(self, id: str, boxes: typing.List[Box]) -> None:
        res = requests.post(
            urljoin(self.url, "/api/v1/box/predict"),
            json={"imageId": id, "boxes": boxes},
        )
        res.raise_for_status()
