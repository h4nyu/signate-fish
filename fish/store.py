import base64
import requests
from urllib.parse import urljoin
import os
import typing

Box = typing.TypedDict("Box", {"x0": float, "y0": float, "x1": float, "y1": float})
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

    def filter(self) -> Rows:
        return requests.post(
            urljoin(self.url, "/api/v1/image/filter"),
            json={},
        ).json()

    def find(self, id: str) -> Row:
        if id not in self.cache:
            img = requests.post(
                urljoin(self.url, "/api/v1/image/find"), json={"id": id}
            ).json()
            boxes = requests.post(
                urljoin(self.url, "/api/v1/box/filter"),
                json={"imageId": id, "isGrandTruth": True},
            ).json()
            img["boxes"] = boxes
            self.cache[id] = img
        return self.cache[id]

    def predict(self, id: str, boxes: typing.List[Box]) -> None:
        res = requests.post(
            urljoin(self.url, "/api/v1/box/predict"),
            json={"imageId": id, "boxes": boxes},
        )
        res.raise_for_status()
