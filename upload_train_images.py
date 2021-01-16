from typing import List
from tqdm import tqdm
from io import BytesIO
from fish.store import StoreApi, Box
from fish.data import read_train_rows
from fish import config
from PIL import Image as PILImage

rows = read_train_rows("/store")
api = StoreApi()

for id, row in tqdm(rows.items()):
    with open(row["image_path"], "rb") as f:
        data = f.read()
    try:
        payload_boxes:List[Box] = [
            dict(
                x0=b[0] / config.original_width,
                y0=b[1] / config.original_height,
                x1=b[2] / config.original_width,
                y1=b[3] / config.original_height,
                label=str(l),
            )
            for b, l in zip(row['boxes'], row['labels'])
        ]
        api.create(id, data)
        api.annotate(id, payload_boxes)
    except Exception as e:
        print(e)
