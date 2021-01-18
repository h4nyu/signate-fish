import shutil, random, torch, json
from typing import *
from PIL import Image as PILImage
from pathlib import Path
from object_detection.entities.box import PascalBoxes, Labels
from fish.data import (
    read_train_rows,
    resize_mix,
    Annotation,
    annot_to_tuple,
)

rows = read_train_rows("/store")
keys = list(rows.keys())

out_dir = Path("/store/resize-mix")
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(exist_ok=True)


for i in range(500):
    base = rows[random.choice(keys)]
    base_id = base["id"]
    other = rows[random.choice(keys)]
    other_id = other["id"]
    id = f"{base_id}-{other_id}"
    img, boxes, labels = resize_mix(
        annot_to_tuple(base), annot_to_tuple(other), scale=0.6
    )
    image_path = str(out_dir.joinpath(f"{id}.jpg"))
    img.save(image_path)
    annot: Annotation = dict(
        id=id,
        boxes=boxes.tolist(),
        labels=labels.tolist(),
        image_path=image_path,
        frame_id=0,
        sequence_id=0,
    )
    with open(out_dir.joinpath(f"{id}.json"), "w") as f:
        json.dump(annot, f)
