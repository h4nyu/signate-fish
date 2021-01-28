import shutil
from pathlib import Path
from tqdm import tqdm
from skimage.io import imread
from fish.data import FileDataset, test_transforms, read_train_rows, inv_normalize
from albumentations.pytorch.transforms import ToTensorV2
from object_detection.utils import DetectionPlot
from object_detection.entities.box import PascalBoxes, Labels
from fish import config

vis_dir = Path("/store/vis-train")
if vis_dir.exists():
    shutil.rmtree(vis_dir)
vis_dir.mkdir(exist_ok=True)

train_rows = read_train_rows("/store")
dataset = FileDataset(
    rows=train_rows,
    transforms=test_transforms,
)
for i in tqdm(range(len(dataset))):
    id, image, boxes, labels, __ = dataset[i]
    _, h, w = image.shape
    row = train_rows[id]
    sequence_id = row["sequence_id"]
    frame_id = row["frame_id"]
    if len(labels) == 0:
        continue
    plot = DetectionPlot(inv_normalize(image))
    for i, c in zip(range(config.num_classes), ["red", "blue"]):
        indices = labels == i
        plot.draw_boxes(
            boxes=PascalBoxes(boxes[indices]),
            labels=Labels(labels[indices]),
            line_width=1,
            color=c,
        )
    plot.save(vis_dir.joinpath(f"seq-{sequence_id}_frame-{frame_id}_{id}.jpg"))
