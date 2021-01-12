from pathlib import Path
from tqdm import tqdm
from skimage.io import imread
from fish.data import FileDataset, test_transforms, read_train_rows
from albumentations.pytorch.transforms import ToTensorV2
from object_detection.utils import DetectionPlot

vis_dir = Path("/store/vis-train")
vis_dir.mkdir(exist_ok=True)

train_rows = read_train_rows("/store")
dataset = FileDataset(
    rows=train_rows,
    transforms=ToTensorV2(),
)
for i in tqdm(range(len(dataset))):
    id, image, boxes, labels = dataset[i]
    _, h, w = image.shape
    row = train_rows[id]
    sequence_id = row["sequence_id"]
    frame_id = row["frame_id"]
    if len(labels) == 0:
        continue
    plot = DetectionPlot(image)
    plot.draw_boxes(boxes=boxes, labels=labels, line_width=2, color="red")
    plot.save(vis_dir.joinpath(f"seq-{sequence_id}_frame-{frame_id}_{id}.jpg"))
