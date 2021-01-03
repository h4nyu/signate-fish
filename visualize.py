from fish.store import ImageStore
from pathlib import Path
from skimage.io import imread
from fish.data import FileDataset, test_transforms
from albumentations.pytorch.transforms import ToTensorV2
from object_detection.utils import DetectionPlot

store = ImageStore("/store")
vis_dir = Path("/store/vis")
vis_dir.mkdir(exist_ok=True)

rows = store.read()
dataset = FileDataset(
    rows=rows,
    transforms=ToTensorV2(),
)
for i in range(len(dataset)):
    id, image, boxes, labels = dataset[i]
    _, h, w = image.shape
    if len(labels) == 0:
        continue
    plot = DetectionPlot(image)
    plot.draw_boxes(boxes=boxes, labels=labels, line_width=4, color="red")
    plot.save(vis_dir.joinpath(f"{id}.jpg"))
