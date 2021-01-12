from pathlib import Path
from tqdm import tqdm
from skimage.io import imread
from fish.data import FileDataset, test_transforms, read_test_rows
from albumentations.pytorch.transforms import ToTensorV2
from object_detection.utils import DetectionPlot

vis_dir = Path("/store/vis-test")
vis_dir.mkdir(exist_ok=True)
transforms = ToTensorV2()

rows = read_test_rows("/store")
for id, row in tqdm(rows.items()):
    sequence_id = row["sequence_id"]
    frame_id = row["frame_id"]
    image = transforms(image=imread(row["image_path"]))["image"]
    plot = DetectionPlot(image)
    plot.save(vis_dir.joinpath(f"seq-{sequence_id}_frame-{frame_id}_{id}.jpg"))
