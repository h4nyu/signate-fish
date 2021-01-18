import glob, tqdm, torch, json, shutil
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader
from object_detection.utils import DetectionPlot
from object_detection.metrics import MeanAveragePrecision
from object_detection.entities.box import (
    PascalBoxes,
    Confidences,
    Labels,
    resize,
)
from fish.data import (
    read_test_rows,
    FileDataset,
    test_transforms,
    kfold,
    inv_normalize,
    add_submission,
    prediction_transforms,
    TestDataset,
    Submission,
)
from fish.effdet import config
from fish.effdet.train import model, model_loader, to_boxes, collate_fn
from ensemble_boxes import weighted_boxes_fusion
from torchvision.transforms.functional import hflip, vflip

from logging import (
    getLogger,
)

logger = getLogger(config.out_dir)


@torch.no_grad()
def predict(device: str) -> None:
    rows = read_test_rows("/store")
    out_dir = Path("/store/submission")
    shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)
    dataset = TestDataset(rows=rows, transforms=prediction_transforms)
    net = model_loader.load_if_needed(model).to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    weights = [1]
    submission: Submission = {}
    for ids, image_batch in tqdm.tqdm(loader):
        image_batch = image_batch.to(device)
        _, _, h, w = image_batch.shape
        box_batch, confidence_batch, label_batch = to_boxes(net(image_batch))
        h_box_batch, h_confidence_batch, h_label_batch = to_boxes(
            net(hflip(image_batch))
        )
        for (
            img,
            id,
            boxes,
            h_boxes,
            confidences,
            h_confidences,
            labels,
            h_labels,
        ) in zip(
            image_batch,
            ids,
            box_batch,
            h_box_batch,
            confidence_batch,
            h_confidence_batch,
            label_batch,
            h_label_batch,
        ):
            plot = DetectionPlot(inv_normalize(img))
            plot.draw_boxes(boxes=boxes, labels=labels, confidences=confidences)
            plot.save(out_dir.joinpath(f"{id}.jpg"))

            boxes = resize(
                boxes, scale=(config.original_width / w, config.original_height / h)
            )

            add_submission(submission, id, boxes=boxes, labels=labels)

    with open(out_dir.joinpath("submission.json"), "w") as f:
        json.dump(submission, f)


if __name__ == "__main__":
    predict("cuda")
