import glob, tqdm, torch, json, shutil, numpy as np
from pathlib import Path
from typing import Dict, Any
from fish.data import (
    read_test_rows,
    FileDataset,
    kfold,
    inv_normalize,
    add_submission,
    TestDataset,
    Submission,
)
from object_detection.utils import DetectionPlot
from skimage.io import imread
from albumentations.pytorch.transforms import ToTensorV2
from ensemble_boxes import weighted_boxes_fusion
from torchvision.transforms.functional import hflip, vflip
from object_detection.entities.box import (
    PascalBoxes,
    Confidences,
    Labels,
    resize,
    box_hflip,
)
from fish import config

from logging import (
    getLogger,
)

files = [
    "/srv/store/efficientdet-6-1-96-67/submission/submission.json",
    "/srv/store/efficientdet-6-1-96-567/submission/submission.json",
]

transforms = ToTensorV2()


if __name__ == "__main__":
    out_dir = Path("/store/submission")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)
    submissions = []
    samples = read_test_rows("/store")
    for p in files:
        with open(p, "r") as f:
            d = json.load(f)
        submissions.append(d)

    submission: Submission = {}
    for id, row in tqdm.tqdm(samples.items()):
        sequence_id = row["sequence_id"]
        frame_id = row["frame_id"]
        image_path = row["image_path"]
        subs = [s[f"{id}.jpg"] for s in submissions]
        image = transforms(image=imread(image_path))["image"]
        plot = DetectionPlot(image / 255.0)
        out_box_list = []
        out_label_list = []
        for category, label in [("Jumper School", 0), ("Breezer School", 1)]:
            box_list = []
            confidence_list = []
            label_list = []
            for sub in subs:
                boxes = np.array(sub.get(category) or [])
                boxes[:, [0, 2]] = boxes[:, [0, 2]] / config.original_width
                boxes[:, [1, 3]] = boxes[:, [1, 3]] / config.original_height
                box_list.append(boxes)
                confidence_list.append(np.linspace(1.0, 0.0, num=len(boxes)))
                label_list.append(np.ones(len(boxes)) * label)
            m_boxes, m_confidences, m_labels = weighted_boxes_fusion(
                box_list,
                confidence_list,
                label_list,
                iou_thr=config.iou_threshold,
                weights=[1.0 for _ in files],
                skip_box_thr=0.0,
            )
            out_box_list.append(m_boxes)
            out_label_list.append(m_labels)
            plot.draw_boxes(
                boxes=resize(
                    PascalBoxes(torch.from_numpy(m_boxes)),
                    (config.original_width, config.original_height),
                ),
                labels=Labels(m_labels),
                confidences=m_confidences,
            )
        out_boxes = resize(
            PascalBoxes(torch.from_numpy(np.concatenate(out_box_list))),
            (config.original_width, config.original_height),
        )
        out_labels = Labels(torch.from_numpy(np.concatenate(out_label_list)))
        add_submission(submission, id, boxes=out_boxes, labels=out_labels)
        plot.save(out_dir.joinpath(f"{sequence_id}-{frame_id}-{id}.jpg"))

    with open(out_dir.joinpath("submission.json"), "w") as f:
        json.dump(submission, f)
