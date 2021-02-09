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
    box_hflip,
)
from fish.data import (
    read_test_rows,
    FileDataset,
    test_transforms,
    kfold,
    inv_normalize,
    add_submission,
    test_transforms,
    TestDataset,
    Submission,
    filter_limit,
    sort_by_size,
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
    out_dir = Path(config.out_dir).joinpath("submission")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)
    dataset = TestDataset(rows=rows, transforms=test_transforms)
    net = model_loader.load_if_needed(model).to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    weights = [2, 1]
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
            m_boxes, m_confidences, m_labels = weighted_boxes_fusion(
                [
                    resize(boxes, (1 / w, 1 / h)),
                    box_hflip(resize(h_boxes, (1 / w, 1 / h)), (1, 1)),
                ],
                [confidences, h_confidences],
                [labels, h_labels],
                iou_thr=config.iou_threshold,
                weights=weights,
            )
            m_boxes = PascalBoxes(torch.from_numpy(m_boxes))
            m_labels = Labels(torch.from_numpy(m_labels))
            m_confidences = Confidences(torch.from_numpy(m_confidences))
            m_boxes, m_labels, m_confidences = filter_limit(
                m_boxes, m_labels, m_confidences
            )
            m_boxes = resize(m_boxes, (w, h))
            plot = DetectionPlot(inv_normalize(img))
            plot.draw_boxes(
                boxes=PascalBoxes(m_boxes),
                labels=Labels(m_labels),
                confidences=m_confidences,
            )
            row = rows[id]
            sequence_id = row["sequence_id"]
            frame_id = row["frame_id"]
            plot.save(out_dir.joinpath(f"{sequence_id}-{frame_id}-{id}.jpg"))

            m_boxes = resize(
                m_boxes, scale=(config.original_width / w, config.original_height / h)
            )

            add_submission(submission, id, boxes=m_boxes, labels=m_labels)

    with open(out_dir.joinpath("submission.json"), "w") as f:
        json.dump(submission, f)


if __name__ == "__main__":
    predict("cuda")
