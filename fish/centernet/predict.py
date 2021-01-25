import glob, tqdm, torch, json, shutil
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader
from object_detection.utils import DetectionPlot
from object_detection.entities.box import (
    yolo_to_pascal,
    yolo_vflip,
    yolo_hflip,
    PascalBoxes,
    Confidences,
    Labels,
    resize,
    filter_size,
    resize,
)
from fish.data import (
    read_test_rows,
    TestDataset,
    test_transforms,
    inv_normalize,
    add_submission,
    Submission,
)
from fish.centernet import config
from fish.centernet.train import model, model_loader, to_boxes
from ensemble_boxes import weighted_boxes_fusion
from torchvision.transforms.functional import hflip, vflip


@torch.no_grad()
def predict(device: str) -> None:
    rows = read_test_rows("/store")
    out_dir = Path("/store/submission")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)
    dataset = TestDataset(rows=rows, transforms=test_transforms)
    net = model_loader.load_if_needed(model).to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False,
    )
    weights = [1, 1]
    submission: Submission = {}
    for ids, image_batch in tqdm.tqdm(loader):
        image_batch = image_batch.to(device)

        h_box_batch, h_confidence_batch, h_label_batch = to_boxes(
            net(hflip(image_batch))
        )
        box_batch, confidence_batch, label_batch = to_boxes(net(image_batch))

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
            _, h, w = img.shape

            boxes, confidences, labels = weighted_boxes_fusion(
                [
                    yolo_to_pascal(boxes, (1, 1)),
                    yolo_to_pascal(yolo_hflip(h_boxes), (1, 1)),
                ],
                [confidences, h_confidences],
                [labels, h_labels],
                iou_thr=config.iou_threshold,
                weights=weights,
                skip_box_thr=config.to_boxes_threshold,
            )
            filter_indices = confidences > config.to_boxes_threshold
            confidences = confidences[filter_indices]
            labels = labels[filter_indices]
            boxes = PascalBoxes(torch.from_numpy(boxes)[filter_indices])
            row = rows[id]
            sequence_id = row["sequence_id"]
            frame_id = row["frame_id"]
            plot = DetectionPlot(inv_normalize(img))
            plot.draw_boxes(boxes=resize(boxes, (w, h)), confidences=confidences, labels=labels)
            plot.save(out_dir.joinpath(f"{sequence_id}-{frame_id}-{id}.jpg"))
            boxes = resize(
                boxes, scale=(config.original_width, config.original_height)
            )
            add_submission(submission, id, boxes=boxes, labels=labels)

    with open(out_dir.joinpath("submission.json"), "w") as f:
        json.dump(submission, f)


if __name__ == "__main__":
    predict("cuda")
