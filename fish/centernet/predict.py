import glob, tqdm, torch, json
from typing import Dict, Any
from torch.utils.data import DataLoader
from object_detection.utils import DetectionPlot
from object_detection.entities.box import (
    yolo_to_pascal,
    yolo_vflip,
    PascalBoxes,
    Labels,
    resize,
)
from fish.data import read_test_rows, TestDataset, prediction_transforms
from fish.centernet import config
from fish.centernet.train import model, model_loader, to_boxes
from ensemble_boxes import weighted_boxes_fusion
from torchvision.transforms.functional import hflip, vflip

Submission = Dict[str, Any]


def add_submission(
    submission: Submission, id: str, boxes: PascalBoxes, labels: Labels
) -> None:
    row = {}
    row["Jumper School"] = boxes[labels == 0].to("cpu").tolist()
    row["Breezer School"] = boxes[labels == 1].to("cpu").tolist()
    submission[f"{id}.jpg"] = row


@torch.no_grad()
def predict(device: str) -> None:
    rows = read_test_rows("/store")
    dataset = TestDataset(
        rows=rows, transforms=prediction_transforms(config.image_size)
    )
    net = model_loader.load_if_needed(model).to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size * 2,
        num_workers=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    weights = [2, 1]
    submission: Submission = {}
    for ids, image_batch in tqdm.tqdm(loader):
        image_batch = image_batch.to(device)
        original = to_boxes(net(image_batch))
        vfliped = to_boxes(net(vflip(image_batch)))

        for i, (img, id) in enumerate(zip(image_batch, ids)):
            _, h, w = img.shape
            boxes, confidences, labels = weighted_boxes_fusion(
                [
                    yolo_to_pascal(original[0][i], (1, 1)),
                    yolo_to_pascal(yolo_vflip(vfliped[0][i]), (1, 1)),
                ],
                [
                    original[1][i],
                    vfliped[1][i],
                ],
                [
                    original[2][i],
                    vfliped[2][i],
                ],
                iou_thr=config.iou_threshold,
                weights=weights,
                skip_box_thr=config.to_boxes_threshold,
            )
            x0, y0, x1, y1 = torch.from_numpy(boxes).unbind(-1)
            boxes = torch.stack(
                [
                    x0 * w,
                    y0 * h,
                    x1 * w,
                    y1 * h,
                ],
                dim=-1,
            )
            plot = DetectionPlot(img)
            plot.draw_boxes(boxes=boxes, confidences=confidences, labels=labels)
            plot.save(f"/store/pred-{id}.jpg")
            boxes = resize(
                boxes, scale=(config.original_width / w, config.original_height / h)
            )
            add_submission(submission, id, boxes=boxes, labels=labels)

    with open("/store/submission.json", "w") as f:
        json.dump(submission, f)


if __name__ == "__main__":
    predict("cuda")
