import glob, tqdm, torch, json, shutil
from pathlib import Path
from typing import Dict, Any, List
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
from fish.data import read_test_rows, TestDataset, test_transforms, inv_normalize
from fish.centernet import config
from fish.centernet.train import model, model_loader, to_boxes
from ensemble_boxes import weighted_boxes_fusion
from torchvision.transforms.functional import hflip, vflip
from fish.store import StoreApi, Box

Submission = Dict[str, Any]


store = StoreApi()


def annotate(id: str, boxes: PascalBoxes, labels: Labels) -> None:
    payload_boxes: List[Box] = [
        dict(
            x0=float(b[0]),
            y0=float(b[1]),
            x1=float(b[2]),
            y1=float(b[3]),
            label=str(l),
        )
        for b, l in zip(boxes.tolist(), labels.tolist())
    ]
    store.annotate(id, payload_boxes)


@torch.no_grad()
def predict(device: str) -> None:
    rows = read_test_rows("/store")
    out_dir = Path("/store/pseudo")
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
    weights = [1, 1, 1]
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

            _boxes, confidences, labels = weighted_boxes_fusion(
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
            _boxes = resize(PascalBoxes(torch.from_numpy(_boxes)), (w, h))
            filter_indices = confidences > config.to_boxes_threshold
            _boxes = _boxes[filter_indices]
            labels = labels[filter_indices]
            plot = DetectionPlot(inv_normalize(img))
            plot.draw_boxes(boxes=_boxes, confidences=confidences, labels=labels)
            plot.save(out_dir.joinpath(f"{id}.jpg"))
            _boxes = resize(
                _boxes, scale=(config.original_width / w, config.original_height / h)
            )
            annotate(id, boxes=_boxes, labels=labels)


if __name__ == "__main__":
    predict("cuda")
