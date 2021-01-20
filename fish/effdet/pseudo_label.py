import glob, tqdm, torch, json, shutil
from pathlib import Path
from typing import Dict, Any, List
from torch.utils.data import DataLoader
from object_detection.utils import DetectionPlot
from object_detection.metrics import MeanAveragePrecision
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
    box_hflip,
    box_vflip,
)
from fish.data import (
    read_test_rows,
    LabeledDataset,
    test_transforms,
    inv_normalize,
    annotate,
    pseudo_predict,
    read_test_rows,
    read_train_rows,
    FileDataset,
)
from fish.effdet import config
from fish.effdet.train import model, model_loader, to_boxes, collate_fn, criterion
from ensemble_boxes import weighted_boxes_fusion
from torchvision.transforms.functional import hflip, vflip
from fish.store import StoreApi, Box
from toolz.curried import pipe, filter, map, keyfilter

Submission = Dict[str, Any]


store = StoreApi()


@torch.no_grad()
def predict(device: str) -> None:
    rows = store.filter(state="Todo")
    rows = pipe(rows, filter(lambda x: "test" in x["id"]), list)
    out_dir = Path("/store/pseudo")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)
    dataset = LabeledDataset(rows=rows, transforms=test_transforms)
    net = model_loader.load_if_needed(model).to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    weights = [2, 1]
    for image_batch, gt_box_batch, gt_label_batch, ids in tqdm.tqdm(loader):
        image_batch = image_batch.to(device)
        gt_box_batch = [x.to(device) for x in gt_box_batch]
        gt_label_batch = [x.to(device) for x in gt_label_batch]
        _, _, h, w = image_batch.shape
        netout = net(image_batch)
        loss, box_loss, label_loss = criterion(
            image_batch,
            netout,
            gt_box_batch,
            gt_label_batch,
        )
        box_batch, confidence_batch, label_batch = to_boxes(netout)
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
            gt_boxes,
            gt_labels,
        ) in zip(
            image_batch,
            ids,
            box_batch,
            h_box_batch,
            confidence_batch,
            h_confidence_batch,
            label_batch,
            h_label_batch,
            gt_box_batch,
            gt_label_batch,
        ):

            metrics = MeanAveragePrecision(
                iou_threshold=0.3, num_classes=config.num_classes
            )
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
            m_confidences = torch.from_numpy(m_confidences)
            indices = m_confidences > config.pseudo_threshold
            m_boxes = PascalBoxes(torch.from_numpy(m_boxes)[indices])
            m_labels = Labels(torch.from_numpy(m_labels)[indices])
            m_confidences = Confidences(m_confidences[indices])
            # annotate(store, id, boxes=resize(gt_boxes, (1 / w, 1 / h)), labels=gt_labels)
            pseudo_predict(store, id, boxes=m_boxes, labels=m_labels)
            print(id)
            plot = DetectionPlot(inv_normalize(img))
            plot.draw_boxes(
                boxes=resize(m_boxes, (w, h)),
                labels=m_labels,
                confidences=m_confidences,
            )
            plot.save(out_dir.joinpath(f"{id}.jpg"))


if __name__ == "__main__":
    predict("cuda")
