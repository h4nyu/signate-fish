import torch
from fish.data import (
    FileDataset,
    train_transforms,
    read_train_rows,
    read_test_rows,
    kfold,
    resize_mix,
    find_prev_frame,
    FrameDataset,
    inv_normalize,
    LabeledDataset,
    resize_mix,
    annot_to_tuple,
    ResizeMixDataset,
    test_transforms,
)
from albumentations.pytorch.transforms import ToTensorV2
from toolz.curried import filter, pipe
from object_detection.models.anchors import Anchors
from object_detection.entities.box import PascalBoxes
from object_detection.entities import ImageBatch
from fish.store import StoreApi
from toolz import valfilter
from object_detection.utils import DetectionPlot
from torchvision.transforms import ToTensor


def test_resize_mix() -> None:
    annotations = list(read_train_rows("/store").values())
    base = annot_to_tuple(annotations[197])
    other = annot_to_tuple(annotations[196])
    img, boxes, labels = resize_mix(base, other, scale=0.7)
    img_tensor = ToTensor()(img)
    plot = DetectionPlot(img_tensor)
    plot.draw_boxes(boxes=boxes, labels=labels)
    plot.save(f"store/test-resize-mix.jpg")


def test_resize_mix_dataset() -> None:
    annotations = read_train_rows("/store")
    dataset = ResizeMixDataset(rows=annotations, transforms=train_transforms)
    for i in range(3):
        id, image, boxes, labels = dataset[i]
        plot = DetectionPlot(inv_normalize(image))
        plot.draw_boxes(boxes=boxes, labels=labels)
        plot.save(f"store/test-resize-dataset-{i}.png")


def test_dataset() -> None:
    annotations = read_train_rows("/store")

    dataset = FileDataset(rows=annotations, transforms=train_transforms)
    for i in range(10):
        id, image, boxes, labels = dataset[10]
        plot = DetectionPlot(inv_normalize(image))
        plot.draw_boxes(boxes=boxes, labels=labels)
        plot.save(f"store/test-plot-{id}-{i}.png")


def test_anchors() -> None:
    annotations = read_train_rows("/store")
    dataset = FileDataset(rows=annotations, transforms=train_transforms)
    anchor_size = 2
    anchors = Anchors(size=anchor_size)
    fpn_level = 7
    for i in range(1):
        id, image, boxes, labels = dataset[5]
        dummy = ImageBatch(torch.zeros(1, *image.shape))
        anchor_box_level0 = anchors(dummy, 2 ** 7)
        anchor_box_level1 = anchors(dummy, 2 ** 6)
        plot = DetectionPlot(inv_normalize(image))
        # plot.draw_boxes(boxes=boxes, labels=labels)
        plot.draw_boxes(
            boxes=PascalBoxes(
                anchor_box_level0[
                    len(anchor_box_level0) // 2 : len(anchor_box_level0) // 2 + 4
                ]
            ),
            color="red",
        )
        plot.draw_boxes(
            boxes=PascalBoxes(
                anchor_box_level1[
                    len(anchor_box_level1) // 2 : len(anchor_box_level1) // 2 + 4
                ]
            ),
            color="blue",
        )
        plot.save(f"store/test-anchors-{id}-{i}.png")


def test_frame_dataset() -> None:
    annotations = read_train_rows("/store")

    dataset = FrameDataset(rows=annotations, transforms=train_transforms)
    for i in range(10):
        id, image0, image1, boxes, labels = dataset[5]
        plot = DetectionPlot(inv_normalize(image0))
        plot.draw_boxes(boxes=boxes, labels=labels)
        plot.save(f"store/test-plot-{id}-0-{i}.png")

        plot = DetectionPlot(inv_normalize(image1))
        plot.draw_boxes(boxes=boxes, labels=labels)
        plot.save(f"store/test-plot-{id}-1-{i}.png")


def test_labeled_dataset() -> None:
    api = StoreApi()
    rows = api.filter()
    rows = pipe(rows, filter(lambda x: x["state"] == "Done"), list)
    dataset = LabeledDataset(rows=rows, transforms=train_transforms)
    if len(rows) < 4:
        return
    for i in range(3):
        id, image, boxes, labels = dataset[i]
        plot = DetectionPlot(inv_normalize(image))
        plot.draw_boxes(boxes=boxes, labels=labels)
        plot.save(f"store/test-labeled-{id}.png")


def test_fold() -> None:
    train_rows = read_train_rows("/store")
    train, test = kfold(train_rows)
    train_seqs = set([r["sequence_id"] for r in train.values()])
    test_seqs = set([r["sequence_id"] for r in test.values()])

    assert len(train_seqs.intersection(test_seqs)) == 0


def test_find_prev_frame() -> None:
    rows = read_train_rows("/store")
    res = find_prev_frame(rows, sequence_id=5, frame_id=4)
    assert res is not None
    assert res["sequence_id"] == 5
    assert res["frame_id"] == 3

    res = find_prev_frame(rows, sequence_id=5, frame_id=0)
    assert res is None
