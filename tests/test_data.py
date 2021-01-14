from fish.data import (
    FileDataset,
    train_transforms,
    read_train_rows,
    read_test_rows,
    kfold,
    cutmix,
    find_prev_frame,
    FrameDataset,
    inv_normalize,
)
from toolz import valfilter
from object_detection.utils import DetectionPlot


def test_dataset() -> None:
    annotations = read_train_rows("/store")

    dataset = FileDataset(rows=annotations, transforms=train_transforms)
    for i in range(10):
        id, image, boxes, labels = dataset[5]
        plot = DetectionPlot(inv_normalize(image) * 255)
        plot.draw_boxes(boxes=boxes, labels=labels)
        plot.save(f"store/test-plot-{id}-{i}.png")


def test_frame_dataset() -> None:
    annotations = read_train_rows("/store")

    dataset = FrameDataset(rows=annotations, transforms=train_transforms)
    for i in range(10):
        id, image0, image1, boxes, labels = dataset[5]
        plot = DetectionPlot(image0)
        plot.draw_boxes(boxes=boxes, labels=labels)
        plot.save(f"store/test-plot-{id}-0-{i}.png")

        plot = DetectionPlot(image1)
        plot.draw_boxes(boxes=boxes, labels=labels)
        plot.save(f"store/test-plot-{id}-1-{i}.png")


def test_fold() -> None:
    train_rows = read_train_rows("/store")
    train, test = kfold(train_rows)
    train_seqs = set([r["sequence_id"] for r in train.values()])
    test_seqs = set([r["sequence_id"] for r in test.values()])

    assert len(train_seqs.intersection(test_seqs)) == 0


def test_cutmix() -> None:
    rows = read_train_rows("/store")
    rows = valfilter(lambda x: x["sequence_id"] == 0, rows)
    cutmix(rows)


def test_find_prev_frame() -> None:
    rows = read_train_rows("/store")
    res = find_prev_frame(rows, sequence_id=5, frame_id=4)
    assert res is not None
    assert res["sequence_id"] == 5
    assert res["frame_id"] == 3

    res = find_prev_frame(rows, sequence_id=5, frame_id=0)
    assert res is None
