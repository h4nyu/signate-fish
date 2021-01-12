from fish.data import (
    FileDataset,
    train_transforms,
    read_train_rows,
    read_test_rows,
    kfold,
    cutmix
)
from toolz import valfilter
from object_detection.utils import DetectionPlot


def test_dataset() -> None:
    annotations = read_train_rows("/store")

    dataset = FileDataset(rows=annotations, transforms=train_transforms(1080 * 2))
    for i in range(10):
        id, image, boxes, labels = dataset[5]
        plot = DetectionPlot(image)
        plot.draw_boxes(boxes=boxes, labels=labels)
        plot.save(f"store/test-plot-{id}-{i}.png")


def test_fold() -> None:
    train_rows = read_train_rows("/store")
    train, test = kfold(train_rows)
    train_seqs = set([r["sequence_id"] for r in train.values()])
    test_seqs = set([r["sequence_id"] for r in test.values()])

    assert len(train_seqs.intersection(test_seqs)) == 0

def test_cutmix() -> None:
    rows = read_train_rows("/store")
    rows = valfilter(lambda x:x['sequence_id'] == 0, rows)
    cutmix(rows)
