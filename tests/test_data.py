from fish.data import FileDataset, train_transforms, read_annotations
from object_detection.utils import DetectionPlot


def test_dataset() -> None:
    annotations = read_annotations("/store")

    dataset = FileDataset(rows=annotations, transforms=train_transforms(1080 * 2))
    for i in range(10):
        id, image, boxes, labels = dataset[5]
        plot = DetectionPlot(image)
        plot.draw_boxes(boxes=boxes, labels=labels)
        plot.save(f"store/test-plot-{id}-{i}.png")
