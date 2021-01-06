from fish.data import FileDataset, train_transforms
from fish.store import ImageStore
from object_detection.utils import DetectionPlot


def test_dataset() -> None:
    store = ImageStore("/store")
    annotations = store.read()

    dataset = FileDataset(rows=annotations, transforms=train_transforms(1080*2))
    for i in range(10):
        id, image, boxes, labels = dataset[5]
        plot = DetectionPlot(image)
        plot.draw_boxes(boxes=boxes, labels=labels)
        plot.save(f"store/test-plot-{id}-{i}.png")
