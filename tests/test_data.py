from fish.data import FileDataset
from fish.store import ImageStore
from object_detection.utils import DetectionPlot


def test_dataset() -> None:
    store = ImageStore("/store")
    annotations = store.read()

    dataset = FileDataset(rows=annotations)
    for i in range(10):
        id, image, boxes, labels = dataset[5]
        _, h, w = image.shape
        plot = DetectionPlot(w=w, h=h)
        plot.with_image(image)
        plot.with_pascal_boxes(boxes=boxes, labels=labels)
        plot.save(f"store/test-plot-{id}-{i}.png")
