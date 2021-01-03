from fish.data import FileDataset
from fish.store import ImageStore
from object_detection.utils import DetectionPlot


def test_dataset() -> None:
    store = ImageStore("/store")
    annotations = store.read()

    dataset = FileDataset(rows=annotations, mode="test")
    id, image, boxes, labels = dataset[2]
    _, h, w = image.shape
    plot = DetectionPlot(w=w, h=h)
    plot.with_image(image)
    plot.with_pascal_boxes(boxes=boxes, labels=labels)
    plot.save(f"store/test-plot-{id}.png")
