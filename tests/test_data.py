from fish.data import FileDataset
from fish.store import ImageStore


def test_dataset() -> None:
    store = ImageStore("/store")
    annotations = store.read()

    dataset = FileDataset(rows=annotations)
    sample = dataset[0]
    print(sample)
