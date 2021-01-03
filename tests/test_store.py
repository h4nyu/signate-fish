from fish.store import ImageStore


def test_image_store() -> None:
    store = ImageStore("/store")
    annotations = store.read()
    assert len(annotations) == 3387
    id, value = next(iter(annotations.items()))
    assert id == "train_0168"
    assert len(value["boxes"]) == len(value["labels"])
    assert value["image_path"] == "/store/train_images/train_0168.jpg"
