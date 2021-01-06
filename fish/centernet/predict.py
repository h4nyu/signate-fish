import glob, tqdm, torch
from torch.utils.data import DataLoader
from object_detection.utils import DetectionPlot
from object_detection.entities.box import yolo_to_pascal
from fish.data import read_test_rows, TestDataset, prediction_transforms
from fish.centernet import config
from fish.centernet.train import model, model_loader, to_boxes



@torch.no_grad()
def predict(device: str) -> None:
    rows = read_test_rows("/store")
    dataset = TestDataset(rows=rows, transforms=prediction_transforms(config.image_size))
    net = model_loader.load_if_needed(model).to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size * 3,
        num_workers=config.batch_size,
        shuffle=False,
    )

    for ids, image_batch in tqdm.tqdm(loader):
        image_batch = image_batch.to(device)
        netout = net(image_batch)
        box_batch, confidence_batch, label_batch = to_boxes(netout)

        for img, boxes, confidences, labels, id in zip(
            image_batch, box_batch, confidence_batch, label_batch, ids
        ):
            _, h, w = img.shape
            plot = DetectionPlot(img)
            plot.draw_boxes(boxes=yolo_to_pascal(boxes, (w, h)), confidences=confidences, labels=labels)
            plot.save(f"/store/pred-{id}.jpg")


if __name__ == "__main__":
    predict("cuda")
