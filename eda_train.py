import torch, matplotlib.pyplot as plt, pandas as pd
from fish.data import read_test_rows, read_train_rows
import numpy as np
from fish import config
from toolz.curried import map, pipe, groupby, valmap, concat
from torchvision.ops.boxes import box_area


test_rows = read_test_rows("/store")
train_rows = read_train_rows("/store")
train_seq = pipe(test_rows.values(), map(lambda x: x["sequence_id"]), set)
test_seq = pipe(train_rows.values(), map(lambda x: x["sequence_id"]), set)


labels = pipe(
    train_rows.values(),
    map(lambda x: set(x["labels"])),
    groupby(lambda x: str(x)),
    valmap(lambda x: len(x)),
)
print(labels)

# train_boxes = pipe(train_rows.values(), map(lambda x: x["boxes"]), concat, list)
# area = box_area(torch.tensor(train_boxes))
# fig = plt.figure()
# plt.hist(area.numpy(), bins=100)
# fig.savefig("/store/box-hist.png")

# box_df = pd.DataFrame(train_boxes, columns=["x0", "y0", "x1", "y1"])
# print(f"{config.scale=}")
# box_df = box_df * config.scale
# box_df["w"] = box_df["x1"] - box_df["x0"]
# box_df["h"] = box_df["y1"] - box_df["y0"]
# box_df["area"] = box_df["w"] * box_df["h"]
# box_df["aspect"] = box_df["w"] / box_df["h"]
# print(f"{box_df.describe()}")
# print(f"{box_df['area'].median() **(1/2)=}")
# print(f"{(box_df['area'].median() * 4) **(1/2)=}")
