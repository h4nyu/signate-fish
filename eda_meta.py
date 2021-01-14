import torch, matplotlib.pyplot as plt, pandas as pd
from fish.data import read_test_rows, read_train_rows
import numpy as np
from toolz.curried import map, pipe, groupby, valmap, concat
from torchvision.ops.boxes import box_area


test_rows = read_test_rows("/store")
train_rows = read_train_rows("/store")
train_seq = pipe(test_rows.values(), map(lambda x: x["sequence_id"]), set)
test_seq = pipe(train_rows.values(), map(lambda x: x["sequence_id"]), set)

# groups = pipe(test_rows.values(), groupby(lambda x: x["sequence_id"]))
# test_frames = pipe(groups, valmap(len))
# max_test_frame = max(test_frames.values())
# print(f"{max_test_frame=}")
# min_test_frame = min(test_frames.values())
# print(f"{min_test_frame=}")
# mean_test_frame = np.mean(np.fromiter(test_frames.values(), dtype=int))
# print(f"{mean_test_frame=}")


train_boxes = pipe(train_rows.values(), map(lambda x: x["boxes"]), concat, list)
# area = box_area(torch.tensor(train_boxes))
# fig = plt.figure()
# plt.hist(area.numpy(), bins=1000)
# fig.savefig("/store/box-hist.png")

box_df = pd.DataFrame(train_boxes, columns=["x0", "y0", "x1", "y1"])
box_df["w"] = box_df["x1"] - box_df["x0"]
box_df["h"] = box_df["y1"] - box_df["y0"]
box_df["area"] = box_df["w"] * box_df["h"]
print(box_df.describe())

# filtered_df = box_df[box_df["w"] > 10]
# print(filtered_df.describe())

# filtered_df = box_df[box_df["h"] > 10]
# print(filtered_df.describe())

filtered_df = box_df[box_df["area"] > 500]
print(filtered_df.describe())
