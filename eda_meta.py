from fish.data import read_test_rows, read_train_rows
import numpy as np
from toolz.curried import map, pipe, groupby, valmap


test_rows = read_test_rows("/store")
train_rows = read_train_rows("/store")
train_seq = pipe(test_rows.values(), map(lambda x: x["sequence_id"]), set)
test_seq = pipe(train_rows.values(), map(lambda x: x["sequence_id"]), set)

groups = pipe(test_rows.values(), groupby(lambda x: x["sequence_id"]))
test_frames = pipe(groups, valmap(len))
max_test_frame = max(test_frames.values())
print(f"{max_test_frame=}")
min_test_frame = min(test_frames.values())
print(f"{min_test_frame=}")
mean_test_frame = np.mean(np.fromiter(test_frames.values(), dtype=int))
print(f"{mean_test_frame=}")
