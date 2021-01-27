import torch, matplotlib.pyplot as plt, pandas as pd
from fish.data import read_test_rows
import numpy as np
from fish import config
from toolz.curried import map, pipe, groupby, valmap, concat, filter, count
from torchvision.ops.boxes import box_area

rows = pipe(
    read_test_rows("/store").values(),
    list,
)
has_box_count = pipe(
    rows, filter(lambda x: x["sequence_id"] not in config.negative_seq_ids), count
)
print(has_box_count)
no_box_count = pipe(
    rows, filter(lambda x: x["sequence_id"] in config.negative_seq_ids), count
)
print(no_box_count)
