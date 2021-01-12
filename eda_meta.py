from fish.data import read_test_rows, read_train_rows
from toolz.curried import map, pipe


test_rows = read_test_rows("/store")
train_rows = read_train_rows("/store")
train_seq = pipe(test_rows.values(), map(lambda x: x["sequence_id"]), set)
test_seq = pipe(train_rows.values(), map(lambda x: x["sequence_id"]), set)
