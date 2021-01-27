# data
original_width = 3840
original_height = 2160
aspect_ratio = original_width / original_height
min_box_width = 10
min_box_height = 10
min_box_area = 200
normalize_mean = (0.485, 0.456, 0.406)
normalize_std = (0.485, 0.456, 0.406)
# image_width = 1024 + 512
image_width = 1024 + 1024
image_height = int(image_width / aspect_ratio)
scale = image_width / original_width
num_classes = 2
test_seq_ids = set(
    [
        13,
        48,
        88,
        105,
        175,
        121,
    ]
)
negative_seq_ids = set([0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13])
ignore_seq_ids = set([81, 82, 83, 84])

# to_boxes
to_box_limit = 20

# fold
n_splits = 8

# metrics
ap_iou = 0.3

pos_neg = 1238 / 243
