# data
original_width = 3840
original_height = 2160
aspect_ratio = original_width / original_height
min_box_width = 10
min_box_height = 10
min_box_area = 200
normalize_mean = (0.485, 0.456, 0.406)
normalize_std = (0.485, 0.456, 0.406)
# image_size = 1024 * 2
image_size = 512 + 1024
scale = image_size / original_width
num_classes = 2

# fold
n_splits = 5
