# data
original_width = 3840
original_height = 2160
aspect_ratio = original_width / original_height
min_box_width = 10
min_box_height = 10
min_box_area = 200
normalize_mean = (0.485, 0.456, 0.406)
normalize_std = (0.485, 0.456, 0.406)
image_size = 1024 + 512
scale = image_size / original_width

# fold
n_splits = 8
