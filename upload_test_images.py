import base64
from io import BytesIO
from fish.store import StoreApi
from fish.data import read_test_rows
from PIL import Image as PILImage

rows = read_test_rows("/store")
api = StoreApi()

for id, row in rows.items():
    with open(row['image_path'], "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data)
    try:
        a = api.create(id, encoded)
    except:
        continue
