from logging import (
    getLogger,
    StreamHandler,
    Formatter,
    INFO,
    FileHandler,
)
import torch
import numpy as np
from object_detection.utils import init_seed

__version__ = "0.1.0"
logger = getLogger()
logger.setLevel(INFO)
handler_format = Formatter("%(asctime)s,%(name)s,%(message)s")
fh = FileHandler("app.log")
sh = StreamHandler()
sh.setFormatter(handler_format)
fh.setFormatter(handler_format)
logger.addHandler(sh)
logger.addHandler(fh)

init_seed(0)
