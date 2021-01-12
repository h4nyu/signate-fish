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
stream_handler = StreamHandler()
handler_format = Formatter("%(asctime)s,%(name)s,%(message)s")
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

init_seed(0)
