from logging import (
    getLogger,
    StreamHandler,
    Formatter,
    INFO,
    FileHandler,
)
import torch
import numpy as np


__version__ = "0.1.0"
logger = getLogger()
logger.setLevel(INFO)
stream_handler = StreamHandler()
handler_format = Formatter("%(asctime)s,%(name)s,%(message)s")
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
