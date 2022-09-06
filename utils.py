import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def init_seed(seed: int):
    """시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False
