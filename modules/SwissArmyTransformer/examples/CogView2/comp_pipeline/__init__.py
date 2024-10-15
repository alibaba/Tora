from .base_completion import BaseCompletion
from .patch_completion import PatchCompletion

import torch
def cord2mask(h0, w0, h1, w1, size=480, device=None):
    assert 0 <= h0 <= 1
    ret = torch.zeros(size, size, dtype=torch.bool, device=device)
    ret[int(size*h0):int(size*h1), int(size*w0):int(size*w1)] = True
    return ret