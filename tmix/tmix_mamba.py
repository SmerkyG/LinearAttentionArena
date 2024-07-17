import torch
from torch import nn, Tensor
import torch.nn.functional as F

import mamba_ssm

from src.tmix import TimeMixState, Shared

class TMix_Mamba(mamba_ssm.Mamba):
    def __init__(self, args, layer_id):
        super().__init__(d_model=args.n_embd, d_state=16, d_conv=4, expand=2)

    def forward(self, x, xo, kv_cache, last_state:TimeMixState, shared:Shared):
        return super().forward(x), last_state
