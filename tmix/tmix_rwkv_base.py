import torch
from torch import Tensor

from src.tmix import TimeMixState
from configs import Transformer_Config

def get_default_state(x:Tensor, config:Transformer_Config, requires_grad:bool):
    B, T, C = x.size()
    return TimeMixState(
        torch.zeros([B, config.dim_att // config.head_size, config.head_size, config.head_size], dtype=x.dtype, device=x.device, requires_grad=requires_grad), 
        torch.zeros([B, C], dtype=x.dtype, device=x.device, requires_grad=requires_grad)
    )
