import torch
from torch import Tensor

from src.state import TimeMixState
from configs import Transformer_Config

def get_default_state(x:Tensor, config:Transformer_Config, requires_grad:bool):
    B, T, C = x.size()
    return TimeMixState(
        torch.zeros([2, B, 0, C], dtype=x.dtype, device=x.device, requires_grad=requires_grad), 
        torch.tensor([]),
    )
