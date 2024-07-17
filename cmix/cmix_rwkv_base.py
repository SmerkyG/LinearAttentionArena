import torch
from torch import Tensor, nn

from src.state import ChannelMixState
from configs import Transformer_Config

def get_default_state(x:Tensor, config:Transformer_Config, requires_grad:bool):
    B, T, C = x.size()
    return ChannelMixState(
        torch.zeros([B, C], dtype=x.dtype, device=x.device, requires_grad=requires_grad)
    )
