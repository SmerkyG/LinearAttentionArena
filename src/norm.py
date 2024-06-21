import torch
from torch import nn, Tensor
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim : int, weight_scaling : bool = True, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.dim = dim
        starting_scale = dim ** -0.5
        if weight_scaling:
            self.register_parameter("scale", nn.Parameter(torch.ones(dim) * starting_scale))
        else:
            self.scale = starting_scale

    def forward(self, x):
        assert(self.dim == x.size(-1))
        rms_norm = self.scale * x.norm(2, dim=-1, keepdim=True)
        return x / rms_norm.clamp(self.eps)
    
def rms_norm(x, eps:float = 1e-8):
    #return x * (x.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    rms_norm = (x.size(-1) ** -0.5) * x.norm(2, dim=-1, keepdim=True)
    return x / (rms_norm + eps)

class Norm(nn.Module):
    def __init__(self, dim : int, weight_scaling : bool = True, eps = 1e-8):
        super().__init__()
        self.eps = eps
        if weight_scaling:
            self.register_parameter("scale", nn.Parameter(torch.ones(dim)))
        else:
            self.scale = 1

    def forward(self, x):
        return self.scale * x / x.norm(2, dim=-1, keepdim=True).clamp(self.eps)

def l2_norm(x, eps:float = 1e-8):
    # assumes that vector 'normally' has length 1, not length vec.size(-1)**0.5 (which would be if every component had an average absolute value of 1!)
    return x / (x.norm(2, dim=-1, keepdim=True) + eps)
