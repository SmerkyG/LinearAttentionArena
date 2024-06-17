import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .CoreDependencies import *
from typing import Tuple

from .tmix import TimeMixState
from .cmix import ChannelMixState

class Llama3_CMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args

        self.dim_ffn = args.n_embd * 4 * 2 // 3 // 32 * 32

        self.w1 = nn.Linear(args.n_embd, self.dim_ffn, bias=False)
        self.w2 = nn.Linear(self.dim_ffn, args.n_embd, bias=False)
        self.w3 = nn.Linear(args.n_embd, self.dim_ffn, bias=False)

    @MyFunction
    def forward(self, x, last_state:ChannelMixState):
        return self.w2(F.silu(self.w1(x)) * self.w3(x)), last_state

def generate_rotary_embedding(max_seqlen:int, dim:int, theta:float = 10000.0):
    angular_velocity = theta ** -(torch.arange(0, dim, 2, dtype=torch.float) / dim) # frequencies from 1.0 ... 1/theta
    angles = torch.outer(torch.arange(max_seqlen), angular_velocity)
    return torch.polar(torch.ones_like(angles), angles)

def apply_rotary_embedding(q, k, angles, seq_dim:int = -2) -> Tuple[Tensor, Tensor]:
    q_dtype, k_dtype = q.dtype, k.dtype
    L = q.size(seq_dim)
    angles = angles[-L:].view(1, 1, L, angles.size(1))
    q = torch.view_as_complex(q.float().reshape(q.size(0), q.size(1), q.size(2), -1, 2)) * angles
    k = torch.view_as_complex(k.float().reshape(k.size(0), k.size(1), k.size(2), -1, 2)) * angles
    return torch.view_as_real(q).flatten(3).to(q_dtype), torch.view_as_real(k).flatten(3).to(k_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, max_sequence_length:int, dim:int, seq_dim:int = -2, theta:float = 10000):
        super().__init__()
        self.angles = generate_rotary_embedding(max_sequence_length, dim, theta)
        self.seq_dim = seq_dim

    def forward(self, q, k):
        return apply_rotary_embedding(q, k, self.angles, self.seq_dim)

class Llama3_Tmix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        self.wq = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.wk = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.wv = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.wo = nn.Linear(args.dim_att, args.n_embd, bias=False)

        self.angles = generate_rotary_embedding(args.ctx_len * 2, self.head_size)

    @MyFunction
    def forward(self, x, xo, last_timemix_state:TimeMixState):
        B, L, D = x.size()
        H = self.n_head

        q = self.wq(x).view(B,L,H,-1).transpose(1,2)
        k = self.wk(x).view(B,L,H,-1).transpose(1,2)
        v = self.wv(x).view(B,L,H,-1).transpose(1,2)
        q, k = apply_rotary_embedding(q, k, self.angles)
        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1,2).reshape(B,L,D)
        y = self.wo(y)
        return y, TimeMixState(wkv_state, last_timemix_state.shift_state)
