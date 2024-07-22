import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple

from src.state import TimeMixState, Shared
from .kv_cache import get_default_state

from src.rotary import generate_rotary_embedding, generate_binary_rotary_embedding, apply_rotary_embedding

from src.CoreDependencies import *

class TMix_llama(nn.Module):
    def get_default_state_factory(self): return get_default_state

    def __init__(self, args, layer_id):
        super().__init__()
        MyLinear = get_quantized_linear(args.quant)
        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        self.wq = MyLinear(args.n_embd, args.dim_att, bias=False)
        self.wk = MyLinear(args.n_embd, args.dim_att, bias=False)
        self.wv = MyLinear(args.n_embd, args.dim_att, bias=False)
        self.wo = MyLinear(args.dim_att, args.n_embd, bias=False)

    def forward(self, x, xo, kv_cache, last_state:TimeMixState, shared:Shared):
        B, L, D = x.size()
        H = self.n_head

        q = self.wq(x) 
        k = self.wk(x)
        v = self.wv(x)
        wkv_state = last_state.wkv_state

        # handle recurrent inference via maintaining a kv cache
        if not self.training:
            new_kv_cache = torch.stack([k, v], dim=0)
            wkv_state = torch.cat([wkv_state, new_kv_cache], dim=-2)
            k, v = wkv_state.unbind(0)
            k, v = k.contiguous(), v.contiguous()

        is_causal = q.size(1)==k.size(1)

        q = q.view(B,-1,H,D//H).transpose(1,2)
        k = k.view(B,-1,H,D//H).transpose(1,2)
        v = v.view(B,-1,H,D//H).transpose(1,2)
        q, k = apply_rotary_embedding(q, k, shared.angles)
        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)
        y = y.transpose(1,2).reshape(B,L,D)
        y = self.wo(y)
        return y, TimeMixState(wkv_state, last_state.shift_state)
