import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple

from src.state import TimeMixState, Shared

from src.rotary import generate_rotary_embedding, generate_binary_rotary_embedding, apply_rotary_embedding

class TMix_Llama(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        self.wq = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.wk = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.wv = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.wo = nn.Linear(args.dim_att, args.n_embd, bias=False)

    def forward(self, x, xo, kv_cache, last_timemix_state:TimeMixState, shared:Shared):
        B, L, D = x.size()
        H = self.n_head

        q = self.wq(x) 
        k = self.wk(x)
        v = self.wv(x)
        wkv_state = last_timemix_state.wkv_state
        if not self.training:
            new_kv_cache = torch.cat([k, v], dim=-1)
            wkv_state = torch.cat([wkv_state, new_kv_cache], dim=-2)
            k, v = wkv_state.chunk(2, dim=-1)
            k, v = k.contiguous(), v.contiguous()
        # if self.layer_id==0:
        #     print("\nq,k,kv", q.shape, k.shape, wkv_state.shape)
        q = q.view(B,-1,H,D//H).transpose(1,2)
        k = k.view(B,-1,H,D//H).transpose(1,2)
        v = v.view(B,-1,H,D//H).transpose(1,2)
        q, k = apply_rotary_embedding(q, k, shared.angles)
        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=q.size(-2)==k.size(-2))
        y = y.transpose(1,2).reshape(B,L,D)
        y = self.wo(y)
        return y, TimeMixState(wkv_state, last_timemix_state.shift_state)
