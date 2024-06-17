import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .CoreDependencies import *
from typing import Tuple

from .tmix import TimeMixState
from .cmix import ChannelMixState

from .rotary import generate_rotary_embedding, generate_binary_rotary_embedding, apply_rotary_embedding

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

class Llama3_Tmix(MyModule):
    def __init__(self, args, layer_id, angles, bias_mask):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        self.wq = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.wk = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.wv = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.wo = nn.Linear(args.dim_att, args.n_embd, bias=False)

        #self.register_buffer('angles', generate_rotary_embedding(args.ctx_len * 2, self.head_size), persistent=False)
        #self.register_buffer('angles', torch.zeros(0))
        self.angles = angles
        self.bias_mask = bias_mask

    @MyFunction
    def forward(self, x, xo, last_timemix_state:TimeMixState):
        B, L, D = x.size()
        H = self.n_head

        #if getattr(self, 'angles', None) is None:
        #if self.angles.size(0) == 0:
        #    self.angles = generate_rotary_embedding(self.ctx_len * 2, self.head_size).to(device=x.device, dtype=x.dtype)

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
        if self.angles is not None:
            self.angles = self.angles.to(x.device)
            q, k = apply_rotary_embedding(q, k, self.angles)
        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=q.size(-2)==k.size(-2))
        y = y.transpose(1,2).reshape(B,L,D)
        y = self.wo(y)
        return y, TimeMixState(wkv_state, last_timemix_state.shift_state)
