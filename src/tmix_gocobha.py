import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .CoreDependencies import *
from .cuda6 import RUN_CUDA_RWKV6

from .tmix import TimeMixState

import math

from typing import Tuple

from .rotary import generate_rotary_embedding, generate_binary_rotary_embedding, apply_rotary_embedding
from .norm import rms_norm

class GPTAlpha_Tmix_gocobha(MyModule):
    def __init__(self, args, layer_id, angles, bias_mask):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.n_layer = args.n_layer

        self.k_head_size = self.v_head_size = self.head_size = args.head_size_a
        self.n_kv_head = self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_q = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            self.time_maa_v_cache = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_h = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            D_MIX_LORA = 32
            self.time_maa_q_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA))
            self.time_maa_q_w2 = nn.Parameter(torch.empty(D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))
            self.time_maa_kv_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*2))
            self.time_maa_kv_w2 = nn.Parameter(torch.empty(2, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            D_VALUE_LORA = max(args.n_embd // 16, 64)
            self.time_key_w1 = nn.Parameter(torch.zeros(args.n_embd, D_VALUE_LORA))
            self.time_key_w2 = nn.Parameter(torch.zeros(D_VALUE_LORA, args.dim_att).uniform_(-0.01, 0.01))
            self.time_value_w1 = nn.Parameter(torch.zeros(args.n_embd, D_VALUE_LORA))
            self.time_value_w2 = nn.Parameter(torch.zeros(D_VALUE_LORA, args.dim_att).uniform_(-0.01, 0.01))
            #MLA_FACTOR = 16
            #self.key = nn.Linear(args.n_embd // MLA_FACTOR + args.n_embd, args.dim_att, bias=False)

        self.query = nn.Linear(args.n_embd, args.dim_att, bias=False)
        #self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        #self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_q = nn.LayerNorm(args.dim_att)
        self.ln_k = nn.LayerNorm(args.dim_att)
        self.ln_v = nn.LayerNorm(args.dim_att)
        self.ln_x = nn.LayerNorm(args.dim_att)

        self.time_headmix = nn.Parameter(torch.zeros(2, args.n_embd))
        D_HEADMIX_LORA = args.n_embd // 12
        self.time_headmix_w1 = nn.Parameter(torch.zeros(args.n_embd, D_HEADMIX_LORA*2))
        self.time_headmix_w2 = nn.Parameter(torch.zeros(2, D_HEADMIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.angles = angles
        self.bias_mask = bias_mask

    def batch_lora(self, xw1, w2): 
        B,T,Ctotal = xw1.shape
        n_bound = w2.size(0)
        assert Ctotal % n_bound == 0
        return (xw1.view(B*T,n_bound,-1).transpose(0,1) @ w2).view(n_bound,B,T,-1)

    @MyFunction
    def forward(self, x, xo, k_cache, last_time_mix_state:TimeMixState):

        B, T, C = x.size()
        H = self.n_head
        K = C // H
        V = C // H

        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_time_mix_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x

        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, self.time_maa_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), B, T, C)
        mq, mh = xxx.unbind(dim=0)

        xo = rms_norm(xo)
        dxo_prev = self.time_shift(xo) - xo
        xxx = xo + dxo_prev * self.time_maa_v_cache
        xxx = torch.tanh(xxx @ self.time_maa_kv_w1).view(B*xo.size(1), self.time_maa_kv_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_kv_w2).view(self.time_maa_kv_w2.size(0), B, xo.size(1), C)
        mk, mv = xxx.unbind(dim=0)

        k = k_cache
        dkprev = self.time_shift(k) - k
        v = xo
        dvprev = self.time_shift(v) - v

        xq = x + dxprev * (self.time_maa_q + mq)
        xh = x + dxprev * (self.time_maa_h + mh)

        k = k + dkprev * (self.time_maa_k + mk)
        v = v + dvprev * (self.time_maa_v + mv)
        
        q = self.receptance(xq)
        
        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)


        n_sha_heads = H // 4
        mha = F.scaled_dot_product_attention(q.view(B,T,H,K).transpose(1,2), k.view(B,T,H,K).transpose(1,2), v.view(B,T,H,V).transpose(1,2), is_causal=True)
        sha = F.scaled_dot_product_attention(q.view(B,T,n_sha_heads,-1).transpose(1,2), k.view(B,T,n_sha_heads,-1).transpose(1,2), v.view(B,T,n_sha_heads,-1).transpose(1,2), is_causal=True)

        headmix_mha, headmix_sha = (self.time_headmix.view(2,1,1,C) + self.batch_lora(torch.tanh(xh @ self.time_headmix_w1), self.time_headmix_w2).view(2,B,T,C)).unbind(0)

        x = mha.transpose(1,2).reshape(B,T,C) * (1.0 + headmix_mha) + sha.transpose(1,2).reshape(B,T,C) * headmix_sha

        x = self.ln_x(x)
        #x = F.layer_norm(x.float(), self.ln_x.normalized_shape, self.ln_x.weight.float(), self.ln_x.bias.float()).to(x.dtype)

        x = self.output(x)

        return x, TimeMixState(last_time_mix_state.wkv_state, shift_state)

