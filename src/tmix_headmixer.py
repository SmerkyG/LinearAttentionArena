import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .CoreDependencies import *
from .cuda6 import RUN_CUDA_RWKV6

from .tmix import TimeMixState, Shared

import math

from .rotary import generate_rotary_embedding, generate_binary_rotary_embedding, apply_rotary_embedding
from .norm import rms_norm

class RWKV_Tmix_headmixer(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.n_layer = args.n_layer

        self.k_head_size = self.v_head_size = self.head_size = args.head_size
        self.n_kv_head = self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            D_MIX_LORA = 32
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*3))
            self.time_maa_w2 = nn.Parameter(torch.zeros(3, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_r = nn.LayerNorm(args.dim_att)
        self.ln_k = nn.LayerNorm(args.dim_att)
        self.ln_v = nn.LayerNorm(args.dim_att)
        self.ln_x = nn.LayerNorm(args.dim_att)
        
        D_HEADAVG_LORA = self.n_head * 2
        self.headavg_w1 = nn.Parameter(torch.zeros(args.n_embd, D_HEADAVG_LORA*4))
        self.headavg_w2 = nn.Parameter(torch.zeros(4, D_HEADAVG_LORA, self.n_head).uniform_(-0.01, 0.01))

        self.bias_mask = AlibiMask(args.ctx_len, self.n_kv_head, layer_id)

    def batch_lora(self, xw1, w2): 
        B,T,Ctotal = xw1.shape
        n_bound = w2.size(0)
        assert Ctotal % n_bound == 0
        return (xw1.view(B*T,n_bound,-1).transpose(0,1) @ w2).view(n_bound,B,T,-1)

    def forward(self, x, xo, kv_cache, last_state:TimeMixState, shared:Shared):
        B, T, C = x.size()
        H = self.n_head
        K = C // H
        V = C // H

        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x

        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, self.time_maa_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), B, T, C)

        mq, mk, mv = xxx.unbind(dim=0)
        xq = x + dxprev * (self.time_maa_r + mq)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        
        q = self.receptance(xq)
        k = self.key(xk)
        v = self.value(xv)
        
        q = self.ln_r(q).view(B,-1,H,K).transpose(1,2)
        k = self.ln_k(k).view(B,-1,H,K).transpose(1,2)
        v = self.ln_v(v)
        v = v.view(B,-1,H,V).transpose(1,2)

        # x = nn.functional.scaled_dot_product_attention(
        #     r, k, v,
        #     attn_mask=self.bias_mask(r), dropout_p=0.0, is_causal=self.bias_mask is None)
        #     #is_causal=True)

        pre_q, pre_k, post_q, post_k = ( self.batch_lora(torch.tanh(x @ self.headavg_w1), self.headavg_w2).view(4,B,T,H) / self.n_head ).unbind(0)

        scale = K ** -0.5
        y = (q * scale) @ k.mT
        y = y + torch.einsum('BHTS,BTH->BTS', y, pre_q).unsqueeze(1)
        y = y + torch.einsum('BHTS,BSH->BTS', y, pre_k).unsqueeze(1) 
        y = y + self.bias_mask(q)
        y = F.softmax(y, dim=-1)
        y = y + torch.einsum('BHTS,BTH->BTS', y, post_q).unsqueeze(1)
        y = y + torch.einsum('BHTS,BSH->BTS', y, post_k).unsqueeze(1) 
        y = y @ v

        y = y.transpose(1,2).reshape(B,T,C)
       
        y = self.ln_x(y)

        y = self.output(y)

        return y, TimeMixState(last_state.wkv_state, shift_state)

