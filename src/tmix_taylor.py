import torch
from torch import nn, Tensor
from .CoreDependencies import *
from .cuda6 import RUN_CUDA_RWKV6

import math

class RWKV_Tmix_taylor(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.n_layer = args.n_layer

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
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
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            D_MIX_LORA = 32
            self.time_maa_rkvw_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*3))
            self.time_maa_rkvw_w2 = nn.Parameter(torch.zeros(3, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.decay = nn.Linear(args.n_embd, self.n_head, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_x = nn.LayerNorm(args.dim_att)

        self.ln_q = nn.LayerNorm(self.head_size)
        self.ln_k = nn.LayerNorm(self.head_size)
        self.ln_v = nn.LayerNorm(self.head_size)
        self.ln_out = nn.LayerNorm(args.n_embd)

    @MyFunction
    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head
        K = C // H
        V = C // H

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_rkvw_w1).view(B*T, 3, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_rkvw_w2).view(3, B, T, C)

        q, k, v = xxx.unbind(dim=0)
        xq = x + xx * (self.time_maa_r + q)
        xk = x + xx * (self.time_maa_k + k)
        xv = x + xx * (self.time_maa_v + v)
        q = self.receptance(xq).view(B,T,H,K).transpose(1,2)
        k = self.key(xk).view(B,T,H,K).transpose(1,2)
        v = self.value(xv).view(B,T,H,V).transpose(1,2)
        w = self.decay(xv).view(B,T,H,1).transpose(1,2)
        w = (-w.exp()).exp()

        # normalize each head
        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        dt = (torch.arange(T, device=q.device)[:, None] - torch.arange(T, device=q.device)[None, :]).tril() # NOTE - tril is important to not break pow by causing infinities
        w = w.pow(dt)
        w = w.tril() # causality

        attn = q @ k.mT
        attn = 1 + attn + 0.5 * attn.square() # taylor series approximation to exp
        attn = (attn * w).to(q.dtype)

        # NOTE - we may eventually want denominator, a la rwkv4
        #attn = attn / attn.sum(-1, keepdim=True).clamp(eps)
        x = attn @ v
        x = x.transpose(1,2).reshape(B,T,C)

        x = self.ln_x(x)

        x = self.output(x)

        x = self.ln_out(x) / math.sqrt(2 * self.n_layer)

        return x
