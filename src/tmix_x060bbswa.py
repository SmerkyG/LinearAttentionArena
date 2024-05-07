import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .CoreDependencies import *
from .cuda6 import RUN_CUDA_RWKV6

from .tmix import TimeMixState

def causal_bias_mask(T):
    return torch.full((T, T), float('-inf')).triu(1)

def alibi_mask(T, H):
    bias = (torch.arange(T)[None, :] - torch.arange(T)[:, None]).float() # (T, T)
    bias = bias + causal_bias_mask(T) # (T, T)
    bias = bias.expand(H, -1, -1) # (H, T, T)
    head_bias_slopes = (2 ** torch.linspace(-8.0/H, -8.0, H)).unsqueeze(-1).unsqueeze(-1) # (H, 1, 1)
    bias = bias * head_bias_slopes # (H, T, T)
    return bias

class AlibiMask(nn.Module):
    def __init__(self, block_size : int, n_heads : int, layer_id : int):
        super().__init__()
        T = block_size
        H = n_heads
        self.register_buffer('mask', alibi_mask(T, H))

    def forward(self, q:Tensor):
        return self.mask[:, :q.size(-2), :q.size(-2)]

class RWKV_Tmix_x060bbswa(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

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
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            D_MIX_LORA = 32
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*4))
            self.time_maa_w2 = nn.Parameter(torch.zeros(4, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_r = nn.LayerNorm(args.dim_att)
        self.ln_k = nn.LayerNorm(args.dim_att)
        #self.ln_v = nn.LayerNorm(args.dim_att)
        self.ln_x = nn.LayerNorm(args.dim_att)
        self.ln_x2 = nn.LayerNorm(args.dim_att)
        self.ln_x3 = nn.LayerNorm(args.dim_att)

        self.bias_mask = AlibiMask(args.ctx_len, self.n_kv_head, layer_id)

    @MyFunction
    def forward(self, x, last_state:TimeMixState):
        B, T, C = x.size()
        H = self.n_head
        K = C // H
        V = C // H

        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x

        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 4, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(4, B, T, C)

        mr, mk, mv, mw = xxx.unbind(dim=0)
        xr = x + dxprev * (self.time_maa_r + mr)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xw = x + dxprev * (self.time_maa_w + mw)
        
        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        w = self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        u = self.time_faaaa

        lr = self.ln_r(r)
        lk = self.ln_k(k)
        lv = v #self.ln_v(v)

        L = T
        T = 64
        Z = 2 # min_blocks_seen
        blocks = []
        #wkv_state = last_state.wkv_state.clone()
        wkv_state = last_state.wkv_state.to(torch.bfloat16).contiguous()
        for e in range(T, L+1, T): # end of latest block
            b = max(0, e-T*(Z+1)) # beginning of earliest block
            # FIXME - maybe support dropout
            y = nn.functional.scaled_dot_product_attention(
                lr[:,b:e].view(B,-1,H,K).transpose(1,2),
                lk[:,b:e].view(B,-1,H,K).transpose(1,2),
                lv[:,b:e].view(B,-1,H,V).transpose(1,2),
                attn_mask=self.bias_mask(r[:,b:e]), dropout_p=0.0, is_causal=self.bias_mask is None)
            y = y[:,:,-T:].transpose(1,2).reshape(B,T,C) # FIXME - inefficient to recalc extra blocks each time, but this is just a proof of concept
            # if b >= T:
            # if there's at least one whole block prior to b, apply it to the output via linear attention, and update the linear attention state
            # r, k, v, w = lr[:,e-T:e], lk[:,b-T:b], lv[:,b-T:b], lw[:,b-T:b]
            # wkv_state = wkv_state.clone()
            # x = x + RUN_CUDA_RWKV6(B, T, C, H, r.contiguous(), k.contiguous(), v.contiguous(), w.contiguous(), u, wkv_state)
            blocks.append(y)
        y = torch.cat(blocks, dim=-2)

        #r, k, v = self.receptance(x[:,T*Z:]), self.key(x[:,:-T*Z]), self.value(x[:,:-T*Z])
        w = self.time_decay + k
        tau = 9
        # w in log-log space, securely clamped
        w = -nn.functional.elu(-w+tau)+tau
        k = (1 - ( (-w.exp()).exp() )).to(r)
        wkv_state = wkv_state.clone()
        x = RUN_CUDA_RWKV6(B, L, C, H, r.contiguous(), k.contiguous(), v.contiguous(), w.contiguous(), torch.zeros_like(u), wkv_state)
        
        x = self.ln_x3(self.ln_x(x) + self.ln_x2(y))
        x = self.output(x)
        return x, TimeMixState(wkv_state, shift_state)

