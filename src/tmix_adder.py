import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .CoreDependencies import *

from .tmix import TimeMixState

import math

class RWKV_Tmix_adder(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        self.dim_v = args.dim_att
        self.dim_k = args.dim_att

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_q = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v_first = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            D_MIX_LORA = 32
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*self.time_maa_w2.size(0)))

            decay_speed = torch.ones(self.dim_k) * 2
            #for n in range(self.dim_k):
            #    decay_speed[n] = -6 + 5 * (n / (self.dim_k - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,self.dim_k))
            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, self.dim_k).uniform_(-0.01, 0.01))

            self.time_value2_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_value2_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, self.dim_v).uniform_(-0.01, 0.01))

            # tmp = torch.zeros(self.dim_k)
            # for n in range(self.dim_k):
            #     zigzag = ((n + 1) % 3 - 1) * 0.1
            #     tmp[n] = ratio_0_to_1 * (1 - (n / (self.dim_k - 1))) + zigzag
            # self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.query = nn.Linear(args.n_embd, self.dim_k, bias=False)
        self.key = nn.Linear(args.n_embd, self.dim_k, bias=False)

        self.value = nn.Linear(args.n_embd, self.dim_v, bias=False)
        self.output = nn.Linear(self.dim_v, args.n_embd, bias=False)
        self.ln_x = nn.LayerNorm(self.dim_v)

    @MyFunction
    def forward(self, x, last_state:TimeMixState):
        B, T, C = x.size()
        H = self.n_head
        Q = self.dim_k // H
        K = self.dim_k // H
        V = self.dim_v // H

        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, self.time_maa_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), B, T, C)

        mr, mk, mv, mw, mv_first = xxx.unbind(dim=0)
        xq = x + dxprev * (self.time_maa_q + mr)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xw = x + dxprev * (self.time_maa_w + mw)
        xv_first = x + dxprev * (self.time_maa_v_first + mv_first)

        # o = c @ h + v_first
        # h = a * h + b.mT * (1-a) * v

        # c @ (a_cum * b.mT) @ ((1-a_cum)*v)

        # c @ b.mT @ v

        # rope(q) @ rope(k).mT @ v

        q = self.query(xq)
        k = self.key(xk)
        v = self.value(xv)
        v_first = self.value(xv_first) + torch.tanh(xw @ self.time_value2_w1) @ self.time_value2_w2
        #w_log = -torch.exp(self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2)
        #w_log = w_log.clamp(-5, 0)
        #w_log = w_log.clamp(math.log(0.005))
        #w = w_log.exp()
        w = 0.005 + 0.995 * F.sigmoid(self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2)
        k = (k * (1 - w)).to(k.dtype)
        w_log = w.log()

        q = q.view(B,T,H,Q).transpose(1,2).view(B,H,T,Q)
        k = k.view(B,T,H,K).transpose(1,2).view(B,H,T,K)
        v = v.view(B,T,H,V).transpose(1,2).view(B,H,T,V)
        v_first = v_first.view(B,T,H,V).transpose(1,2).view(B,H,T,V)

        # chunk
        L = T
        T = 16
        N = L // T
        assert T * N == L, 'context length must be an even multiple of chunk size'
        q = q.view(B,H,N,T,Q)
        k = k.view(B,H,N,T,K)
        v = v.view(B,H,N,T,V)
        v_first = v_first.view(B,H,N,T,V)
        w_log = w_log.view(B,H,N,T,K)

        w_log_cumsum = w_log.cumsum(dim=-2).view(B,H,N,T,K)
        w_chunk = w_log_cumsum[:,:,:,-1:,:].view(B,H,N,1,K) # decay across full chunk
        w_inter = w_chunk - w_log_cumsum    # w1:4 = w0:4 - w0:1
        w_intra = w_log_cumsum - w_log      # w1:3 = w0:3 - w0

        #shifted_w_cumprod = F.pad(w_log_cumsum, (0, 0, 1, -1)).exp().view(B,H,N,T,K)
        w_cumprod = w_log_cumsum.exp().view(B,H,N,T,K)
        w_chunk = w_chunk.exp().to(k.dtype).view(B,H,N,1,K)
        w_inter = w_inter.exp().to(k.dtype).view(B,H,N,T,K)
        w_intra = w_intra.exp().to(k.dtype).view(B,H,N,T,K)

        # intra-chunk and v_first
        att = ((q * w_cumprod) @ (k / w_cumprod).mT).to(k.dtype).tril(-1) # + torch.eye(T, T).expand(B, T, T)
        y = att @ v + v_first

        # inter-chunk        
        wkv_state = last_state.wkv_state
        wkv_states = []
        wkv = ((k * w_inter).mT @ v).view(B,H,N,K,V)
        wkv = list(wkv.unbind(dim=-3)) # N x BHKV
        w_chunk = list(w_chunk.unbind(dim=-3))
        for n in range(N):
            wkv_states.append(wkv_state)
            wkv_state = wkv_state * w_chunk[n].mT + wkv[n]
        wkv_states = torch.stack(wkv_states, dim=2) # BHNKV       
        y = y + (q * w_intra) @ wkv_states

        # dechunk
        y = y.view(B,H,L,V).to(x.dtype)
        wkv_state = wkv_state.view(B,H,K,V)

        y = y.transpose(1,2).reshape(B,L,H*V)

        y = self.ln_x(y)
        y = self.output(y)
        return y, TimeMixState(wkv_state, shift_state)
