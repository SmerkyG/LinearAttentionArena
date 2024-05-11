import torch
from torch import nn, Tensor
from .CoreDependencies import *

import math

from .tmix import TimeMixState

class RWKV_Tmix_taylorchunked(MyModule):
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
            self.time_maa_rkv_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*3))
            self.time_maa_rkv_w2 = nn.Parameter(torch.zeros(3, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.decay = nn.Linear(args.n_embd, self.n_head, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_x = nn.LayerNorm(args.dim_att)

        self.ln_q = nn.LayerNorm(self.head_size)
        self.ln_k = nn.LayerNorm(self.head_size)

    @MyFunction
    def forward(self, x, last_state:TimeMixState):
        B, T, C = x.size()
        H = self.n_head
        Q = C // H
        K = C // H
        V = C // H

        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_rkv_w1).view(B*T, 3, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_rkv_w2).view(3, B, T, C)

        q, k, v = xxx.unbind(dim=0)
        xq = x + dxprev * (self.time_maa_r + q)
        xk = x + dxprev * (self.time_maa_k + k)
        xv = x + dxprev * (self.time_maa_v + v)
        q = self.receptance(xq).view(B,T,H,K).transpose(1,2)
        k = self.key(xk).view(B,T,H,K).transpose(1,2)
        v = self.value(xv).view(B,T,H,V).transpose(1,2)
        w = self.decay(xv).view(B,T,H).transpose(1,2)
        #w = (-w.exp()).exp()
        #w_log = -((w - 2).clamp(None,20)).exp() # start at around 87%
        #w_log = w_log.clamp(math.log(0.005))
        #w_log = torch.ones_like(w_log) * 0.9 # FIXME
        w = torch.sigmoid(2.0 + w.float())
        w_log = (0.005 + 0.995 * w).log()

        # normalize each head
        q = self.ln_q(q)
        k = self.ln_k(k)

        # chunk
        L = T
        T = 512 #16
        N = L // T
        assert T * N == L, 'context length must be an even multiple of chunk size'
        q = q.view(B,H,N,T,Q)
        k = k.view(B,H,N,T,K)
        v = v.view(B,H,N,T,V)
        #w = w.view(B,H,N,T)
        #w_log = w_log.view(B,H,N,T,1)

        w_log_cumsum = w_log.cumsum(dim=-1).to(torch.float16)
        w_log_cumsum = w_log_cumsum.view(B,H,N,T,1) #.to(q.dtype)
        # w_chunk = w_log_cumsum[:,:,:,-1:,:].view(B,H,N,1,1) # decay across full chunk
        # w_inter = w_chunk - w_log_cumsum    # w1:4 = w0:4 - w0:1
        # w_intra = w_log_cumsum - w_log      # w1:3 = w0:3 - w0

        # #shifted_w_cumprod = F.pad(w_log_cumsum, (0, 0, 1, -1)).exp().view(B,H,N,T,K)
        # w_cumprod = w_log_cumsum.exp().view(B,H,N,T,1)
        # w_chunk = w_chunk.exp().to(k.dtype).view(B,H,N,1,1)
        # w_inter = w_inter.exp().to(k.dtype).view(B,H,N,T,1)
        # w_intra = w_intra.exp().to(k.dtype).view(B,H,N,T,1)

        # #w = w_intra.expand(B,H,N,T,T).masked_fill(torch.ones(T,T,dtype=torch.bool,device=k.device).triu(),1).tril()

        # # intra-chunk and v_first
        # attn = ((q * w_cumprod) @ (k / w_cumprod).mT).to(k.dtype).tril(-1) # + torch.eye(T, T).expand(B, T, T)
        # attn = q @ k.mT
        # attn = 1 + attn + 0.5 * attn.square() # taylor series approximation to exp
        # #attn = (attn * w).to(q.dtype)
        # # NOTE - we may eventually want denominator, a la rwkv4
        # #attn = attn / attn.sum(-1, keepdim=True).clamp(eps)
        # y = attn @ v + v

        # # inter-chunk        
        # wkv_state = last_state.wkv_state
        # wkv_states = []
        # wkv = ((k * w_inter).mT @ v).view(B,H,N,K,V)
        # wkv = list(wkv.unbind(dim=-3)) # N x BHKV
        # w_chunk = list(w_chunk.unbind(dim=-3))
        # for n in range(N):
        #     wkv_states.append(wkv_state)
        #     wkv_state = wkv_state * w_chunk[n].mT + wkv[n]
        # wkv_states = torch.stack(wkv_states, dim=2) # BHNKV       
        # y = y + (q * w_intra) @ wkv_states

        y = torch.zeros((B, H, N, T, V), device=q.device, dtype=q.dtype)
        for i in range(N):
            for j in range(i+1):
                qi, kj, vj = q[:,:,i], k[:,:,j], v[:,:,j]
                log_wi, log_wj = w_log_cumsum[:,:,i], w_log_cumsum[:,:,j]
                mask = (log_wi-log_wj.mT).clamp(None, 0).exp().to(q.dtype)
                if i == j:
                    mask = mask.tril()
                #print(float(mask.flatten().min()), float(mask.flatten().max()))
                #if i == j:
                #    mask = mask.masked_fill(torch.ones(T,T,dtype=torch.bool,device=k.device).triu(),1).tril()
                att = qi @ kj.mT
                #att = 1 + att + 0.5 * att.square()
                att = att.square()
                att = att * mask
                #if i == j:
                #    att = att.tril(-1)
                #if i == j:
                #    att = att.masked_fill(torch.ones(T,T,dtype=torch.bool,device=k.device).triu(),1).tril()
                y[:,:,i] += att @ vj #((qq @ kk).square() * mask) @ vv
        #y += v
                

        # dechunk
        y = y.view(B,H,L,V).to(x.dtype)
        
        y = y.transpose(1,2).reshape(B,L,C)

        y = self.ln_x(y)

        y = self.output(y)

        return y, TimeMixState(last_state.wkv_state, shift_state)
