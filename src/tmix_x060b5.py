import torch
from torch import nn, Tensor
from .CoreDependencies import *
from .cuda6 import RUN_CUDA_RWKV6

from .tmix import TimeMixState, Shared

class RWKV_Tmix_x060b5(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.dim_att)
            for i in range(args.dim_att):
                ddd[0, 0, i] = i / args.dim_att

            #self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            # self.time_maa_all = nn.Parameter(torch.cat([
            #     1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0), # r
            #     1.0 - torch.pow(ddd, ratio_1_to_almost0), # k
            #     1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1), # v
            #     1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1), # v2
            #     1.0 - torch.pow(ddd, ratio_1_to_almost0), # w
            # ]))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_v2 = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            D_MIX_LORA = 32
            self.time_maa_w2 = nn.Parameter(torch.zeros(4, D_MIX_LORA, args.dim_att).uniform_(-0.01, 0.01))
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*self.time_maa_w2.size(0)))

            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA))
            self.time_maa_w_w2 = nn.Parameter(torch.zeros(D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))
            D_DECAY_LORA = 128
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            # self.time_receptance_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            # self.time_receptance_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))
            # self.time_key_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            # self.time_key_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_receptance = nn.Parameter(torch.zeros(args.dim_att).uniform_(-1.0, 1.0)) #nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.time_key = nn.Parameter(torch.zeros(args.dim_att).uniform_(-1.0, 1.0)) #nn.Linear(args.n_embd, args.dim_att, bias=False)
        #self.time_devolve = nn.Parameter(torch.zeros(args.dim_att).uniform_(-1.0, 1.0))

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_x = nn.LayerNorm(args.dim_att)

    def forward(self, x_in, xo, kv_cache, last_state:TimeMixState, shared:Shared):
        B, T, C = x_in.size()

        #xxx = x + dxprev * self.time_maa_x
        xxx = x_in
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, self.time_maa_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), B, T, -1)

        x = self.value(x_in)

        H = self.n_head

        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

        half_x = x[..., :C]
        half_dxprev = dxprev[..., :C]
        mw = torch.tanh(half_x @ self.time_maa_w_w1) @ self.time_maa_w_w2
        xw = half_x + half_dxprev * (self.time_maa_w + mw)
        w = self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2 # + xw * self.time_devolve 

        # xr, xk, xv, xv2, xw = (x + dxprev * (self.time_maa_all.view(5, 1, 1, C) + xxx)).unbind(dim=0)
        mr, mk, mv, mv2 = xxx.unbind(dim=0)
        xv = x + dxprev * (self.time_maa_v + mv)
        xr = x + dxprev * (self.time_maa_r + mr)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv2 = x + dxprev * (self.time_maa_v2 + mv2)
        
        v = xv
        r = xr * self.time_receptance# + torch.tanh(x_in @ self.time_receptance_w1) @ self.time_receptance_w2
        k = xk * self.time_key# + torch.tanh(x_in @ self.time_key_w1) @ self.time_key_w2
        v2 = xv2 #+ torch.tanh(xw @ self.time_value2_w1) @ self.time_value2_w2
        k = k * (1 - (-w.exp()).exp())
        u = torch.zeros_like(self.time_faaaa)

        wkv_state = last_state.wkv_state.clone()
        y = RUN_CUDA_RWKV6(B, T, v.size(-1), H, r, k, v, w, u, wkv_state)
        y = y + v2

        y = self.ln_x(y)
        y = self.output(y)
        return y, TimeMixState(wkv_state, shift_state)
