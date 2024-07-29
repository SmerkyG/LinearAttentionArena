import torch
from torch import nn, Tensor
import torch.nn.functional as F
from cuda.rwkv6_cuda import RUN_CUDA_RWKV6
from src.state import ModelState, TimeMixState, Shared

from configs import FinchC2_Config
from .tmix_rwkv_base import get_default_state

class TMix_x060c2(nn.Module):
    def get_default_state_factory(self): return get_default_state
    
    def __init__(self, args:FinchC2_Config, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        self.use_one_minus_w = getattr(args, 'use_one_minus_w', 1)
        self.use_v2 = getattr(args, 'use_v2', 1)

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
            self.time_maa_v2 = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            D_MIX_LORA = 32
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*self.time_maa_w2.size(0)))

            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))
            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            self.time_value2_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_value2_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_x = nn.LayerNorm(args.dim_att)

    def forward(self, x, xo, kv_cache, last_model_state:ModelState, shared:Shared):
        last_state = last_model_state.block_states[self.layer_id].time_mix_state
        B, T, C = x.size()
        H = self.n_head

        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, self.time_maa_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), B, T, C)

        mr, mk, mv, mw, mv2 = xxx.unbind(dim=0)
        xr = x + dxprev * (self.time_maa_r + mr)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xw = x + dxprev * (self.time_maa_w + mw)
        xv2 = x + dxprev * (self.time_maa_v2 + mv2)
        
        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        v2 = (self.value(xv2) + torch.tanh(xv2 @ self.time_value2_w1) @ self.time_value2_w2).to(r.dtype)
        w = (self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2).to(r.dtype)

        if self.use_one_minus_w:
            k = (k * (1 - (-w.exp()).exp())).to(r.dtype)
            #k = (k.float() * (1 - (-w.float().exp()).exp())).to(r.dtype)

        if self.use_v2:
            u = torch.zeros_like(self.time_faaaa)
        else:
            u = self.time_faaaa

        wkv_state = last_state.wkv_state.clone()
        y = RUN_CUDA_RWKV6(r, k, v, w, u, wkv_state)

        if self.use_v2:
            y = y + v2

        if self.training:
            y = self.ln_x(y)
        else:
            y = F.layer_norm(y.float(), self.ln_x.normalized_shape, self.ln_x.weight.float(), self.ln_x.bias.float()).to(y.dtype)

        y = self.output(y)
        return y, TimeMixState(wkv_state, shift_state)
