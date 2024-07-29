import torch
from torch import nn, Tensor

from src.state import ChannelMixState
from .cmix_rwkv_base import get_default_state

class CMix_x060lora(nn.Module):
    def get_default_state_factory(self): return get_default_state

    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0)).requires_grad_(False)
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0)).requires_grad_(False)

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False).requires_grad_(False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False).requires_grad_(False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False).requires_grad_(False)
        if args.lora_rank > 0:
            self.key_w1 = nn.Parameter(torch.zeros(args.n_embd, args.lora_rank))
            self.key_w2 = nn.Parameter(torch.empty(args.lora_rank, args.dim_ffn).uniform_(-0.01, 0.01))
            self.receptance_w1 = nn.Parameter(torch.zeros(args.n_embd, args.lora_rank))
            self.receptance_w2 = nn.Parameter(torch.empty(args.lora_rank, args.n_embd).uniform_(-0.01, 0.01))
            self.value_w1 = nn.Parameter(torch.zeros(args.dim_ffn, args.lora_rank))
            self.value_w2 = nn.Parameter(torch.empty(args.lora_rank, args.n_embd).uniform_(-0.01, 0.01))
        else:
            self.key_w1 = None
            self.key_w2 = None
            self.receptance_w1 = None
            self.receptance_w2 = None
            self.value_w1 = None
            self.value_w2 = None

    def forward(self, x, last_state:ChannelMixState):
        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        xk = x + dxprev * self.time_maa_k
        xr = x + dxprev * self.time_maa_r

        k = self.key(xk)
        if self.key_w1 is not None:
            k = k + (xk @ self.key_w1).tanh() @ self.key_w2
        k = torch.relu(k) ** 2
        kv = self.value(k)
        if self.value_w1 is not None:
            kv = kv + (k @ self.value_w1).tanh() @ self.value_w2
        r = self.receptance(xr)
        if self.receptance_w1 is not None:
            r = r + (xr @ self.receptance_w1).tanh() @ self.receptance_w2
        return torch.sigmoid(r) * kv, ChannelMixState(shift_state=shift_state)
