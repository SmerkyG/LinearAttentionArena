import torch
from torch import nn, Tensor

from src.state import ChannelMixState
from .cmix_rwkv_base import get_default_state

from PEER_pytorch import PEER, ChunkedPEER
import math

class CMix_peer(nn.Module):
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
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        num_experts_per_head = 8 #16
        hidden_neuron_pool_size = 64 * 64 # 4096 # must be a square, this is the pool from which hidden neurons are chosen
        num_hidden_neurons_used = 256 # heads is really hidden neurons used / num_experts_per_head
        self.peer = PEER(
            dim = args.n_embd,
            num_experts_per_head = num_experts_per_head,
            heads = 8, #num_hidden_neurons_used // num_experts_per_head,
            num_experts = hidden_neuron_pool_size, # int(math.sqrt(args.n_embd)) ** 2), # maybe use dim_ffn
            separate_embed_per_head = False,
        )
        # self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)

    def forward(self, x, last_state:ChannelMixState):
        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        xk = x + dxprev * self.time_maa_k
        xr = x + dxprev * self.time_maa_r

        kv = self.peer(xk)
        return kv, ChannelMixState(shift_state=shift_state)
        #return torch.sigmoid(self.receptance(xr)) * kv, ChannelMixState(shift_state=shift_state)
