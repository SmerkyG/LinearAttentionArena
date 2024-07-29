import torch
from torch import nn, Tensor

from src.state import ChannelMixState
from configs import Transformer_Config

from moe.layer import MoE
from cmix.cmix_rwkv_base import get_default_state

class CMoE_c(nn.Module):
    def get_default_state_factory(self): return get_default_state

    def __init__(self, args:Transformer_Config, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.num_experts = args.num_experts

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.ffn_receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        nn.init.zeros_(self.ffn_receptance.weight)

        primes = [5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443]
        hash_prime = primes[layer_id]
        
        self.moe = RWKV_Expert(args, layer_id)
        if args.num_experts > 1:
            self.moe = MoE(input_size=args.n_embd, expert=self.moe, num_experts = args.num_experts, ep_size=args.ep_size, k=1, min_capacity=4, capacity_factor=1, eval_capacity_factor=1, drop_tokens=True, hash_prime=hash_prime)

    def forward(self, x:Tensor, token_ids:Tensor, last_state:ChannelMixState):
        dxprev = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        xk = x + dxprev * self.time_maa_k
        xr = x + dxprev * self.time_maa_r
        if self.num_experts > 1:
            y = self.moe(xk, token_ids, used_token=torch.tensor([]))
        else:
            y = self.moe(xk)

        return torch.sigmoid(self.ffn_receptance(xr)) * y, ChannelMixState(x[:, -1])
        
class RWKV_Expert(nn.Module):
    def __init__(self, args:Transformer_Config, layer_id):
        super().__init__()
        dim_ffn_expert = args.dim_ffn_expert
        if dim_ffn_expert <= 0:
            dim_ffn_expert = args.dim_ffn
        self.ffn_key = nn.Linear(args.n_embd, dim_ffn_expert, bias=False)
        self.ffn_value = nn.Linear(dim_ffn_expert, args.n_embd, bias=False)
        nn.init.orthogonal_(self.ffn_key.weight)
        nn.init.zeros_(self.ffn_value.weight)

    def forward(self, xk):
        return self.ffn_value( torch.relu( self.ffn_key(xk) ).square() )
