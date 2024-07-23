import torch
from torch import nn, Tensor

from src.state import ChannelMixState
from configs import Transformer_Config

from moe.layer import MoE

class CMoE_a(nn.Module):
    def __init__(self, args:Transformer_Config, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        # with torch.no_grad():  # fancy init of time_mix
        #     ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
        #     ddd = torch.ones(1, 1, args.n_embd)
        #     for i in range(args.n_embd):
        #         ddd[0, 0, i] = i / args.n_embd
        #     self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        primes = [5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443]
        hash_prime = primes[layer_id]

        example_expert = RWKV_Expert(args, layer_id)
        self.moe = MoE(hidden_size=args.n_embd, expert=example_expert, num_experts = args.num_experts, ep_size=args.ep_size, k=1, min_capacity=4, capacity_factor=1, eval_capacity_factor=1, drop_tokens=True, hash_prime=hash_prime)

    def forward(self, x:Tensor, token_ids:Tensor, last_state:ChannelMixState):
        # dxprev = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        # xk = x + dxprev * self.time_maa_k
        xk = x
        out = self.moe(xk, token_ids, used_token=torch.tensor([]))
        return out
        
class RWKV_Expert(nn.Module):
    def __init__(self, args:Transformer_Config, layer_id):
        super().__init__()
        self.ffn_key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.ffn_receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.ffn_value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)
        nn.init.uniform_(self.ffn_key.weight)
        nn.init.zeros_(self.ffn_receptance.weight)
        nn.init.zeros_(self.ffn_value.weight)

    def forward(self, x):
        kv = self.ffn_value( torch.relu( self.ffn_key(x) ) ** 2 )
        # NOTE - we use the same x for receptance here, unlike in normal chanmix
        return torch.sigmoid(self.ffn_receptance(x)) * kv
