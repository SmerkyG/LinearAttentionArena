import torch
from torch import nn, Tensor

from src.state import ModelState, ChannelMixState
from configs import Transformer_Config

from moe.layer import MoE
from .cmix_rwkv_base import get_default_state

class CMix_x060moe(nn.Module):
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

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

        primes = [5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443]
        hash_prime = primes[layer_id]
        
        if args.num_experts > 1:
            self.moe = MoE(input_size=args.n_embd, expert=RWKV_Expert(args, layer_id), num_experts = args.num_experts, ep_size=args.ep_size, k=1, min_capacity=4, capacity_factor=1, eval_capacity_factor=1, drop_tokens=True, hash_prime=hash_prime)
            self.ffn = None
        else:
            self.moe = None
            self.ffn = RWKV_Expert(args, layer_id)


    def forward(self, x, last_model_state:ModelState):
        last_state = last_model_state.block_states[self.layer_id].channel_mix_state
        token_ids = last_model_state.input_tokens_cache
        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        xk = x + dxprev * self.time_maa_k
        xr = x + dxprev * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)

        empty_tensor = x[0:1]

        if self.moe is not None:
            B, T, C = xk.shape
            if not self.training: #T % self.num_experts != 0:
                Tnew = (T + self.num_experts - 1) // self.num_experts * self.num_experts
                # FIXME - pad it a lot more so that we don't go overcapacity during inference
                Tnew = Tnew * 4
                xk = torch.nn.functional.pad(xk, [0, 0, 0, Tnew - T])
                token_ids = torch.nn.functional.pad(token_ids, [0, Tnew - T])
                dkv = self.moe.forward(xk, token_ids, empty_tensor)
                dkv = dkv[:, :T]
                kv = kv + dkv
            else:
                kv = kv + self.moe.forward(xk, token_ids, empty_tensor)
        if self.ffn is not None:
            kv = kv + self.ffn(xk)

        return torch.sigmoid(self.receptance(xr)) * kv, ChannelMixState(shift_state=shift_state)

class RWKV_Expert(nn.Module):
    def __init__(self, args:Transformer_Config, layer_id):
        super().__init__()
        dim_ffn_expert = args.dim_ffn_expert
        if dim_ffn_expert <= 0:
            dim_ffn_expert = args.dim_ffn
        self.ffn_key = nn.Linear(args.n_embd, dim_ffn_expert, bias=False)
        self.ffn_value = nn.Linear(dim_ffn_expert, args.n_embd, bias=False)

    def forward(self, xk):
        return self.ffn_value( torch.relu( self.ffn_key(xk) ).square() )
