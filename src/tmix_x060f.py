import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .CoreDependencies import *
from .cuda6 import RUN_CUDA_RWKV6

class RWKV_Tmix_x060f(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.n_embd = args.n_embd
        self.layer_id = layer_id
        self.dim_ffn = int(args.n_embd * 2) // 32 * 32
        self.dim_k = args.n_embd
        self.dim_v = args.n_embd

        self.k_head_size = args.head_size
        self.v_head_size = int(args.head_size * self.dim_v / self.dim_k)
        self.n_head = args.dim_att // self.k_head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for k,v,r
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*3))
            self.time_maa_w2 = nn.Parameter(torch.zeros(3, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.k_head_size))

        self.time_v_bonus = nn.Parameter(torch.full([self.dim_v], 2.0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, self.dim_k, bias=False) # DK params
        self.key = nn.Linear(args.n_embd, self.dim_k, bias=False) # DK params
        self.v_ffn_biggate = nn.Linear(args.n_embd, self.dim_v + self.dim_ffn + self.dim_v + self.dim_ffn, bias=False) # 2D(V+F) params
        self.output = nn.Linear(self.dim_v + self.dim_ffn, args.n_embd, bias=False) # D(V+F) params
        self.ln_x = nn.LayerNorm(args.dim_att)

    def forward(self, x, xo, kv_cache, last_state:TimeMixState, shared:Shared):
        B, T, C = x.size()
        H = self.n_head
        K = self.k_head_size
        V = self.v_head_size

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 3, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(3, B, T, -1)
        mk, mv, mr = xxx.unbind(dim=0)

        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)

        r = self.receptance(xr)
        w = self.key(xk)
        k = 1.0 - torch.exp(-torch.exp(w))
        vffn, biggate = self.v_ffn_biggate(xv).split([self.dim_v+self.dim_ffn, self.dim_v+self.dim_ffn], dim=-1)
        vffn = vffn * F.silu(biggate)
        v, ffn = vffn.split([self.dim_v, self.dim_ffn], dim=-1)
        v = v.contiguous()
        u = self.time_faaaa

        # FIXME - GQA

        # FIXME - support different rk, v sizing
        x = RUN_CUDA_RWKV6(r, k, v, w, u)

        x = self.ln_x(x)
        x = torch.cat([x, ffn], dim=-1)
        x = self.output(x)
        return x
