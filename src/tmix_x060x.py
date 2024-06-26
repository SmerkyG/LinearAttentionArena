import torch
from torch import nn, Tensor
from .CoreDependencies import *
#from .cuda6 import RUN_CUDA_RWKV6

from .rwkv_triton_v6hypno2 import fused_recurrent_rwkv6hypno2

class RWKV_Tmix_x060x(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.k_head_size = self.v_head_size = self.head_size = args.head_size_a
        self.n_kv_head = self.n_head = args.dim_att // self.head_size
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
            self.time_maa_rkvw_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*4))
            self.time_maa_rkvw_w2 = nn.Parameter(torch.zeros(4, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))
            # D_DECAY_LORA = 64
            # self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            # self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att, self.v_head_size)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
                tmp[n, :] += torch.randn_like(tmp[n, :]) * 1e-4

            time_first = tmp.reshape(self.n_kv_head, self.k_head_size, self.v_head_size)
            self.time_first = nn.Parameter(time_first) # (KVH, K, K)

            self.time_filter = nn.Parameter(1 + 1e-5 * torch.randn(self.n_kv_head, self.k_head_size, self.v_head_size))

            # self.receptance = nn.Linear(self.head_size, self.head_size, bias=False)
            # self.key = nn.Linear(self.head_size, self.head_size, bias=False)
            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        #self.v_amt = nn.Linear(args.n_embd, self.n_head, bias=False)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_x = nn.LayerNorm(args.dim_att)

    @MyFunction
    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head
        K = C // H
        V = C // H

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_rkvw_w1).view(B*T, 4, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_rkvw_w2).view(4, B, T, C)

        r, k, v, w = xxx.unbind(dim=0)
        r = x + xx * (self.time_maa_r + r)
        k = x + xx * (self.time_maa_k + k)
        v = x + xx * (self.time_maa_v + v)
        w = x + xx * (self.time_maa_w + w)
        
        #r = self.receptance(r.view(B*T*H,1,K)).view(B,T,H,K).transpose(1,2)
        #k = self.key(k.view(B*T*H,1,K)).view(B,T,H,K).transpose(1,2)
        r = self.receptance(r).view(B,T,H,K).transpose(1,2)
        k = self.key(k).view(B,T,H,K).transpose(1,2)
        v = self.value(v).view(B,T,H,V).transpose(1,2)
        w = self.time_decay.float().view(1, H, 1, K) + k
        w = (-w.exp()).exp()
        k = (1-w).to(r.dtype)

        # w = -torch.exp(w) # log(exp(-exp))
        eps = 1e-5
        w = (eps + (1 - eps) * w).log()  # (B, H, T, K)        
        u = self.time_first.view(H,K,V).float()
        z = self.time_filter.view(H,K,V).float()

        #x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, torch.zeros(H, K, device=r.device, dtype=r.dtype))
        x, s = fused_recurrent_rwkv6hypno2(r, k, v, w, u, z, initial_state=torch.zeros([0], dtype=r.dtype, device=r.device), scale=-1, output_final_state=False, causal=True)
        x = x.transpose(1,2).reshape(B,T,C)
        
        x = self.ln_x(x)
        x = self.output(x)
        return x

