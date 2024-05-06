import torch
from torch import nn, Tensor
from .CoreDependencies import *
from .cuda6 import RUN_CUDA_RWKV6

from .tmix import TimeMixState

class RWKV_Tmix_x060o3(MyModule):
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

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_x = nn.LayerNorm(args.dim_att)

    @MyFunction
    def forward(self, x):        
        #dtype = torch.get_autocast_gpu_dtype()
        #x = x.to(dtype)

        B, T, C = x.size()
        H = self.n_head
        K = C // H
        V = C // H

        dxprev = self.time_shift(x) - x

        xxx = x + dxprev * self.time_maa_x

        xxx = torch.tanh(xxx @ self.time_maa_rkvw_w1).view(B*T, 4, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_rkvw_w2).view(4, B, T, C)

        r, k, v, w = xxx.unbind(dim=0)
        r = x + dxprev * (self.time_maa_r + r)
        k = x + dxprev * (self.time_maa_k + k)
        v = x + dxprev * (self.time_maa_v + v)
        w = x + dxprev * (self.time_maa_w + w)
        
        r = self.receptance(r)
        k = self.key(k)
        v = self.value(v)
        w = self.time_decay + k
        tau = 9
        # w in log-log space, securely clamped
        w = -nn.functional.elu(-w+tau)+tau
        k = (1 - ( (-w.exp()).exp() )).to(r)

        u = self.time_faaaa

        #wkv_state = last_state.wkv_state.to(torch.bfloat16)#.clone()
        x = RUN_CUDA_RWKV6(B, T, C, H, r.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), w.to(torch.bfloat16), u.to(torch.bfloat16))
        #wkv_state = last_state.wkv_state.to(torch.bfloat16)
        #x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u, wkv_state)

        #x, s = fused_recurrent_rwkv6hypno2(r, k, v, w, u, z, initial_state=torch.zeros([0], dtype=r.dtype, device=r.device), scale=-1, output_final_state=False, causal=True)
        #x = x.transpose(1,2).reshape(B,T,C)
        
        x = self.ln_x(x)
        x = self.output(x)
        return x

