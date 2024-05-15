import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .CoreDependencies import *
from .cuda6 import RUN_CUDA_RWKV6

from .tmix import TimeMixState

import math

def causal_bias_mask(T):
    return torch.full((T, T), float('-inf')).triu(1)

def alibi_mask(T, H):
    bias = (torch.arange(T)[None, :] - torch.arange(T)[:, None]).float() # (T, T)
    bias = bias + causal_bias_mask(T) # (T, T)
    bias = bias.expand(H, -1, -1) # (H, T, T)
    head_bias_slopes = (2 ** torch.linspace(-8.0/H, -8.0, H)).unsqueeze(-1).unsqueeze(-1) # (H, 1, 1)
    bias = bias * head_bias_slopes # (H, T, T)
    return bias

class AlibiMask(nn.Module):
    def __init__(self, block_size : int, n_heads : int, layer_id : int):
        super().__init__()
        T = block_size
        H = n_heads
        self.register_buffer('mask', alibi_mask(T, H))

    def forward(self, q:Tensor):
        return self.mask[:, :q.size(-2), :q.size(-2)]

class RMSNorm(nn.Module):
    def __init__(self, dim : int, weight_scaling : bool = True, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.dim = dim
        starting_scale = dim ** -0.5
        if weight_scaling:
            self.register_parameter("scale", nn.Parameter(torch.ones(dim) * starting_scale))
        else:
            self.scale = starting_scale

    def forward(self, x):
        assert(self.dim == x.size(-1))
        rms_norm = self.scale * x.norm(2, dim=-1, keepdim=True)
        return x / rms_norm.clamp(self.eps)
    
def rms_norm(x, eps:float = 1e-8):
    rms_norm = (x.size(-1) ** -0.5) * x.norm(2, dim=-1, keepdim=True)
    return x / (rms_norm + eps)

class Norm(nn.Module):
    def __init__(self, dim : int, weight_scaling : bool = True, eps = 1e-8):
        super().__init__()
        self.eps = eps
        if weight_scaling:
            self.register_parameter("scale", nn.Parameter(torch.ones(dim)))
        else:
            self.scale = 1

    def forward(self, x):
        return self.scale * x / x.norm(2, dim=-1, keepdim=True).clamp(self.eps)

def l2_norm(x, eps:float = 1e-8):
    # assumes that vector 'normally' has length 1, not length vec.size(-1)**0.5 (which would be if every component had an average absolute value of 1!)
    return x / (x.norm(2, dim=-1, keepdim=True) + eps)

class RWKV_Tmix_dcattn(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.n_layer = args.n_layer

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
            self.time_maa_q = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            D_MIX_LORA = 32
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*3))
            self.time_maa_w2 = nn.Parameter(torch.zeros(3, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_r = nn.LayerNorm(args.dim_att)
        self.ln_k = nn.LayerNorm(args.dim_att)
        self.ln_v = nn.LayerNorm(args.dim_att)
        self.ln_x = nn.LayerNorm(args.dim_att)

        self.DYNPROJ_EXPANSION = 1
        D_DYNPROJ_LORA = args.dim_att // 8
        self.dynproj_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DYNPROJ_LORA*8))
        self.dynproj_w2 = nn.Parameter(torch.zeros(8, D_DYNPROJ_LORA, self.n_head).uniform_(-0.01, 0.01))
        self.dyngate_w1 = nn.Parameter(torch.zeros(args.n_embd, self.n_head*4).uniform_(-0.01, 0.01))

        self.bias_mask = AlibiMask(args.ctx_len, self.n_kv_head, layer_id)

    def batch_lora(self, xw1, w2): 
        B,T,Ctotal = xw1.shape
        n_bound = w2.size(0)
        assert Ctotal % n_bound == 0
        return (xw1.view(B*T,n_bound,-1).transpose(0,1) @ w2).view(n_bound,B,T,-1)

    def compose(self, x, qpw1, qpw2, kpw1, kpw2, qgw, kgw): # sw, 
        # N/M=n_head T/S=seq_dim for q and k
        # sw: (N,N)
        # qpw1, qpw2: (B,T,N)
        # kpw1, kpw2: (B,S,N)
        # qgw: (B,T,N)
        # kgw: (B,S,N)
        # 'base projection' head mixing - just a matrix of size (N,N) 
        y = x
        # if sw is not None:
        #     y = y + torch.einsum('BNTS,NM->BMTS', inputs, sw)
        #for i in range(qpw1.shape[-2]):
        # 'dynamic projection'
        # "lora" by reducing heads away, then adding heads back in
        y = y + torch.einsum('BNTS,BTN,BTM->BMTS', x, qpw1, qpw2) #* self.DYNPROJ_EXPANSION
        y = y + torch.einsum('BNTS,BSN,BTM->BMTS', x, kpw1, kpw2) #* self.DYNPROJ_EXPANSION
        # 'dynamic gating'
        y = y + torch.einsum('BNTS,BTN->BNTS', x, qgw)
        y = y + torch.einsum('BNTS,BSN->BNTS', x, kgw)
        return y
        
    @MyFunction
    def forward(self, x, last_state:TimeMixState):
        B, T, D = x.size()
        N = self.n_head
        K = D // N
        V = D // N

        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x

        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, self.time_maa_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), B, T, D)

        mq, mk, mv = xxx.unbind(dim=0)
        xq = x + dxprev * (self.time_maa_q + mq)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        
        q = self.receptance(xq)
        k = self.key(xk)
        v = self.value(xv)
        
        q = self.ln_r(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        q = q.view(B,-1,N,K).transpose(1,2)
        k = k.view(B,-1,N,K).transpose(1,2)
        v = v.view(B,-1,N,K).transpose(1,2)

        B,N,T,Q = q.shape


        dynproj_w1s, dynproj_w2s = self.batch_lora(F.gelu(x @ self.dynproj_w1), self.dynproj_w2).view(2,4,B,T,N).unbind(0)
        pre_qpw1, pre_kpw1, post_qpw1, post_kpw1 = rms_norm(dynproj_w1s).unbind(0)
        pre_qpw2, pre_kpw2, post_qpw2, post_kpw2 = dynproj_w2s.unbind(0)

        pre_qgw, pre_kgw, post_qgw, post_kgw = torch.tanh(x @ self.dyngate_w1).view(B,T,4,N).unbind(2)

        #pre = dict(qpw1=pre_qpw1, qpw2=pre_qpw2, kpw1=pre_kpw1, kpw2=pre_kpw2, qgw=pre_qgw, kgw=pre_kgw)
        #post = dict(qpw1=post_qpw1, qpw2=post_qpw2, kpw1=post_kpw1, kpw2=post_kpw2, qgw=post_qgw, kgw=post_kgw)


        scale = D ** -0.5
        q = q * scale

        L = T

        logits = q @ k.mT
        logits = self.compose(logits, pre_qpw1, pre_qpw2, pre_kpw1, pre_kpw2, pre_qgw, pre_kgw)
        #logits = logits.tril()
        logits = logits + self.bias_mask(q)
        probs = F.softmax(logits, dim=-1)
        probs = self.compose(probs, post_qpw1, post_qpw2, post_kpw1, post_kpw2, post_qgw, post_kgw)
        y = probs @ v

        # # chunk
        # L = T
        # T = 64
        # C = L//T
        # assert L % T == 0, 'context length must be an even multiple of chunk size'
        # q = q.view(B,N,C,T,-1)
        # k = k.view(B,N,C,T,-1)
        # v = v.view(B,N,C,T,-1)
        # pre_qpw1 = pre_qpw1.view(B,C,T,N)
        # pre_qpw2 = pre_qpw2.view(B,C,T,N)
        # pre_kpw1 = pre_kpw1.view(B,C,T,N)
        # pre_kpw2 = pre_kpw2.view(B,C,T,N)
        # pre_qgw = pre_qgw.view(B,C,T,N)
        # pre_kgw = pre_kgw.view(B,C,T,N)
        # post_qpw1 = post_qpw1.view(B,C,T,N)
        # post_qpw2 = post_qpw2.view(B,C,T,N)
        # post_kpw1 = post_kpw1.view(B,C,T,N)
        # post_kpw2 = post_kpw2.view(B,C,T,N)
        # post_qgw = post_qgw.view(B,C,T,N)
        # post_kgw = post_kgw.view(B,C,T,N)

        # y = torch.zeros((B, N, C, T, V), device=q.device, dtype=q.dtype)
        # for i in range(C):
        #     for j in range(i+1):
        #         qi, kj, vj = q[:,:,i], k[:,:,j], v[:,:,j]
        #         logits = qi @ kj.mT
        #         logits = self.compose(logits, pre_qpw1[:,i], pre_qpw2[:,i], pre_kpw1[:,j], pre_kpw2[:,j], pre_qgw[:,i], pre_kgw[:,j])
        #         if i == j:
        #             logits = logits.tril()
        #         #logits = logits + self.bias_mask(q)
        #         probs = F.softmax(logits, dim=-1)
        #         probs = self.compose(probs, post_qpw1[:,i], post_qpw2[:,i], post_kpw1[:,j], post_kpw2[:,j], post_qgw[:,i], post_kgw[:,j])
        #         y[:,:,i] += probs @ vj

        # # dechunk
        # y = y.view(B,N,L,V).to(x.dtype)

        # y = nn.functional.scaled_dot_product_attention(
        #     lr.view(B,-1,H,K).transpose(1,2),
        #     lk.view(B,-1,H,K).transpose(1,2),
        #     lv.view(B,-1,H,V).transpose(1,2),
        #     attn_mask=self.bias_mask(lr), dropout_p=0.0, is_causal=self.bias_mask is None)
        #     #is_causal=True)
        
        y = y.transpose(1,2).reshape(B,L,D)
       
        y = self.ln_x(y)

        #x = x * lg

        y = self.output(y)

        return y, TimeMixState(last_state.wkv_state, shift_state)

