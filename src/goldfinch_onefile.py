import math
import torch
from torch import nn
from dataclasses import dataclass

@dataclass(kw_only=True)
class Config:
    vocab_size:int
    d_model:int
    n_layer:int
    d_ffn:int = 0
    d_head:int = 64
    inverse_layer_ratio:float = 3.0

@dataclass(kw_only=True)
class TimeMixState:
    wkv_state : torch.Tensor
    shift_state : torch.Tensor

@dataclass(kw_only=True)
class ChannelMixState:
    shift_state : torch.Tensor

@dataclass(kw_only=True)
class BlockState:
    time_mix_state : TimeMixState
    channel_mix_state : ChannelMixState

@dataclass(kw_only=True)
class ModelState:
    seq_pos : int = 0
    input_tokens_cache : torch.Tensor = torch.tensor([])
    k_cache : torch.Tensor = torch.tensor([])
    block_states : list[BlockState] = []

def rms_norm(x, eps:float = 1e-8):
    rms_norm = (x.size(-1) ** -0.5) * x.norm(2, dim=-1, keepdim=True)
    return x / (rms_norm + eps)

def get_first_goco_layer_id(config:Config):
    return int(config.n_layer / (1.0 - config.inverse_layer_ratio))

class Transformer(nn.Module):
    def __init__(self, config:Config, att_factory0, att_factory1, ffn_factory, do_init_weights:bool=True):
        super().__init__()
        self.config = config
        if config.d_ffn == 0:
            config.d_ffn = int(config.d_model * 3.5) // 32 * 32

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_layernorm = nn.LayerNorm(config.d_model)
        first_goco_layer_id = get_first_goco_layer_id(config)
        self.blocks = nn.ModuleList([
            TransformerBlock(config, 
                att_factory0(config) if layer_id < first_goco_layer_id else att_factory1(config), 
                ffn_factory(config)
            ) 
            for layer_id in range(config.n_layer)
        ])
        self.ln_out = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        K_COMPRESSION_FACTOR = 16
        self.w_k_cache_a = nn.Linear(config.d_model, config.d_model // K_COMPRESSION_FACTOR, bias=False)
        self.w_k_cache_b = nn.Linear(config.d_model // K_COMPRESSION_FACTOR + config.d_model, config.d_model, bias=False)

        # these initializations are important for performance (but skip them for speed if you are loading a model from a pre-trained checkpoint)
        if do_init_weights:
            self.init_weights()

    def forward(self, inputs, model_state_in:ModelState|None = None):
        config = self.config
        B, T = inputs.size()

        x = self.embedding(inputs)
        x = self.embedding_layernorm(x)
        original_input_embeddings = x
        k_cache = torch.tensor([]) if model_state_in is None else model_state_in.k_cache
        model_state_out = ModelState()
        for layer_id in range(config.n_layer):
            x, model_state_out.block_states[layer_id] = self.blocks[layer_id](x, original_input_embeddings, original_input_embeddings, k_cache)

        return model_state_in

    def init_weights(self):
        for name, m in self.named_modules():               
            scale = 1.0
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

                for kk in [".att.output", ".ffn.value", ".ffn.receptance"]:
                    if name.endswith(kk):
                        scale = 0

                for kk in [".att.key"]:
                    if name.endswith(kk):
                        scale = 0.1

                if name == "head":
                    if self.config.vocab_size > self.config.d_model:
                        scale = 0.5 * math.sqrt(self.config.vocab_size / self.config.d_model)
                    else:
                        scale = 0.5

                if scale == 0:
                    nn.init.zeros_(m.weight)
                else:   
                    nn.init.orthogonal_(m.weight, gain=scale)

                print(f"{name} scale={scale}")
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, a=-1e-4, b=1e-4)
                print(name, "embed init")
            elif name.endswith('.ln_x'):
                layer_scale = (1+int(name.split('.')[1])) / self.config.n_layer
                m.weight = nn.Parameter((m.weight * 0.0) + (layer_scale ** 0.7))
                print(name, "layer_scale init")
            else:
                print(name, "(default init)")

class GoldFinchTransformer(Transformer):
    def __init__(self, config:Config, do_init_weights:bool=True):
        super().__init__(config, FinchB2TimeMix, GoCOAttention, FinchChannelMix, do_init_weights=do_init_weights)

    def forward(self, input_token_indices, model_state_in:ModelState|None = None):
        config = self.config
        B, T = input_token_indices.size()

        if model_state_in is None:
            dtype, device = self.embedding.weight.dtype, self.embedding.weight.device
            model_state_in = ModelState()
            wkv_state_shape = [B, config.d_model//config.d_head, config.d_head, config.d_head]
            model_state_in.block_states = [
                BlockState(
                    time_mix_state=TimeMixState(
                        wkv_state=torch.zeros(wkv_state_shape, dtype=dtype, device=device), 
                        shift_state=torch.zeros([B, config.d_model], dtype=dtype, device=device)
                    ), 
                    channel_mix_state=ChannelMixState(
                        shift_state=torch.zeros([B, config.d_model], dtype=dtype, device=device)
                    )
                ) 
                for layer_id in range(config.n_layer)
            ]
            model_state_in.input_tokens_cache = torch.zeros([B, 0], dtype=torch.long, device=device, requires_grad=False)
            if self.is_cache_once:
                model_state_in.k_cache = torch.zeros([B, 0, config.d_model], dtype=dtype, device=device)

        model_state_out = ModelState()
        model_state_out.input_tokens_cache = torch.cat([model_state_in.input_tokens_cache, input_token_indices], dim=-1)

        x = self.embedding(input_token_indices)
        x = self.embedding_layernorm(x)

        if self.training:
            original_input_embeddings = x
        else:
            # use the input_tokens_cache to recreate the original_input_embeddings for prior sequence regions
            original_input_embeddings = torch.cat( [
                self.embedding_layernorm( self.embedding( model_state_in.input_tokens_cache ) ),
                x
            ], dim=-2)

        k_cache = model_state_in.k_cache
        first_goco_layer_id = get_first_goco_layer_id(config)
        for layer_id in range(config.n_layer):
            x, model_state_out.block_states[layer_id] = self.blocks[layer_id](x, original_input_embeddings, k_cache)

            if layer_id == first_goco_layer_id:
                new_compressed_k_cache_entries = self.w_k_cache_a(x)
                c = torch.cat([original_input_embeddings, new_compressed_k_cache_entries],dim=-1)
                # NOTE - instead of decompressing here, you can keep compressed_k_cache and decompress as you go during each sub-layer for extra memory savings
                new_k_cache_entries = rms_norm(self.w_k_cache_b(c))

                model_state_out.seq_pos = model_state_in.seq_pos + T
                if self.training:
                    k_cache = new_k_cache_entries
                else:
                    prefilling = k_cache.size(1) == 0
                    k_cache = torch.cat([k_cache, new_k_cache_entries], dim=-2)
                    if prefilling:
                        # NOTE - no need to run the GoCO attention layers during prefill!
                        break      
        
        model_state_out.k_cache = k_cache
        return model_state_out

class TransformerBlock(nn.Module):
    def __init__(self, config:Config, att:nn.Module, ffn:nn.Module):
        super().__init__()
        self.att = att
        self.ffn = ffn
        self.layernorm_att = nn.LayerNorm(config.d_model)
        self.layernorm_ffn = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x = x + self.att(self.layernorm_att(x))
        x = x + self.ffn(self.layernorm_ffn(x))
        return x

class FinchB2TimeMix(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.n_head = config.d_model // config.d_head
        self.time_maa_x = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.time_maa_r = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.time_maa_k = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.time_maa_v = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.time_maa_w = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.time_maa_v2 = nn.Parameter(torch.empty(1, 1, config.d_model))

        D_MIX_LORA = 32
        self.time_maa_w2 = nn.Parameter(torch.empty(5, D_MIX_LORA, config.d_model))
        self.time_maa_w1 = nn.Parameter(torch.empty(config.d_model, D_MIX_LORA * 5))

        self.time_decay = nn.Parameter(torch.empty(config.d_model))

        D_DECAY_LORA = 64
        self.time_decay_w1 = nn.Parameter(torch.empty(config.d_model, D_DECAY_LORA))
        self.time_decay_w2 = nn.Parameter(torch.empty(D_DECAY_LORA, config.d_model))

        self.time_value2_w1 = nn.Parameter(torch.empty(config.d_model, D_DECAY_LORA))
        self.time_value2_w2 = nn.Parameter(torch.empty(D_DECAY_LORA, config.d_model))

        self.receptance = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.ln_x = nn.LayerNorm(config.d_model)

    def forward(self, x, last_state:TimeMixState):
        B, T, C = x.size() # Batch Index, Sequence Index, Embedding Channel Index
        N = self.n_head
        H = C // N

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
        v2 = self.value(xv2) + torch.tanh(xv2 @ self.time_value2_w1) @ self.time_value2_w2
        w = self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2

        k = (k.float() * (1 - (-w.float().exp()).exp())).to(r.dtype)

        wkv_state = last_state.wkv_state.clone()

        r = r.view(B,T,N,H).transpose(1,2).view(B,N,T,H)
        k = k.view(B,T,N,H).transpose(1,2).view(B,N,T,H)
        v = v.view(B,T,N,H).transpose(1,2).view(B,N,T,H)
        v2 = v2.view(B,T,N,H).transpose(1,2).view(B,N,T,H)
        w = w.view(B,T,N,H).transpose(1,2).view(B,N,T,H)

        # NOTE - the recurrent form of Finch time mixing is implemented very inefficently here for clarity (see training codebase for CUDA, Triton, and efficent chunked versions)
        y = torch.empty_like(x)
        for t in range(T):
            y[:,:,t] = (r @ wkv_state + v2[:,:,t]).reshape(B,C)
            wkv_state = wkv_state * w[:,:,t] + k.mT @ v

        y = self.ln_x(y)

        y = self.output(y)
        return y, TimeMixState(wkv_state=wkv_state, shift_state=shift_state)

class GoCOAttention(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.n_head = config.d_model // config.d_head
        self.time_maa_x = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.time_maa_q = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.time_maa_v_cache = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.time_maa_k = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.time_maa_v = nn.Parameter(torch.empty(1, 1, config.d_model))

        D_MIX_LORA = 32
        self.time_maa_q_w1 = nn.Parameter(torch.empty(config.d_model, D_MIX_LORA))
        self.time_maa_q_w2 = nn.Parameter(torch.empty(D_MIX_LORA, config.d_model))
        self.time_maa_kv_w1 = nn.Parameter(torch.empty(config.d_model, D_MIX_LORA*2))
        self.time_maa_kv_w2 = nn.Parameter(torch.empty(2, D_MIX_LORA, config.d_model))

        D_VALUE_LORA = max(config.d_model // 16, 64)
        self.time_key_w1 = nn.Parameter(torch.zeros(config.d_model, D_VALUE_LORA))
        self.time_key_w2 = nn.Parameter(torch.zeros(D_VALUE_LORA, config.d_model).uniform_(-0.01, 0.01))
        self.time_value_w1 = nn.Parameter(torch.zeros(config.d_model, D_VALUE_LORA))
        self.time_value_w2 = nn.Parameter(torch.zeros(D_VALUE_LORA, config.d_model).uniform_(-0.01, 0.01))

        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.ln_q = nn.LayerNorm(config.d_model)
        self.ln_k = nn.LayerNorm(config.d_model)
        self.ln_v = nn.LayerNorm(config.d_model)
        self.ln_x = nn.LayerNorm(config.d_model)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

    def forward(self, x, xo, k_cache, last_time_mix_state:TimeMixState):
        B, T, C = x.size()
        N = self.n_head
        K = C // N
        V = C // N

        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_time_mix_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x
        mq = torch.tanh(xxx @ self.time_maa_q_w1) @ self.time_maa_q_w2
       
        xo = rms_norm(xo)
        dxo_prev = self.time_shift(xo) - xo
        xxx = xo + dxo_prev * self.time_maa_v_cache
        xxx = torch.tanh(xxx @ self.time_maa_kv_w1).view(B*xo.size(1), self.time_maa_kv_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_kv_w2).view(self.time_maa_kv_w2.size(0), B, xo.size(1), C)
        mk, mv = xxx.unbind(dim=0)

        k = k_cache
        dkprev = self.time_shift(k) - k
        v = xo
        dvprev = self.time_shift(v) - v

        xq = x + dxprev * (self.time_maa_q + mq)
        k = k + dkprev * (self.time_maa_k + mk)
        v = v + dvprev * (self.time_maa_v + mv)

        k = k + torch.tanh(k @ self.time_key_w1) @ self.time_key_w2
        v = v + torch.tanh(v @ self.time_value_w1) @ self.time_value_w2     

        q = self.query(xq)
        
        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        q = q.view(B,T,N,K).transpose(1,2)
        k = k.view(B,T,N,K).transpose(1,2)
        v = v.view(B,T,N,V).transpose(1,2)

        x = nn.functional.scaled_dot_product_attention(q,k,v,is_causal=q.size(-2)>1)

        x = x.transpose(1,2).reshape(B,-1,C)
       
        x = self.ln_x(x)

        x = self.output(x)

        return x, TimeMixState(wkv_state=last_time_mix_state.wkv_state, shift_state=shift_state)

class FinchChannelMix(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.time_maa_k = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.time_maa_r = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.key = nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.receptance = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value = nn.Linear(config.d_ffn, config.d_model, bias=False)

    def forward(self, x, last_state:ChannelMixState):
        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        xk = x + dxprev * self.time_maa_k
        xr = x + dxprev * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv, ChannelMixState(shift_state=shift_state)



# this class is not used in GoldFinch - it is provided to show how GPTAlpha Time Mixing works in a fully GPTAlpha Transformer model
class GPTAlphaTransformer(Transformer):
    def __init__(self, config:Config, do_init_weights:bool=True):
        super().__init__(config, GPTAlphaTimeMix, GPTAlphaTimeMix, FinchChannelMix, do_init_weights=do_init_weights)

from typing import Tuple
def apply_rotary_embedding(q, k, angles, seq_dim:int = -2) -> Tuple[torch.Tensor, torch.Tensor]:
    q_dtype, k_dtype = q.dtype, k.dtype
    L = q.size(seq_dim)
    angles = angles[-L:].view(1, 1, L, angles.size(1))
    if q.ndim == 3:
        q = torch.view_as_complex(q.float().reshape(q.size(0), q.size(1), -1, 2)) * angles
        k = torch.view_as_complex(k.float().reshape(k.size(0), k.size(1), -1, 2)) * angles
    else:
        q = torch.view_as_complex(q.float().reshape(q.size(0), q.size(1), q.size(2), -1, 2)) * angles
        k = torch.view_as_complex(k.float().reshape(k.size(0), k.size(1), k.size(2), -1, 2)) * angles
    return torch.view_as_real(q).flatten(3).to(q_dtype), torch.view_as_real(k).flatten(3).to(k_dtype)

class GPTAlphaTimeMix(nn.Module):
    def __init__(self, config:Config, angles):
        super().__init__()
        self.time_maa_x = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.time_maa_r = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.time_maa_k = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.time_maa_v = nn.Parameter(torch.empty(1, 1, config.d_model))

        D_MIX_LORA = 32
        self.time_maa_w1 = nn.Parameter(torch.empty(config.d_model, D_MIX_LORA*3))
        self.time_maa_w2 = nn.Parameter(torch.empty(3, D_MIX_LORA, config.d_model))

        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.ln_r = nn.LayerNorm(config.d_model)
        self.ln_k = nn.LayerNorm(config.d_model)
        self.ln_v = nn.LayerNorm(config.d_model)
        self.ln_x = nn.LayerNorm(config.d_model)

        self.angles = angles

    def forward(self, x, xo, kv_cache, last_time_mix_state:TimeMixState):
        B, T, C = x.size()
        N = self.n_head
        K = C // N
        V = C // N

        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_time_mix_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x

        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, self.time_maa_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), B, T, C)

        mr, mk, mv = xxx.unbind(dim=0)
        xq = x + dxprev * (self.time_maa_r + mr)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        
        q = self.query(xq)
        k = self.key(xk)
        v = self.value(xv)
        
        q = self.ln_r(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        q = q.view(B,T,N,K).transpose(1,2)
        k = k.view(B,T,N,K).transpose(1,2)
        v = v.view(B,T,N,V).transpose(1,2)

        if self.angles is not None:
            self.angles = self.angles.to(x.device)
            q, k = apply_rotary_embedding(q, k, self.angles)

        x = nn.functional.scaled_dot_product_attention(
            q, k, v,
            #attn_mask=self.bias_mask(lr), dropout_p=0.0, is_causal=self.bias_mask is None)
            is_causal=True)
        x = x.transpose(1,2).reshape(B,T,C)
       
        x = self.ln_x(x)

        x = self.output(x)

        return x, TimeMixState(wkv_state=last_time_mix_state.wkv_state, shift_state=shift_state)

