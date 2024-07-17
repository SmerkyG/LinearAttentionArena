########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch
import torch.utils.checkpoint
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import lightning.pytorch as pl
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only
from lightning.pytorch.strategies import DeepSpeedStrategy

from .tmix import TimeMixState, ModelState, Shared
from .cmix import ChannelMixState

import src

from configs import TrainerCLI_Config, Model_Config, Transformer_Config, Train_Config

import src.cmix_x052
import src.cmix_x060

from .rotary import generate_rotary_embedding, generate_binary_rotary_embedding, apply_rotary_embedding
from .norm import rms_norm

import numpy as np

if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from .CoreDependencies import *

from src.logger import print0 as print

try:
    print('RWKV_MODEL_TYPE', os.environ["RWKV_MODEL_TYPE"])
except:
    os.environ["RWKV_MODEL_TYPE"] = ''

# import timemix modules based on model type name
model_type = os.environ["RWKV_MODEL_TYPE"]
for model_subtype in model_type.split('_'):
    if not importlib.util.find_spec('src.tmix_' + model_subtype):
        print(f"couldn't find src.tmix_{model_subtype}, despite it being listed as part of the model name")
        exit()
    importlib.import_module('src.tmix_' + model_subtype)


########################################################################################################
# CUDA Kernel
########################################################################################################

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################

class BlockState:
    def __init__(self, time_mix_state: TimeMixState, channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state

def get_second_submodel_layer_id(model_config:Model_Config):
    return int(model_config.n_layer * (model_config.inv_other_layer_ratio - 1) / model_config.inv_other_layer_ratio)

class Block(nn.Module):
    def __init__(self, config:Model_Config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)

        self.parallel = False

        ffnFactory = lambda: src.cmix_x060.RWKV_CMix_x060(config, layer_id)

        mt = config.model_type
        if 'parallel' in mt:
            self.parallel = True

        model_subtypes = mt.split('_')
        match model_subtypes[0]:
            case 'x060bbswa':
                attFactory = lambda: RWKV_Tmix_x060bbswa(config, layer_id)
            case 'x060c2':
                attFactory = lambda: src.tmix_x060c2.RWKV_Tmix_x060c2(config, layer_id)
            case 'x060b':
                attFactory = lambda: src.tmix_x060b.RWKV_Tmix_x060b(config, layer_id)
            case 'x052':
                attFactory = lambda: src.tmix_x052.RWKV_Tmix_x052(config, layer_id)
                ffnFactory = lambda: src.cmix_x052.RWKV_CMix_x052(config, layer_id)
            case 'x060':
                attFactory = lambda: src.tmix_x060.RWKV_Tmix_x060(config, layer_id)
            case 'gptalpha':
                attFactory = lambda: src.tmix_gptalpha.GPTAlpha_Tmix(config, layer_id)
            case 'llama':
                attFactory = lambda: src.tmix_llama.Llama_Tmix(config, layer_id)
                ffnFactory = lambda: src.tmix_llama.Llama_CMix(config, layer_id)
            case 'mamba':
                attFactory = lambda: src.tmix_mamba.Mamba(config, layer_id)
                ffnFactory = lambda: src.tmix_mamba.MambaFFN(config, layer_id)
            case _:
                print(f"Unsupported model type: {mt}")
                exit(0)
        
        if len(model_subtypes) > 1 and 'taylor' in model_subtypes:
            if layer_id >= get_second_submodel_layer_id(config): #config.n_layer * 2 // 3 - 1 and layer_id < config.n_layer - 1:
                if 'taylorchunked' in mt:
                    attFactory = lambda: src.tmix_taylorchunked.RWKV_Tmix_taylorchunked(config, layer_id)
                else:
                    attFactory = lambda: src.tmix_taylor.RWKV_Tmix_taylor(config, layer_id)

        if len(model_subtypes) > 1 and model_subtypes[1] == 'gptalpha':
            if layer_id >= get_second_submodel_layer_id(config):
                attFactory = lambda: src.tmix_gptalpha.GPTAlpha_Tmix(config, layer_id)

        self.is_cache_once = len(model_subtypes) > 1 and model_subtypes[1] == 'gold'
        if self.is_cache_once:
            if layer_id >= get_second_submodel_layer_id(config) and layer_id < config.n_layer:
                if model_subtypes[1] == 'goldbha':
                    attFactory = lambda: src.tmix_goldbha.GPTAlpha_Tmix_goldbha(config, layer_id)
                else:
                    attFactory = lambda: src.tmix_gold.GPTAlpha_Tmix_gold(config, layer_id)
                if model_subtypes[0] == 'mamba':
                    ffnFactory = lambda: src.cmix_x060.RWKV_CMix_x060(config, layer_id)

        self.att = attFactory()
        self.default_time_mix_state_factory = self.att.get_default_state_factory() if hasattr(self.att, 'get_default_state_factory') else lambda x, c, r: TimeMixState()
        self.att = TJIT(self.att)
        
        if ffnFactory is None:
            self.ln2 = None
            self.ffn = None
        else:
            self.ffn = ffnFactory()
        self.default_channel_mix_state_factory = self.ffn.get_default_state_factory() if hasattr(self.ffn, 'get_default_state_factory') else lambda x, c, r: ChannelMixState()
        self.ffn = TJIT(self.ffn)


        if config.dropout > 0:
            self.drop0 = nn.Dropout(p = config.dropout)
            self.drop1 = nn.Dropout(p = config.dropout)
        else:
            self.drop0 = nn.Identity()
            self.drop1 = nn.Identity()

    def forward(self, x, x_original_cache, kv_cache, last_model_state:ModelState, shared:Shared):
        B, T, C = x.size()
        # if self.layer_id == 0:
        #     x = self.ln0(x)

        last_block_state = last_model_state.block_states[self.layer_id]

        # if len(kv_cache.shape) > 0:
        #     att_x = (att_x, kv_cache)
        if not self.parallel:
            if self.att is not None:
                dx, time_mix_state = self.att(self.ln1(x), x_original_cache, kv_cache, last_block_state.time_mix_state, shared)
                x = self.drop0(x + dx)
            else:
                time_mix_state = last_block_state.time_mix_state
            if self.ln2 is not None and self.ffn is not None:
                dx, channel_mix_state = self.ffn(self.ln2(x), last_block_state.channel_mix_state)
                x = self.drop0(x + dx)
            else:
                channel_mix_state = ChannelMixState()
        else:
            # parallel
            dx_att, time_mix_state = self.att(self.ln1(x), x_original_cache, last_block_state.time_mix_state, shared)
            dx_ffn, channel_mix_state = self.ffn(self.ln2(x), last_block_state.channel_mix_state)
            x = self.drop0(x + dx_att + dx_ffn)

        return x, BlockState(time_mix_state, channel_mix_state)


def causal_bias_mask(T):
    return torch.full((T, T), float('-inf')).triu(1)

def alibi_mask(T, H):
    bias = (torch.arange(T)[None, :] - torch.arange(T)[:, None]).float() # (T, T)
    bias = bias + causal_bias_mask(T) # (T, T)
    bias = bias.expand(H, -1, -1) # (H, T, T)
    head_bias_slopes = (2 ** torch.linspace(-8.0/H, -8.0, H)).unsqueeze(-1).unsqueeze(-1) # (H, 1, 1)
    bias = bias * head_bias_slopes # (H, T, T)
    return bias

class Transformer(nn.Module):
    def __init__(self, config:TrainerCLI_Config):
        super().__init__()

        self.config = config

        args:Transformer_Config = config.model

        if args.dim_att <= 0:
            args.dim_att = args.n_embd
        if args.dim_ffn <= 0:
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)  # default = 3.5x emb size

        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        mt = config.model.model_type
        self.is_cache_once = '_gold' in mt
        self.is_llama = mt.startswith('llama')

        self.shared = Shared()

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(config.model, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)


        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
        else:
            self.drop0 = nn.Identity()

        if self.is_cache_once:
            self.w_kv_cache_a = nn.Linear(args.n_embd, int(args.n_embd / args.kv_cache_compression_ratio), bias=False)
            self.w_kv_cache_b = nn.Linear(int(args.n_embd / args.kv_cache_compression_ratio) + args.n_embd, args.dim_att, bias=False)

    def ckpt(self, block, *block_args):
        if block.training and self.config.train.grad_cp == 1:
            if "deepspeed" in self.config.train.strategy:
                x, next_block_state = deepspeed.checkpointing.checkpoint(block, *block_args)
            else:
                x, next_block_state = torch.utils.checkpoint.checkpoint(block, *block_args, use_reentrant=False)
        else:
            x, next_block_state = block(*block_args)
        return x, next_block_state

    @TCompile
    def forward(self, idx, last_model_state:ModelState|None = None):
        config : Transformer_Config = self.config.model
        B, T = idx.size()

        shared = self.shared
        if config.rope is not None and shared.angles.size(0) == 0:
            shared.angles = generate_rotary_embedding(config.ctx_len, config.head_size, config.rope.base * config.rope.rebase, config.rope.rescale).to(idx.device)
        elif config.brope is not None and shared.angles.size(0) == 0:
            shared.angles = generate_binary_rotary_embedding(config.ctx_len, config.head_size, config.brope.rescale).to(idx.device)
        elif config.alibi is not None and self.bias_mask.size(0) == 0:
            shared.bias_mask = alibi_mask(config.ctx_len, self.n_kv_head).to(idx.device)

        assert (shared.angles.size(0) == 0 or T <= shared.angles.size(0)) or (shared.bias_mask.size(0) == 0 or T <= shared.bias_mask.size(0))

        x = self.emb(idx)

        total_n_layer = config.n_layer

        # might need to be true for BPTT support
        requires_grad = self.training
        if last_model_state is None:
            #dtype = x.dtype
            last_model_state = ModelState()
            for layer_id in range(total_n_layer):
                block = self.blocks[layer_id]
                last_model_state.block_states.append(BlockState(
                    block.default_time_mix_state_factory(x, config, requires_grad),
                    block.default_channel_mix_state_factory(x, config, requires_grad),
                ))
            if self.is_cache_once:
                last_model_state.input_tokens_cache = torch.zeros([B, 0], dtype=torch.long, device=idx.device, requires_grad=False)
                last_model_state.k_cache = torch.zeros([B, 0, config.dim_att], dtype=x.dtype, device=x.device, requires_grad=requires_grad)

        x = self.drop0(x)
        x = self.blocks[0].ln0(x)
        x_original_chunk = x
        
        k_cache = last_model_state.k_cache
        next_model_state = ModelState()
        if self.is_cache_once:
            x_original_from_input_cache = torch.cat( [
                self.blocks[0].ln0(self.drop0(self.emb( last_model_state.input_tokens_cache ))),
                x
            ], dim=-2)
            next_model_state.input_tokens_cache = torch.cat([last_model_state.input_tokens_cache, idx], dim=-1)
        else:
            x_original_from_input_cache = torch.tensor([])
        for layer_id in range(total_n_layer):
            block = self.blocks[layer_id]

            x, next_block_state = self.ckpt(block, x, x_original_from_input_cache, k_cache, last_model_state, shared)
            if self.is_cache_once and layer_id == get_second_submodel_layer_id(config) - 1:
                compressed_k_cache_chunk = self.w_kv_cache_a(x)
                k_tokencat_chunk = torch.cat([x_original_chunk, compressed_k_cache_chunk],dim=-1)
                k_cache_chunk = rms_norm(self.w_kv_cache_b(k_tokencat_chunk))

                k_cache = torch.cat([k_cache, k_cache_chunk], dim=-2)

            next_model_state.block_states.append(next_block_state)

        x = self.ln_out(x)
        x = self.head(x)
        next_model_state.k_cache = k_cache
        return x, next_model_state

    def get_optim_groups(self):
        train_config = self.config.train

        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if (("_w1" in n) or ("_w2" in n)) and (train_config.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (train_config.layerwise_lr > 0):
                if train_config.train_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (train_config.layerwise_lr > 0):
                if train_config.train_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (train_config.layerwise_lr > 0):
                if train_config.train_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (train_config.layerwise_lr > 0):
                lr_3x.add(n)
            elif ('.A_log' in n) or n.endswith('.bias'): # mamba
                lr_1x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (train_config.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        param_dict = {n: p for n, p in self.named_parameters()}
        param_check = list(lr_decay) + list(lr_1x) + list(lr_2x) + list(lr_3x)
        if not train_config.load_partial:
            assert sorted(param_dict) == sorted(param_check)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        
        print('decay', lr_decay, '\n')
        print('1x', lr_1x, '\n')
        print('2x', lr_2x, '\n')
        print('3x', lr_3x, '\n')

        
        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0, 'name':'lr_1x'},
        ]
        if len(lr_2x) > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0, 'name':'lr_2x'}]
        if len(lr_3x) > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0, 'name':'lr_3x'}]
        if len(lr_decay) > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": train_config.weight_decay, "my_lr_scale": 1.0, 'name':'lr_decay'}]

        return optim_groups

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        n_params = 0
        state_dict = self.state_dict()
        for n in state_dict:
            p = state_dict[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {n}", end="")

            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias'):
                if 'ln_x' in n and n.endswith('.weight'):
                    layer_scale = (1+int(n.split('.')[1])) / self.config.model.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
                print()
            elif n == "emb.weight":
                m[n] = p
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale)
                print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                if self.config.model.vocab_size > self.config.model.n_embd:
                    scale = 0.5 * math.sqrt(self.config.model.vocab_size / self.config.model.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            elif 'mamba' in self.config.model.model_type and ((not self.is_cache_once) or (n.startswith('blocks') and int(n.split('.')[1]) < get_second_submodel_layer_id(self.config.model))):
                m[n] = p
                if '.out_proj.weight' in n:
                    scale = 0
                    nn.init.zeros_(m[n])
                    print(f" [scale {scale}]")
                    # nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    # with torch.no_grad():
                    #     n_residuals_per_layer = 2
                    #     p /= math.sqrt(n_residuals_per_layer * self.config.model.n_layer)
                    # print(f" [scale special residual]")
                elif '.bias' in n:# and 'dt_proj.bias' not in n:
                    scale = 0
                    nn.init.zeros_(m[n])
                    print(f" [scale {scale}]")
                else:
                    print()
            elif len(p.shape) > 2 or "sin" in n or "cos" in n or "freqs" in n:
                m[n] = p
                print()
            else:
                assert n.endswith('.weight') # should always be true

                zero = [".att.output.", ".ffn.value.", ".ffn.receptance."]

                for kk in zero:
                    if kk in n:
                        scale = 0

                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")

                if self.config.train.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()
            n_params += m[n].numel()

            # if n == "emb.weight":
            #     print(m[n])

        print('model params', n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m

