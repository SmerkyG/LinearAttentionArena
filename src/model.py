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
from lightning import Trainer

from .state import ModelState, BlockState, ChannelMixState, TimeMixState, Shared

from configs import TrainerCLI_Config, Model_Config, Transformer_Config, Train_Config

from .rotary import generate_rotary_embedding, generate_binary_rotary_embedding, apply_rotary_embedding
from .norm import rms_norm

import numpy as np

if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from .CoreDependencies import *

from src.logger import print0 as print

from moe.utils import split_params_into_different_moe_groups_for_optimizer

try:
    print('RWKV_MODEL_TYPE', os.environ["RWKV_MODEL_TYPE"])
except:
    os.environ["RWKV_MODEL_TYPE"] = ''

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################

def get_second_submodel_layer_id(model_config:Model_Config):
    return int(model_config.n_layer * (model_config.inv_other_layer_ratio - 1) / model_config.inv_other_layer_ratio)

from pydoc import locate

class Block(nn.Module):
    def __init__(self, config:Model_Config, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.parallel = config.parallel

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)

        if config.tmix2 == '' or layer_id < get_second_submodel_layer_id(config):
            # first layer type
            tmix = config.tmix
            cmix = config.cmix
            cmoe = config.cmoe
        else:
            # second layer type
            tmix = config.tmix2
            cmix = config.cmix2       
            cmoe = config.cmoe2
        
        if tmix == '':
            tmix_factory = lambda config, layer_id: None
        else:
            tmix_typepath = f'tmix.tmix_{tmix}.TMix_{tmix}'
            tmix_factory = locate(tmix_typepath)
            if tmix_factory is None:
                print(f"Unsupported tmix component type: {tmix_typepath}")
                exit(0)
        tmix:nn.Module = tmix_factory(config, layer_id)
        
        if cmix == '':
            cmix_factory = lambda config, layer_id: None
        else:
            cmix_typepath = f'cmix.cmix_{cmix}.CMix_{cmix}'
            cmix_factory = locate(cmix_typepath)
            if cmix_factory is None:
                print(f"Unsupported cmix component type: {cmix_typepath}")
                exit(0)
        cmix:nn.Module = cmix_factory(config, layer_id)

        if cmoe == '' or config.num_experts <= 0:
            cmoe_factory = lambda config, layer_id: None
        else:
            cmoe_typepath = f'cmoe.cmoe_{cmoe}.CMoE_{cmoe}'
            cmoe_factory = locate(cmoe_typepath)
            if cmoe_factory is None:
                print(f"Unsupported cmoe component type: {cmoe_typepath}")
                exit(0)
        cmoe:nn.Module = cmoe_factory(config, layer_id)

        self.is_cache_once = config.tmix2 == 'gold'

        self.default_time_mix_state_factory = tmix.get_default_state_factory() if hasattr(tmix, 'get_default_state_factory') else lambda x, c, r: TimeMixState()
        self.att = TJIT(tmix)
        
        if cmix is not None:
            self.default_channel_mix_state_factory = cmix.get_default_state_factory() if hasattr(cmix, 'get_default_state_factory') else lambda x, c, r: ChannelMixState()
        elif cmoe is not None:
            self.default_channel_mix_state_factory = cmoe.get_default_state_factory() if hasattr(cmoe, 'get_default_state_factory') else lambda x, c, r: ChannelMixState()
        else:
            self.default_channel_mix_state_factory = lambda x, c, r: ChannelMixState()
        self.ffn = TJIT(cmix)

        self.cmoe = cmoe

        if config.dropout > 0:
            self.drop0 = nn.Dropout(p = config.dropout)
            self.drop1 = nn.Dropout(p = config.dropout)
        else:
            self.drop0 = nn.Identity()
            self.drop1 = nn.Identity()

    @TCompile
    def forward(self, x, x_original_cache, k_cache, last_model_state:ModelState, shared:Shared):
        last_block_state:BlockState = last_model_state.block_states[self.layer_id]

        if not self.parallel:
            if self.att is not None:
                dx, time_mix_state = self.att(self.ln1(x), x_original_cache, k_cache, last_model_state, shared)
                x = self.drop0(x + dx)
            else:
                time_mix_state = last_block_state.time_mix_state
            ln2x = self.ln2(x)
            if self.ffn is not None:
                dffn_x, channel_mix_state = self.ffn(ln2x, last_model_state)
                x = x + dffn_x
            if self.cmoe is not None:
                dcmoe_x, channel_mix_state = self.cmoe(ln2x, last_model_state)
                x = x + dcmoe_x
            if self.ffn is None and self.cmoe is None:
                channel_mix_state = ChannelMixState()
            
            x = self.drop0(x)
        else:
            # parallel
            dx_att, time_mix_state = self.att(self.ln1(x), x_original_cache, k_cache, last_model_state, shared)
            dx_ffn, channel_mix_state = self.ffn(self.ln2(x), last_model_state)
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

        self.is_cache_once = args.tmix2 == 'gold'

        self.shared = Shared()

    # def configure_model(self):
    #     if hasattr(self, 'emb'):
    #         return

    #     config = self.config
    #     args:Transformer_Config = self.config.model

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        blocks = [Block(config.model, i) for i in range(args.n_layer)]
        self.blocks = nn.ModuleList([block for block in blocks])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)


        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
        else:
            self.drop0 = nn.Identity()

        if self.is_cache_once:
            self.w_kv_cache_a = nn.Linear(args.n_embd, int(args.n_embd / args.kv_cache_compression_ratio), bias=False)
            self.w_kv_cache_b = nn.Linear(int(args.n_embd / args.kv_cache_compression_ratio) + args.n_embd, args.dim_att, bias=False)

        if self.training and config.train is not None and (config.train.load_partial or getattr(config.train, 'ckpt_path', None) in [None, '']):
            self.init_weights()

    def ckpt(self, block, *block_args):
        if block.training and self.config.train.grad_cp == 1:
            if "deepspeed" in self.config.train.strategy:
                x, next_block_state = deepspeed.checkpointing.checkpoint(block, *block_args)
            else:
                x, next_block_state = torch.utils.checkpoint.checkpoint(block, *block_args, use_reentrant=False)
        else:
            x, next_block_state = block(*block_args)
        return x, next_block_state

    def forward(self, token_ids:Tensor|list, last_model_state:ModelState|None = None):
        config : Transformer_Config = self.config.model
        if isinstance(token_ids, Tensor):
            B, T = token_ids.size()
        else:
            B = 1
            T = len(token_ids)
            token_ids = torch.tensor(token_ids, device=self.emb.weight.device, dtype=torch.long, requires_grad=False)[None, :]

        shared = self.shared
        if config.rope is not None and shared.angles.size(0) == 0:
            shared.angles = generate_rotary_embedding(config.ctx_len, config.head_size, config.rope.base * config.rope.rebase, config.rope.rescale).to(token_ids.device)
        elif config.brope is not None and shared.angles.size(0) == 0:
            shared.angles = generate_binary_rotary_embedding(config.ctx_len, config.head_size, config.brope.rescale).to(token_ids.device)
        elif config.alibi is not None and self.bias_mask.size(0) == 0:
            shared.bias_mask = alibi_mask(config.ctx_len, self.n_kv_head).to(token_ids.device)

        assert (shared.angles.size(0) == 0 or T <= shared.angles.size(0)) or (shared.bias_mask.size(0) == 0 or T <= shared.bias_mask.size(0))

        x = self.emb(token_ids)

        total_n_layer = config.n_layer

        # might need to be true in the future for BPTT support
        requires_grad = self.training
        if last_model_state is None:
            last_model_state = ModelState()
            for layer_id in range(total_n_layer):
                block = self.blocks[layer_id]
                last_model_state.block_states.append(BlockState(
                    block.default_time_mix_state_factory(x, config, requires_grad),
                    block.default_channel_mix_state_factory(x, config, requires_grad),
                ))
            if config.num_experts > 0 or self.is_cache_once:
                last_model_state.input_tokens_cache = torch.zeros([B, 0], dtype=torch.long, device=token_ids.device, requires_grad=False)
            if self.is_cache_once:
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
        else:
            x_original_from_input_cache = torch.zeros([B, 0, config.dim_att], dtype=x.dtype, device=x.device)

        if config.num_experts > 0:
            last_model_state.input_tokens_cache = torch.cat([last_model_state.input_tokens_cache, token_ids], dim=-1)

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
        next_model_state.input_tokens_cache = last_model_state.input_tokens_cache
        return x, next_model_state

    def get_optim_groups(self):
        train_config = self.config.train

        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        lr2 = set()
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if '.moe.' in n or '.ffn.ffn.' in n:
                lr2.add(n)
            #if train_config.load_partial and self.is_cache_once and ('.att.' in n or '.ffn.' in n) and int(n.split('.')[1]) >= get_second_submodel_layer_id(self.config.model):
            #    lr2.add(n)
            # elif self.is_cache_once and ('ln_out.' in n or 'head.' in n):
            #     lr2.add(n)
            elif train_config.load_partial and '.cmoe.' in n:
                lr2.add(n)
            elif (("_w1" in n) or ("_w2" in n)) and (train_config.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (train_config.layerwise_lr > 0):
                # if train_config.train_stage == 2:
                #     lr_2x.add(n)
                # else:
                lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (train_config.layerwise_lr > 0):
                # if train_config.train_stage == 2:
                #     lr_3x.add(n)
                # else:
                lr_2x.add(n)
            elif ("time_faaaa" in n) and (train_config.layerwise_lr > 0):
                # if train_config.train_stage == 2:
                #     lr_2x.add(n)
                # else:
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
        param_check = list(lr_decay) + list(lr_1x) + list(lr_2x) + list(lr_3x) + list(lr2)
        if not train_config.load_partial:
            assert sorted(param_dict) == sorted(param_check)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        lr2 = sorted(list(lr2))
        
        print('decay', lr_decay, '\n')
        print('1x', lr_1x, '\n')
        print('2x', lr_2x, '\n')
        print('3x', lr_3x, '\n')
        print('lr2', lr2, '\n')
        
        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0, 'name':'lr_1x'},
        ]
        if len(lr_2x) > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0, 'name':'lr_2x'}]
        if len(lr_3x) > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0, 'name':'lr_3x'}]
        if len(lr2) > 0:
            # NOTE - split params for DeepSpeed MoE messes with the group name! so can't rely on that to be the same
            optim_groups += [{"params": [param_dict[n] for n in lr2], "weight_decay": 0.0, "my_lr_scale": 1.0, 'name':'lr2', 'lrtype':'lr2'}]
        if len(lr_decay) > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": train_config.weight_decay, "my_lr_scale": 1.0, 'name':'lr_decay'}]

        if self.config.model.num_experts > 1:
            optim_groups = split_params_into_different_moe_groups_for_optimizer(optim_groups)

        return optim_groups

    #
    # Custom overwrite of manual_backwards operation, to skip the "manual_backwards"
    # safety check, so we can perform manual backward operation step, while using
    # the default trainer loop. This is modified from the original code found here:
    # https://github.com/Lightning-AI/lightning/blob/37c244f94be365496def82870b22c2faf0ab889e/src/lightning/pytorch/core/module.py#L999
    #
    # ---
    # 
    # This allow us to avoid disabling the "automatic_optimization" flag
    #
    # Which would have been required to do "segmented learning", or "Backpropagation Through Time"
    # where we would need to implement manual optimization as per
    # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    #
    # Otherwise an error will be thrown if we call `self.manual_backward`
    #
    # However this would mean that we would need to do a full reimplementation
    # of several features that were handled by the automatic optimization.
    # - accumulate_grad_batches
    # - gradient_clip_val
    # - logging behaviour
    # - distributed training co-ordination
    # - (And probably other features that I am not aware of)
    #
    # So this is a hacky work around, to avoid reimplementing all of the above.
    # 
    # From the current code implementatiion, it seem like this is blocked only by 
    # automatic_optimization flag - and has no adverse side effect otherwise
    # https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/core/module.html#LightningModule.manual_backward
    #
    # If anyone have a better idea, let me know
    # (have experimented with, reimplementing the above, but it is not trivial, unfortunately)
    #
    def manual_backward(self, loss: torch.Tensor, *args, **kwargs):
        if self._fabric:
            self._fabric.backward(loss, *args, **kwargs)
        else:
            # self._verify_is_manual_optimization("manual_backward")
            self.trainer.strategy.backward(loss, None, *args, **kwargs)
            
    def init_weights(self):
        for name, m in self.named_modules():               
            scale = 1.0
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

                for kk in [".att.output", ".ffn.value", ".ffn.receptance", ".ffn_value", ".ffn_receptance"]:
                    if name.endswith(kk):
                        scale = 0

                for kk in [".att.key"]:
                    if name.endswith(kk):
                        scale = 0.1

                if name == "head":
                    if self.config.model.vocab_size > self.config.model.n_embd:
                        scale = 0.5 * math.sqrt(self.config.model.vocab_size / self.config.model.n_embd)
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
                layer_id = int(name.split('.')[1])
                layer_scale = (1+layer_id) / self.config.model.n_layer
                m.weight = nn.Parameter((m.weight * 0.0) + (layer_scale ** 0.7))
                print(name, "layer_scale init")
            else:
                print(name, "(default init)")
                
#     def generate_init_weight(self):
#         print(
#             f"""
# ############################################################################
# #
# # Init model weight (slow for large models)...
# #
# ############################################################################
# """
#         )
#         m = {}
#         n_params = 0
#         state_dict = self.state_dict()
#         for n in state_dict:
#             p = state_dict[n]
#             shape = p.shape

#             s0 = str(shape[0]) if len(shape) > 0 else ""
#             s1 = str(shape[1]) if len(shape) > 1 else ""
#             s2 = str(shape[2]) if len(shape) > 2 else ""
#             print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {n}", end="")

#             scale = 1.0
#             if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias'):
#                 if 'ln_x' in n and n.endswith('.weight'):
#                     layer_scale = (1+int(n.split('.')[1])) / self.config.model.n_layer
#                     m[n] = (p * 0.0) + (layer_scale ** 0.7)
#                 else:
#                     m[n] = p
#                 print()
#             elif n == "emb.weight":
#                 m[n] = p
#                 scale = -1e-4
#                 nn.init.uniform_(m[n], a=scale, b=-scale)
#                 print(f" [scale {scale}]")
#             elif n == "head.weight":
#                 m[n] = p
#                 if self.config.model.vocab_size > self.config.model.n_embd:
#                     scale = 0.5 * math.sqrt(self.config.model.vocab_size / self.config.model.n_embd)
#                 else:
#                     scale = 0.5
#                 nn.init.orthogonal_(m[n], gain=scale)
#                 print(f" [scale {scale}]")
#             elif 'mamba' in self.config.model.tmix and ((not self.is_cache_once) or (n.startswith('blocks') and int(n.split('.')[1]) < get_second_submodel_layer_id(self.config.model))):
#                 m[n] = p
#                 if '.out_proj.weight' in n:
#                     scale = 0
#                     nn.init.zeros_(m[n])
#                     print(f" [scale {scale}]")
#                     # nn.init.kaiming_uniform_(p, a=math.sqrt(5))
#                     # with torch.no_grad():
#                     #     n_residuals_per_layer = 2
#                     #     p /= math.sqrt(n_residuals_per_layer * self.config.model.n_layer)
#                     # print(f" [scale special residual]")
#                 elif '.bias' in n:# and 'dt_proj.bias' not in n:
#                     scale = 0
#                     nn.init.zeros_(m[n])
#                     print(f" [scale {scale}]")
#                 else:
#                     print()
#             elif len(p.shape) > 2 or "sin" in n or "cos" in n or "freqs" in n:
#                 m[n] = p
#                 print()
#             else:
#                 assert n.endswith('.weight') # should always be true

#                 zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffn_value.", ".ffn_receptance."]

#                 for kk in zero:
#                     if kk in n:
#                         scale = 0

#                 for kk in [".att.key."]:
#                     if kk in n:
#                         scale = 0.1
#                 for kk in [".att.gate."]:
#                     if kk in n:
#                         scale = 0.1

#                 print(f" [scale {scale}]")

#                 if self.config.train.accelerator.upper() == "GPU":
#                     m[n] = torch.empty((shape[0], shape[1]), device="cuda")
#                 else:
#                     m[n] = torch.empty((shape[0], shape[1]))

#                 if scale == 0:
#                     nn.init.zeros_(m[n])
#                 elif scale < 0:
#                     nn.init.uniform_(m[n], a=scale, b=-scale)
#                 else:
#                     nn.init.orthogonal_(m[n], gain=scale)

#             m[n] = m[n].cpu()
#             if os.environ["RWKV_FLOAT_MODE"] == "fp16":
#                 m[n] = m[n].half()
#             elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
#                 m[n] = m[n].bfloat16()
#             n_params += m[n].numel()

#             # if n == "emb.weight":
#             #     print(m[n])

#         print('model params', n_params)
#         gc.collect()
#         torch.cuda.empty_cache()
#         return m

