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

from .tmix import TimeMixState, ModelState
from .cmix import ChannelMixState
from .tmix_x052 import RWKV_Tmix_x052
from .tmix_x060 import RWKV_Tmix_x060
from .tmix_x060bbswa import RWKV_Tmix_x060bbswa
from .tmix_x060b import RWKV_Tmix_x060b
from .tmix_x060b2 import RWKV_Tmix_x060b2
from .tmix_x060b5 import RWKV_Tmix_x060b5
from .tmix_x060o3 import RWKV_Tmix_x060o3
from .tmix_taylor import RWKV_Tmix_taylor
from .tmix_taylorchunked import RWKV_Tmix_taylorchunked
from .tmix_attn import RWKV_Tmix_attn
from .tmix_poco import RWKV_Tmix_poco

from .cmix_x052 import RWKV_CMix_x052
from .cmix_x060 import RWKV_CMix_x060

import src.metrics as metrics

import numpy as np

if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from .CoreDependencies import *

def console_clear_last_line():
    print('\033[1A', end='\x1b[2K')

try:
    print('RWKV_MODEL_TYPE', os.environ["RWKV_MODEL_TYPE"])
except:
    os.environ["RWKV_MODEL_TYPE"] = ''

########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])

if 'mamba' in os.environ["RWKV_MODEL_TYPE"]:
    from mamba_ssm import Mamba

def rms_norm(x, eps:float = 1e-8):
    rms_norm = (x.size(-1) ** -0.5) * x.norm(2, dim=-1, keepdim=True)
    return x / (rms_norm + eps)

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################

class BlockState:
    def __init__(self, time_mix_state: TimeMixState, channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state

def get_first_poco_layer_id(n_layer):
    return (2*n_layer)//3

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.parallel = False

        ffnFactory = lambda: RWKV_CMix_x060(args, layer_id)

        mt = os.environ["RWKV_MODEL_TYPE"]
        if 'parallel' in mt:
            self.parallel = True

        if mt.startswith('x060bbswa'):
            attFactory = lambda: RWKV_Tmix_x060bbswa(args, layer_id)
        elif mt.startswith('x060b5'):
            attFactory = lambda: RWKV_Tmix_x060b5(args, layer_id)
        elif mt.startswith('x060b2'):
            attFactory = lambda: RWKV_Tmix_x060b2(args, layer_id)
        elif mt.startswith('x060b'):
            attFactory = lambda: RWKV_Tmix_x060b(args, layer_id)
        elif mt.startswith('x060o3'):
            attFactory = lambda: RWKV_Tmix_x060o3(args, layer_id)
        elif mt.startswith('x052'):
            attFactory = lambda: RWKV_Tmix_x052(args, layer_id)
            ffnFactory = lambda: RWKV_CMix_x052(args, layer_id)
        elif mt.startswith('x060'):
            attFactory = lambda: RWKV_Tmix_x060(args, layer_id)
        elif mt.startswith('attn'):
            attFactory = lambda: RWKV_Tmix_attn(args, layer_id)
        elif mt.startswith('mamba'):
            attFactory = lambda: Mamba(d_model=args.n_embd, d_state=16, d_conv=4, expand=2)
            ffnFactory = lambda: Mamba(d_model=args.n_embd, d_state=16, d_conv=4, expand=2)
        else:
            print(f"Unsupported model type: {mt}")
            exit(0)
        
        if 'taylor' in mt:
            if layer_id >= args.n_layer * 2 // 3 - 1 and layer_id < args.n_layer - 1:
                if 'taylorchunked' in mt:
                    attFactory = lambda: RWKV_Tmix_taylorchunked(args, layer_id)
                else:
                    attFactory = lambda: RWKV_Tmix_taylor(args, layer_id)

        if '_attn' in mt:
            if layer_id >= args.n_layer // 2:
                attFactory = lambda: RWKV_Tmix_attn(args, layer_id)

        pocoFactory = lambda: None
        self.is_poco = '_poco' in mt
        if self.is_poco:
            if layer_id >= get_first_poco_layer_id(args.n_layer) and layer_id < args.n_layer:
                attFactory = lambda: None
                if '_pocobha' in mt:
                    pocoFactory = lambda: RWKV_Tmix_pocobha(args, layer_id)
                else:
                    pocoFactory = lambda: RWKV_Tmix_poco(args, layer_id)

        self.att = attFactory()
        self.poco = pocoFactory()
        
        if ffnFactory is None:
            self.ln2 = None
            self.ffn = None
        else:
            self.ffn = ffnFactory()


        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)
        else:
            self.drop0 = nn.Identity()
            self.drop1 = nn.Identity()


    @TCompile
    def forward(self, x, kv_cache, last_model_state:ModelState):
        args = self.args

        B, T, C = x.size()
        # if self.layer_id == 0:
        #     x = self.ln0(x)

        last_block_state = last_model_state.block_states[self.layer_id]

        # if len(kv_cache.shape) > 0:
        #     att_x = (att_x, kv_cache)
        if not self.parallel:
            if self.att is not None:
                dx, time_mix_state = self.att(self.ln1(x), last_block_state.time_mix_state)
                x = self.drop0(x + dx)
            elif self.poco is not None:
                time_mix_state = last_block_state.time_mix_state
                dx, time_mix_state = self.poco(self.ln1(x), kv_cache, last_block_state.time_mix_state)
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
            dx_att, time_mix_state = self.att(self.ln1(x), last_block_state.time_mix_state)
            dx_ffn, channel_mix_state = self.ffn(self.ln2(x), last_block_state.channel_mix_state)
            x = self.drop0(x + dx_att + dx_ffn)

        return x, BlockState(time_mix_state, channel_mix_state)


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.metrics = dict(loss=metrics.Loss(), acc=metrics.Accuracy())

        self.args = args

        if args.dim_att <= 0:
            args.dim_att = args.n_embd
        if args.dim_ffn <= 0:
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)  # default = was 3.5x emb size, now 4x without gate

        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        mt = os.environ["RWKV_MODEL_TYPE"]
        self.is_poco = '_poco' in mt

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
        else:
            self.drop0 = nn.Identity()

        if self.is_poco:
            #self.ln_kv_cache = nn.LayerNorm(args.n_embd)
            #self.w_kv_cache = nn.Linear(args.n_embd, 2 * args.dim_att, bias=False)
            self.w_kv_cache_a = nn.Linear(args.n_embd * 2, args.n_embd // 16, bias=False)
            self.w_kv_cache_b = nn.Linear(args.n_embd // 16 + args.n_embd, 2 * args.dim_att, bias=False)

            with torch.no_grad():
                ddd = torch.ones(1, 1, args.n_embd)
                for i in range(args.n_embd):
                    ddd[0, 0, i] = i / args.n_embd

                self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, 0.5))
                self.time_maa_token = nn.Parameter(1.0 - torch.pow(ddd, 0.5))
                D_MIX_LORA = 32
                self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA))
                self.time_maa_w2 = nn.Parameter(torch.zeros(D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))


    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.train_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
                if args.train_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.train_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif ('.A_log' in n) or n.endswith('.bias'): # mamba
                lr_1x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        param_dict = {n: p for n, p in self.named_parameters()}
        param_check = list(lr_decay) + list(lr_1x) + list(lr_2x) + list(lr_3x)
        assert sorted(param_dict) == sorted(param_check)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        
        if self.trainer.is_global_zero:
            print('decay', lr_decay, '\n')
            print('1x', lr_1x, '\n')
            print('2x', lr_2x, '\n')
            print('3x', lr_3x, '\n')

        
        if args.layerwise_lr > 0:
            if args.train_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, idx, last_model_state:ModelState|None = None):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        dtype = x.dtype#torch.get_autocast_gpu_dtype()

        total_n_layer = args.n_layer

        # FIXME - might need to be true later for BPTT
        requires_grad = self.training
        if last_model_state is None:
            #dtype = x.dtype
            last_model_state = ModelState()

            wkv_state_shape = [B, args.dim_att//args.head_size_a, args.head_size_a, args.head_size_a]
            last_model_state.block_states = [
                BlockState(
                    TimeMixState(
                        torch.zeros(wkv_state_shape, dtype=dtype, device=idx.device, requires_grad=requires_grad), 
                        torch.zeros([B, x.size(-1)], dtype=dtype, device=idx.device, requires_grad=requires_grad)
                    ), 
                    ChannelMixState(
                        torch.zeros([B, x.size(-1)], dtype=dtype, device=idx.device, requires_grad=requires_grad)
                    )
                ) 
                for layer_id in range(total_n_layer)
            ]
            if self.is_poco:
                # FIXME - need max ctx len not just training ctx_len?
                kv_cache_len = 0# if self.training else self.args.ctx_len
                last_model_state.kv_cache = torch.zeros([B, kv_cache_len, args.dim_att * 2], dtype=dtype, device=idx.device, requires_grad=requires_grad)
                last_model_state.embed_state = torch.zeros([B, args.n_embd], dtype=dtype, device=idx.device, requires_grad=requires_grad)

        x = self.drop0(x)
        x = self.blocks[0].ln0(x)
        x_original = x
        
        kv_cache = last_model_state.kv_cache
        embed_state = last_model_state.embed_state
        next_model_state = ModelState()
        for layer_id in range(total_n_layer):
            block = self.blocks[layer_id]

            block_args = [x, kv_cache, last_model_state]
            if self.training and args.grad_cp == 1:
                if "deepspeed" in args.strategy:
                    x, next_block_state = deepspeed.checkpointing.checkpoint(block, *block_args)
                else:
                    x, next_block_state = torch.utils.checkpoint.checkpoint(block, *block_args, use_reentrant=False)
            else:
                x, next_block_state = block(*block_args)
            if self.is_poco and layer_id == get_first_poco_layer_id(args.n_layer) - 1:
                # FIXME - we really need a separate shift state now for the kv_cache
                #dx_original_prev = F.pad(x_original, (0, 0, 1, -1)) - x_original                
                dx_original_prev = torch.concat((embed_state.unsqueeze(1), x_original[:, :-1]), dim=1) - x_original
                embed_state = x_original[:, -1].clone()
                
                xxx = x_original + dx_original_prev * self.time_maa_x
                xxx = torch.tanh(xxx @ self.time_maa_w1) @ self.time_maa_w2
                mtoken = xxx
                xtoken = x_original + dx_original_prev * (self.time_maa_token + mtoken)

                new_compressed_kv_cache = self.w_kv_cache_a(torch.cat([xtoken, x],dim=-1))
                new_kv_cache = rms_norm(self.w_kv_cache_b(rms_norm(torch.cat([xtoken, new_compressed_kv_cache],dim=-1))))

                # FIXME - preallocate and edit in place instead?
                if self.training:
                    # FIXME - cat instead?
                    kv_cache = new_kv_cache
                    pass
                else:
                    kv_cache = torch.cat([kv_cache, new_kv_cache], dim=-2)
                # next_model_state.kv_cache = last_model_state.kv_cache
                # last_model_state.seq_pos = last_model_state.seq_pos + T
                # next_model_state.seq_pos = last_model_state.seq_pos

            next_model_state.block_states.append(next_block_state)

        x = self.ln_out(x)
        x = self.head(x)
        next_model_state.kv_cache = kv_cache
        next_model_state.embed_state = embed_state
        return x, next_model_state

    def _get_loss_logits_preds(self, batch, batch_idx):
        x, y = batch
        logits, next_block_states = self(x)
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.flatten())
        with torch.no_grad():
            preds = logits.argmax(dim=-1)

        if loss.isinf().any():
            raise Exception("loss was infinite")

        if loss.isnan().any():
            raise Exception("loss was NaN")

        return loss, logits, preds, next_block_states
    
    def get_real_global_step(self): return int(self.trainer.global_step + self.args.epoch_begin * self.args.epoch_steps)
    def get_real_tokens(self): return self.get_real_global_step() * self.args.ctx_len * self.args.real_bsz

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        loss, logits, preds, next_block_states = self._get_loss_logits_preds(batch, batch_idx)

        margs = metrics.MetricArgs(inputs, logits, preds, labels, loss)
        # FIXME - sync from other devices/nodes here
        for metric in self.metrics.values():
            metric.update(margs)

        if self.trainer.is_global_zero:
            if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
                # stupid hack bc lightning expects us to log every actual step
                for name, metric in self.metrics.items():
                    metric_value = metric.compute()
                    self.log(name, metric_value, on_step=True, rank_zero_only=True)
                if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
                    logdict = dict(tokens = self.get_real_tokens())
                    #str = f"epoch:{self.current_epoch} token:{self.all_nodes_tokens_processed:,} step:{batch_idx} "
                    for name, metric in self.metrics.items():
                        metric_value = metric.compute()
                        logdict['train/' + name] = metric_value
                        metric.clear()
                        #str += f'{name}:{metric_value:.4f} '
                    #str += f"{gb:.1f}gb {int(ms_per)}ms {ktok_per_sec:.2f}kT/s {self.total_runtime:.1f}sec"
                    #print(str)
                    if len(self.args.wandb) > 0:
                        self.trainer.my_wandb.log(logdict, step=self.get_real_global_step())

        return L2Wrap.apply(loss, logits)

    def on_validation_epoch_start(self):
        if self.trainer.is_global_zero:
            print(f"STARTING VALIDATION")
            print()

            # clear metrics
            for metric in self.metrics.values():
                metric.compute()

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            logdict = dict(tokens = self.get_real_tokens())
            str = f"VALIDATION COMPLETE. "
            for name, metric in self.metrics.items():
                metric_value = metric.compute()
                logdict["val/" + name] = metric_value
                str += f"{metric_value:.4f} "
                metric.clear()
            if len(self.args.wandb) > 0:
                self.trainer.my_wandb.log(logdict, step=self.get_real_global_step())

            console_clear_last_line()
            print(str)
            print()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        loss, logits, preds, next_block_states = self._get_loss_logits_preds(batch, batch_idx)
        margs = metrics.MetricArgs(inputs, logits, preds, labels, loss)
        for name, metric in self.metrics.items():
            metric.update(margs)
            # on_epoch causes this to be logged in aggregate rather than per batch
            #self.log('val/'+name, metric.compute(), on_epoch=True, rank_zero_only=True)
            #metric.clear()
        #self.log("tokens", float(self.all_nodes_tokens_processed), on_epoch=True, rank_zero_only=True)
        return loss

    # def training_step_end(self, batch_parts):
    #     if pl.__version__[0]!='2':
    #         all = self.all_gather(batch_parts)
    #         if self.trainer.is_global_zero:
    #             self.trainer.my_loss_all = all

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
            if len(p.shape) > 2 or "sin" in n or "cos" in n or "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias'):
                if 'ln_x' in n and n.endswith('.weight'):
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
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
                if self.args.vocab_size > self.args.n_embd:
                    scale = 0.5 * math.sqrt(self.args.vocab_size / self.args.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                if 'mamba' in os.environ["RWKV_MODEL_TYPE"]:
                    m[n] = p
                    if '.out_proj.weight' in n:
                        scale = 0
                        nn.init.zeros_(m[n])
                        print(f" [scale {scale}]")
                    elif '.bias' in n:
                        scale = 0
                        nn.init.zeros_(m[n])
                        print(f" [scale {scale}]")
                    else:
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

                    if self.args.accelerator.upper() == "GPU":
                        m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                    else:
                        m[n] = torch.empty((shape[0], shape[1]))

                    if scale == 0:
                        nn.init.zeros_(m[n])
                    elif scale < 0:
                        nn.init.uniform_(m[n], a=scale, b=-scale)
                    else:
                        nn.init.orthogonal_(m[n], gain=scale)

            # if 'blocks.23' in n or 'blocks.' not in n:
            #     print(n, m[n])

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

# SimpleRWKV specific imports
from transformers import PreTrainedTokenizerFast

# Current script dir
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../'))

# SimpleRWKV is a wrapper for RWKV that allows for simple usage of the model
#
# it is not meant to be highly performant, but rather a simple minimal way to run the RWKV trainer module
# in inference mode, and can be used to validate the model trainer code / its changes
class SimpleRWKV():

    def __init__(
            self,
            model,
            args,
            ctx_len:int = 1024,
            device:str = "cuda",
            dtype_str:str = "fp32"
        ):

        self.model = model

        # Log the mismatch dtype
        dtype = torch.float32
        if dtype_str == "16":
            dtype = torch.float16
        elif dtype_str == "bf16":
            dtype = torch.bfloat16
        elif dtype_str == "32":
            dtype = torch.float32
        else:
            print("[SimpleRWKV] Warning: dtype mismatch, only fp16 bf16 fp32 is supported (for now)")

        # Prepare the model config with the model path, and custom torch load
        #model_config = {}
        #model_config["load_model"] = model_path
        #model_config["ctx_len"] = ctx_len

        # FIXME
        #model_config["version"] = "6.0"
        #model_config["strict_loading"] = False
        #model_config["num_experts"] = 8

        # This feature depends on deepspeed
        #model_config["grad_cp"] = False
        # model_config["_torch_load_state"] = loaded_state

        # Save the config settings
        self.ctx_len = ctx_len
        self.device = device

        # Lets actually load the model
        #trainer = Trainer(precision=dtype_str, accelerator='cuda', devices=1)
        #fabric = Lightning.Fabric(precision=dtype_str, accelerator='cuda', devices=1)
        #with fabric.init_module():
        print("dtype of model itself started as ", self.model.ln_out.weight.dtype)

        # Lets map it over to the respective device type
        # and set it to run as eval/inference mode
        print("Desired dtype", dtype)
        self.model.to(dtype)
        self.model.to(device)
        self.model.eval()
        if dtype != torch.float:
            torch.set_autocast_gpu_dtype(dtype)

        print("dtype of model itself became ", self.model.ln_out.weight.dtype)

        # The tokenizer object values
        self.fastTokenizer = None
        self.worldTokenizer = None

        # Setup the tokenizer
        if args.vocab_size == 50277:
            # Use the neox tokenizer
            tokenizer_file = os.path.join(SCRIPT_DIR,"./dataflow/20B_tokenizer.json")
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
            self.fastTokenizer = tokenizer
        elif args.vocab_size == 65536:
            # Use the world tokenizer
            from .dataflow.trie_tokenizer import MT_TRIE_TOKENIZER
            world_tokenizer = MT_TRIE_TOKENIZER(os.path.join(SCRIPT_DIR, "./dataflow/rwkv_vocab_v20230424.txt"))
            self.worldTokenizer = world_tokenizer
        else:
            raise NotImplementedError(f"Unsupported vocab size ({vocab_size}) - custom tokenizer not supported")

    # Encoding strings
    def encode(self, text: str):
        if self.worldTokenizer != None:
            return self.worldTokenizer.encode(text)
        return self.fastTokenizer.encode(text)

    # Decoding strings
    def decode(self, tokens: list):
        if self.worldTokenizer != None:
            return self.worldTokenizer.decode(tokens)
        return self.fastTokenizer.decode(tokens)

    # Forwarding logic, withoout torch._no_grad() context
    def _forward(
            self, tokens, 
            stateObj = None,
            all_logits = False
        ):

        logits_arr = None
        token_len = len(tokens)

        # The all_logits array, if requested
        all_logits_arr = None

        # For each token, process the state, in batches up to ctx_len
        for i in range(0, token_len, self.ctx_len):
            # Token set
            token_set = tokens[i:i+self.ctx_len]

            # Check if tokens are already tensors
            batch_tokens = torch.tensor(
                token_set, 
                dtype=torch.long, device=self.device
            ).unsqueeze(0)
            
            # Compute the logits and state
            logits_arr, stateObj = self.model.forward(
                batch_tokens, stateObj
            )

            # Build the all_logits array
            if all_logits:
                if all_logits_arr is None:
                    all_logits_arr = logits_arr[0]
                else:
                    all_logits_arr = torch.cat([all_logits_arr, logits_arr[0]], dim=0)

        # Return the logits and state
        if all_logits:
            return all_logits_arr, stateObj
        else:
            return logits_arr[0][-1], stateObj
    
    # Forwarding logic, with torch._no_grad() context
    def forward(
            self, tokens:list, 
            stateObj = None,
            all_logits = False
        ):
        with torch.no_grad():
            return self._forward(tokens, stateObj, all_logits)

    # Sampling logits
    def sample_logits(
            self, logits, 
            prv_tokens=[0], 
            temperature=1.0, top_p=0.9,
            token_ban: list = []
            ):
        # Copy to CPU first
        logits = logits.float().cpu()

        # Max negative float
        max_neg = -torch.finfo(torch.float).max

        # Apply token ban
        for x in token_ban:
            logits[x] = max_neg
        
        # Remove NaNs from logits
        for x in range(len(logits)):
            if torch.isnan(logits[x]):
                logits[x] = max_neg

        # Handle sampling with temperature
        if temperature > 0.0:
            probs = F.softmax(logits, dim=-1)
            sorted_probs = torch.sort(probs, descending=True)[0]
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).float().cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if temperature != 1.0:
                probs = probs.pow(1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return out
        else: 
            # Since the tokenizer sample does not support temp==0
            # we handle this case ourself, by fining the top token
            return torch.argmax(logits, dim=-1).item()

    # Completion API
    def completion(self, 
            prompt, 
            max_tokens: int = 32,
            temperature: float = 1.0,
            top_p: float = 0.9,
            token_ban: list = [],
            start_state = None,
            stream_to_stdout: bool = False,
        ):
        # Encode the context, if its a string
        if isinstance(prompt, str):
            enc = self.encode(prompt)
        # Check if the prompt is a list of tokens
        elif isinstance(prompt, list):
            enc = prompt
        else:
            raise ValueError("Prompt must be a string or a list of tokens")

        # Keep track of the logits and state
        logits = None
        stateObj = start_state

        # For each token, process the state
        logits, stateObj = self.forward(enc, stateObj)

        # # Garbage collect
        # gc.collect()
        # torch.cuda.empty_cache()

        # Generate each token
        out_tokens = []
        for i in range(max_tokens):
            ttt = self.sample_logits(
                logits, 
                # prv_tokens=full_tokens,
                temperature=temperature, top_p=top_p,
                token_ban=token_ban
            )
            
            # Append the token
            out_tokens.append(ttt)
            # full_tokens.append(ttt)
            if stream_to_stdout:
                print(self.decode([ttt]), end="", flush=True)

            # Perform the forward pass
            logits, stateObj = self.forward([ttt], stateObj)

        # Decode the tokens
        out_str = self.decode(out_tokens)

        # # Garbage collect
        # gc.collect()
        # torch.cuda.empty_cache()

        # Return the output string, and state
        return out_str, stateObj
