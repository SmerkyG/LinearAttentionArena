########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy

from .tmix_x052 import RWKV_Tmix_x052
from .tmix_x060b import RWKV_Tmix_x060b
from .tmix_x060c import RWKV_Tmix_x060c
from .tmix_x060f import RWKV_Tmix_x060f
from .tmix_x060g import RWKV_Tmix_x060g

from .cmix_x052 import RWKV_CMix_x052
from .cmix_x060 import RWKV_CMix_x060

if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from .CoreDependencies import *

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

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################

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

        if 'x060b' in os.environ["RWKV_MODEL_TYPE"]:
            self.att = RWKV_Tmix_x060b(args, layer_id)
            self.ffn = RWKV_CMix_x060(args, layer_id)
            if 'x060bp' in os.environ["RWKV_MODEL_TYPE"]:
                self.parallel = True
        elif 'x060c' in os.environ["RWKV_MODEL_TYPE"]:
            self.att = RWKV_Tmix_x060c(args, layer_id)
            self.ffn = RWKV_CMix_x060(args, layer_id)
            if 'x060cp' in os.environ["RWKV_MODEL_TYPE"]:
                self.parallel = True
        elif 'x060f' in os.environ["RWKV_MODEL_TYPE"]:
            self.att = RWKV_Tmix_x060f(args, layer_id)
            self.ln2 = None
            self.ffn = None
        elif 'x060g' in os.environ["RWKV_MODEL_TYPE"]:
            self.att = RWKV_Tmix_x060g(args, layer_id)
            self.ln2 = None
            self.ffn = None
        elif 'x052' in os.environ["RWKV_MODEL_TYPE"]:
            self.att = RWKV_Tmix_x052(args, layer_id)
            self.ffn = RWKV_CMix_x052(args, layer_id)
        elif 'mamba' in os.environ["RWKV_MODEL_TYPE"]:
            self.att = Mamba(d_model=args.n_embd, d_state=16, d_conv=4, expand=2)
            self.ffn = Mamba(d_model=args.n_embd, d_state=16, d_conv=4, expand=2)
        
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)
        else:
            self.drop0 = nn.Identity()
            self.drop1 = nn.Identity()


    @TCompile
    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)

        if not self.parallel:
            x = self.drop0(x + self.att(self.ln1(x)))
            if self.ln2 is not None and self.ffn is not None:
                x = self.drop0(x + self.ffn(self.ln2(x)))
        else:
            # parallel
            lnx = self.ln1(x)
            x = self.drop0(x + self.att(lnx) + self.ffn(lnx))

        return x


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
        self.args = args
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
        else:
            self.drop0 = nn.Identity()

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
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        x = self.drop0(x)
        for block in self.blocks:
            if args.grad_cp == 1:
                x = deepspeed.checkpointing.checkpoint(block, x)
            else:
                x = block(x)

        x = self.ln_out(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits = self(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

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
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {n}", end="")

            scale = 1.0
            if len(p.shape) > 2 or "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias'):
                if 'ln_x.weight' in n:
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
