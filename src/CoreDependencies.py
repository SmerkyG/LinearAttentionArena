import os
import torch
from torch import nn, Tensor

def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop
TCompile = __nop
TCompileDisable = __nop
TJIT = __nop

if os.getenv("RWKV_TORCH_COMPILE", '0').lower() in ['1', 'true']:
    TCompile = torch.compile
    TCompileDisable = torch._dynamo.disable
elif os.getenv("RWKV_JIT_ON", '1').lower() in ['1', 'true']:
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
    TJIT = torch.jit.script
