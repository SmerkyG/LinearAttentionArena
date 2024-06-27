########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
#
# pip install rwkv lm_eval --upgrade
#
import os, sys, types, json, math, time
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

import transformers # just for a bugfix for 0.4.2 of lm_eval

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

#from utils import PIPELINE, PIPELINE_ARGS
from rwkv.utils import PIPELINE, PIPELINE_ARGS

from lm_eval import tasks, evaluator
#from lm_eval.models.huggingface import HFLM
from lm_eval.models.gpt2 import GPT2LM
#from lm_eval.api.model import LM, TemplateLM

########################################################################################################

#MODEL_NAME = "/fsx/BlinkDL/HF-MODEL/rwkv-5-world/RWKV-5-World-1.5B-v2-OnlyForTest_14%_trained-20231001-ctx4096"

from dataclasses import dataclass
import typing

@dataclass(kw_only=True)
class CLI_Config:
    path: str
    precision: int | str = '32'
    seed: int | None = None
    recurrent: int = 1
    train:typing.Any = None

class Args(dict):
    def __init__(self, **kwargs):
        for name, value in kwargs:
            self[name] = value
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(e)
    def __setattr__(self, name, value):
         self[name] = value

args = Args()
args.head_size_a=64
args.head_size_divisor=8
args.ctx_len=2048
args.n_layer=24
args.n_embd=2048
args.dim_att=args.n_embd
args.dim_ffn=int(args.n_embd * 3.5) // 32 * 32
args.vocab_size=65536
args.posemb = 'none'
args.dropout=0.0
args.layerwise_lr=True
args.train_stage=2
args.grad_cp=True
args.betas=[0.9,0.99]
args.adam_eps=1e-8
#args..epoch_begin
#args.epoch_steps
#args.real_bsz
args.wandb=''
args.model_type = 'x060'

args.load_partial=0
args.proj_dir=''

args.precision = 32

args.model_path = 'out/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'

args.seed = None


os.environ["RWKV_MODEL_TYPE"] = args.model_type
os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
# Setup the model
from src.model import RWKV

print(f'Loading model - {args.model_path}')
state_dict = torch.load(args.model_path, mmap=True)
with torch.device('meta'):
    model = RWKV(args)
model.load_state_dict(state_dict, assign=True)

match args.precision:
    case 32:
        dtype = torch.float32
    case '32':
        dtype = torch.float32
    case 16:
        dtype = torch.float16
    case '16':
        dtype = torch.float16
    case 'bf16':
        dtype = torch.bfloat16
    case _:
        print("Bad precision type specified")
        exit()

device = 'cuda'
model = model.to(device=device, dtype=dtype)
model.eval()

pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

eval_tasks = []
eval_tasks += ['lambada_openai']
# eval_tasks += ['hellaswag','winogrande']
# eval_tasks += ['lambada_openai','piqa','storycloze_2016','hellaswag','winogrande']
# eval_tasks += ['arc_challenge','arc_easy','headqa','openbookqa','sciq']
# eval_tasks += ['record','copa']
# eval_tasks += ['triviaqa']
# eval_tasks += ['coqa']

RWKV_PAD = pipeline.tokenizer.encode('\n') # we will use '\n' as PAD
# RWKV_PAD = [0] # you can try using [0] as pad
print('RWKV_PAD', RWKV_PAD)

########################################################################################################

logitBuf = {}
correctBuf = {}

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = 0

    def encode(self, string: str, add_special_tokens=False):
        return self.tokenizer.encode(string)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

class EvalHarnessAdapter(GPT2LM):
    # bugfix for lm_eval 0.4.2
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self):
        #super().__init__()
        self.tokenizer = TokenizerWrapper(pipeline.tokenizer)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        global logitBuf, correctBuf

        res = []

        for COUNTER in range(len(requests)):
            n = COUNTER
            raw_src = requests[n][0][0] + requests[n][0][1]

            src = requests[n][1] + requests[n][2]

            raw_src = '\n' + raw_src
            src = RWKV_PAD + src
            inputs = torch.tensor(src, dtype=torch.long, device=model.device, requires_grad=False).unsqueeze(0)

            sss = str(src)

            correct = True
            if sss in logitBuf:
                logit = logitBuf[sss]
                correct = correctBuf[sss]
            else:
                q_len = len(requests[n][1])
                q_len += len(RWKV_PAD)
                logit = 0
                
                with torch.no_grad():
                    outputs = model.forward(inputs)
                    for i in range(q_len-1, len(src)-1):
                        oo = outputs[0,i].detach().float()
                        dst = src[i+1]
                        logit += math.log(F.softmax(oo, dim=-1)[dst])
                        _, s_index = torch.sort(oo, descending=True)
                        pred = s_index[0].item()
                        if pred != dst:
                            correct = False
                    outputs = None
                    pred = None
                logitBuf[sss] = logit
                correctBuf[sss] = correct
            
            res += [(logit, correct)]
            if n % 1000 == 0:
                print(f'{n//1000}/{len(requests)//1000}', end = ' ', flush=True)
        return res

adapter = EvalHarnessAdapter()
with torch.no_grad():
    with torch.amp.autocast(device_type='cuda', dtype=dtype):
        results = evaluator.evaluate(
            lm=adapter,
            task_dict=tasks.get_task_dict(eval_tasks),
            provide_description=False,
            num_fewshot=0,
            limit=None,
            bootstrap_iters=10000,
        )    

print(results['results'])
