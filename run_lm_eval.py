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

from configs import parse_cmdline_configs, TrainerCLI_Config, Model_Config, Runtime_Config, Config

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

from src.pipeline import PIPELINE, PIPELINE_ARGS

from lm_eval import tasks, evaluator
from lm_eval.api.model import TemplateLM

########################################################################################################

#MODEL_NAME = "/fsx/BlinkDL/HF-MODEL/rwkv-5-world/RWKV-5-World-1.5B-v2-OnlyForTest_14%_trained-20231001-ctx4096"

from dataclasses import dataclass
import typing

@dataclass(kw_only=True)
class CLI_Config:
    path: str
    precision: int | str = 'bf16'
    seed: int | None = None
    recurrent: int = 1
    train:typing.Any = None
    model: Model_Config

config, errors = parse_cmdline_configs(sys.argv[1:], CLI_Config)
if errors != '':
    print(errors)
    exit()

os.environ["RWKV_MODEL_TYPE"] = config.model.tmix
os.environ["RWKV_CTXLEN"] = str(config.model.ctx_len)
os.environ["RWKV_HEAD_SIZE_A"] = str(config.model.head_size)

model_path = config.path

# Setup the model
from src.model import Transformer

print(f'Loading model - {model_path}')
state_dict = torch.load(model_path, mmap=True)
with torch.device('meta'):
    model = Transformer(config)
model.load_state_dict(state_dict, assign=True)

match config.precision:
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

#eval_tasks += ['lambada_openai','piqa','hellaswag','winogrande','arc_challenge','arc_easy','sciq']

# eval_tasks += ['hellaswag','winogrande']
# eval_tasks += ['lambada_openai','piqa','storycloze_2016','hellaswag','winogrande']
# eval_tasks += ['arc_challenge','arc_easy','headqa','openbookqa','sciq']
# eval_tasks += ['record','copa']
# eval_tasks += ['triviaqa']
# eval_tasks += ['coqa']

#RWKV_PAD = []
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

class EvalHarnessAdapter(TemplateLM):
    # bugfix for lm_eval 0.4.2
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self):
        super().__init__()
        self.tokenizer = TokenizerWrapper(pipeline.tokenizer)

    # def greedy_until(self, requests): # designed for coqa
    #     res = []
    #     for i in range(len(requests)):
    #         if i % 50 == 0:
    #             print(i)
    #         otoken = []
    #         while True:
    #             src = self.tokenizer.encode(requests[i][0]) + otoken

    #             src = src[-4096:]
    #             outputs, _ = model.forward(src, None)
                
    #             otoken += [int(torch.argmax(outputs))]
    #             ss = self.tokenizer.decode(otoken)
    #             if '\n' in ss or len(ss) > 200:
    #                 if not ss.endswith('\n'):
    #                     ss = ss + '\n'
    #                 print(ss)
    #                 res += [(ss)]
    #                 break
    #     print(res)
    #     return res


    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "`loglikelihood_rolling` is currently not supported"
        )
    
    def generate_until(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "`generate_until` is currently not supported"
        )

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # FIXME - is this correct? is it even used? didn't seem to be
        # FIXME - we really should support recurrent inference
        return config.model.ctx_len
        # try:
        #     return self.gpt2.config.n_ctx
        # except AttributeError:
        #     # gptneoconfig doesn't have n_ctx apparently
        #     return self.gpt2.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        # FIXME - is this correct? is it even used? didn't seem to be, since the model itself complained at 512 when returning 256 here
        # FIXME - we really should support recurrent inference
        return config.model.ctx_len

    @property
    def batch_size(self):
        return 1 
        # TODO: fix multi-gpu
        #return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)
            
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        global logitBuf, correctBuf

        res = []

        for COUNTER in range(len(requests)):
            n = COUNTER

            src = requests[n][1] + requests[n][2]

            src = RWKV_PAD + src
            inputs = torch.tensor(src, dtype=torch.long, device=device, requires_grad=False).unsqueeze(0)

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
                    outputs, _ = model.forward(inputs, None)
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

if config.seed is None:
    config.seed = 1234 

adapter = EvalHarnessAdapter()
with torch.no_grad():
    with torch.amp.autocast(device_type='cuda', dtype=dtype):
	    results = evaluator.simple_evaluate(
	        model=adapter,
	        tasks=eval_tasks,
	        #provide_description=False,
	        num_fewshot=0,
	        limit=None,
	        bootstrap_iters=10000,
	        numpy_random_seed = config.seed,
	        torch_random_seed = config.seed,
	        # fewshot_random_seed = config.seed, # FIXME - needed in next version of lm_eval
	    )

print(results['results'])
