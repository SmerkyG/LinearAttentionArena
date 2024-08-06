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

from src.logger import print0 as print

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
from src.lit import LightningModelWrapper

print(f'Loading model - {model_path}')
if config.path.lower().endswith('.pth'):
    with torch.device('meta'):
        model = Transformer(config)
        wrapper = LightningModelWrapper(model, config)
        wrapper.train(False) # important to avoid inits which are slow, and for the ds moe hack
        wrapper.configure_model()
    state_dict = torch.load(model_path, mmap=True)
    model.load_state_dict(state_dict, assign=True)
else:
    from lightning import Trainer
    from lightning.pytorch.utilities.migration import pl_legacy_patch
    from lightning.pytorch.utilities.migration.utils import _pl_migrate_checkpoint

    trainer = Trainer(
        use_distributed_sampler=False, 
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        logger=False,
        max_epochs=-1,

        accelerator='gpu',#config.train.accelerator, 
        strategy='deepspeed_stage_2',#config.train.strategy, 
        devices=8,#config.train.devices, 
        num_nodes=1,#config.train.num_nodes, 
        precision='bf16-mixed',#config.train.precision,
    )
    #with torch.device('meta'):
    with trainer.init_module(empty_init=True):
        model = Transformer(config)
        wrapper = LightningModelWrapper(model, config)
        wrapper.train(False) # important to avoid inits which are slow, and for the ds moe hack
        #model.configure_model() # done later
    
    # from torch.optim.adamw import AdamW
    # optim_groups = [
    #     {'params': list(model.parameters())[0:1], 'moe':True},
    # ]
    # trainer.optimizers = [AdamW(optim_groups)]
    #trainer.predict(wrapper, ckpt_path=config.path)
    

    # simulate the entire Lightning Trainer setup (like what happens when you call fit() or predict())
    
    trainer.strategy._lightning_module = wrapper  

    from lightning.pytorch.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
    trainer.state.fn = TrainerFn.FITTING #  TrainerFn.PREDICTING #
    trainer.state.status = TrainerStatus.RUNNING
    trainer.training = True # Needed for deepspeed strategy to notice optimizers so it doesn't assert about lack of MoE groups
    #trainer.predicting = True

    def my_setup_deepspeed_and_load_ckpt(wrapper, trainer, config):
        # Attach the trainer to the LightningModule (deepspeed also senses this when deciding to setup for training or inference)
        wrapper.trainer = trainer

        # links data to the trainer
        from torch.utils.data import DataLoader, Dataset
        trainer._data_connector.attach_data(wrapper, predict_dataloaders=[DataLoader(Dataset())], datamodule=None)

        # attach model to the strategy
        trainer.strategy.connect(wrapper)
        # trainer._callback_connector._attach_model_callbacks()
        # trainer._callback_connector._attach_model_logging_functions()

        # hook
        #log.debug(f"{self.__class__.__name__}: preparing data")
        # trainer._data_connector.prepare_data()
        # import lightning.pytorch.trainer.call
        # lightning.pytorch.trainer.call._call_setup_hook(trainer)  # allow user to setup lightning_module in accelerator environment


        trainer.strategy.setup_environment()
        #self.__setup_profiler()

        wrapper.configure_model()

        # strategy will configure model and move it to the device
        trainer.strategy.setup(trainer)

        # so that deepspeed doesn't load the optimizer states!
        trainer.state.fn = TrainerFn.PREDICTING
        trainer.predicting = True

        with pl_legacy_patch():
            loaded_checkpoint = trainer.strategy.load_checkpoint(config.path)
        loaded_checkpoint = _pl_migrate_checkpoint(loaded_checkpoint, config.path)
        #wrapper = LightningModelWrapper.load_from_checkpoint(config.path) # doesnt' work with sharded models
    
    import lightning.pytorch.trainer.call
    lightning.pytorch.trainer.call._call_and_handle_interrupt(trainer, my_setup_deepspeed_and_load_ckpt, wrapper, trainer, config)


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

        with torch.no_grad():
            B = 12
            for nb in range(0, len(requests), B):
                ne = min(nb+B, len(requests))

                # stack and pad to longest
                batch = []
                batch_info = []
                maxlen = 0
                for i in range(nb, ne):
                    q = RWKV_PAD + requests[i][1]
                    src = q + requests[i][2]
                    input = torch.tensor(src, dtype=torch.long, device=device, requires_grad=False)
                    batch.append( input )
                    batch_info.append((len(q), len(src)))
                    maxlen = max(maxlen, len(src))

                maxlen = (maxlen + 7) // 8 * 8 # round pad size up to nearest 8 for better GPU usage
                for i in range(len(batch)):
                    batch[i] = F.pad(batch[i], (0, maxlen - batch[i].size(0)))
                batch = torch.stack(batch, dim=0)

                outputs, _ = model.forward(batch, None)

                batched_logits = F.log_softmax(outputs, dim=-1)
                # Check if per-token argmax is exactly equal to continuation
                batched_greedy_toks = batched_logits.argmax(dim=-1)

                for i, info in enumerate(batch_info):
                    q_len, src_len = info
                    a_len = src_len - q_len
                    logits, a_toks, greedy_toks = batched_logits[i, q_len-1 : src_len-1], batch[i, q_len : src_len], batched_greedy_toks[i, q_len-1 : src_len-1]
                    assert logits.size(0) == a_len
                    assert a_toks.size(0) == a_len
                    assert greedy_toks.size(0) == a_len
                    max_equal = (greedy_toks == a_toks).all()

                    # Obtain log-probs at the corresponding continuation ('answer') token indices
                    logprobs = torch.gather(logits, 1, a_toks.unsqueeze(-1)).squeeze(-1)
                    assert logprobs.size(0) == a_len
                
                    # Answer: (log prob, is-exact-match)
                    answer = (float(logprobs.sum()), bool(max_equal))

                    res.append(answer)

                FREQ = 10 * B
                if nb % FREQ == 0:
                    print(f'{nb//FREQ}/{len(requests)//FREQ}', end = ' ', flush=True)
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
