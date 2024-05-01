#!/usr/bin/env python3
import sys
import os
import torch
from src.model import RWKV

# ----
# This script is used to preload the huggingface dataset
# that is configured in the config.yaml file
# ----

# Check for argument, else throw error
if len(sys.argv) < 2:
    print("No arguments supplied")
    print("Usage: python3 dragon_test.py <model-path> [device] [length]") # [tokenizer]")
    sys.exit(1)

# download models: https://huggingface.co/BlinkDL
MODEL_PATH=sys.argv[1]

# If model device is not specified, use 'cuda' as default
RAW_DEVICE = "cpu fp32"
DEVICE = "cuda"
DTYPE  = "bf16"

# Get the raw device settings (if set)
if len(sys.argv) >= 3:
    RAW_DEVICE = sys.argv[2]

# Check if we are running a reference run
IS_REF_RUN = False
if RAW_DEVICE == "ref":
    DEVICE = "cpu"
    DTYPE  = "fp32"
    IS_REF_RUN = True

# Get the output length
LENGTH=200
if len(sys.argv) >= 4:
    LENGTH=int(sys.argv[3])

# Backward support for older format, we extract only cuda/cpu if its contained in the string
if RAW_DEVICE.find('cuda') != -1:
    DEVICE = 'cuda'
    
# The DTYPE setting
if RAW_DEVICE.find('fp16') != -1:
    DTYPE = "16"
elif RAW_DEVICE.find('bf16') != -1:
    DTYPE = "bf16"
elif RAW_DEVICE.find('fp32') != -1:
    DTYPE = "32"

print("DTYPE str", DTYPE)

# Disable torch compile for dragon test
#os.environ["RWKV_TORCH_COMPILE"] = "0"

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
args.ctx_len=512
args.n_layer=12
args.n_embd=768
args.vocab_size=65536
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

args.load_partial=0
args.proj_dir=''

# Setup the model
from src.model import SimpleRWKV
model = RWKV(args)

model_path = MODEL_PATH

print(f"########## Loading {model_path}... ##########")
try:
    load_dict = torch.load(model_path, map_location="cpu")
    load_keys = list(load_dict.keys())
    for k in load_keys:
        if k.startswith("_forward_module."):
            load_dict[k.replace("_forward_module.", "")] = load_dict[k]
            del load_dict[k]
except:
    print(f"Bad checkpoint {model_path}")
    if args.train_stage >= 2:  # try again using another checkpoint
        max_p = args.my_pile_prev_p
        if max_p == -1:
            model_path = f"{args.proj_dir}/rwkv-init.pth"
        else:
            model_path = f"{args.proj_dir}/rwkv-{max_p}.pth"
        args.epoch_begin = max_p + 1
        print(f"Trying {model_path}")
        load_dict = torch.load(model_path, map_location="cpu")

if args.load_partial == 1:
    load_keys = load_dict.keys()
    for k in model.state_dict():
        if k not in load_keys:
            load_dict[k] = model.state_dict()[k]
model.load_state_dict(load_dict)

model = SimpleRWKV(model, args, device=DEVICE, dtype_str=DTYPE)

# Dummy forward, used to trigger any warning / optimizations / etc
model.completion("\nIn a shocking finding", max_tokens=1, temperature=1.0, top_p=0.7)

# And perform the dragon prompt
prompt = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
if IS_REF_RUN:
    print(f"--- DRAGON PROMPT (REF RUN) ---{prompt}", end='')
    model.completion(prompt, stream_to_stdout=True, max_tokens=LENGTH, temperature=0.0)
else:
    print(f"--- DRAGON PROMPT ---{prompt}", end='')
    model.completion(prompt, stream_to_stdout=True, max_tokens=LENGTH, temperature=1.0, top_p=0.7)

# Empty new line, to make the CLI formatting better
print("")
