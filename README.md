# LinearAttentionArena fork for experiments

### PLEASE NOTE THAT THIS CODEBASE IS UNDERGOING ACTIVE DEVELOPMENT AND IS ENTIRELY UNSUPPORTED
### EVERYTHING MAY CHANGE AT ANY TIME AND BROKEN VERSIONS MAY BE COMMITTED
### !!!USE AT YOUR OWN RISK!!!

## setup

```
pip install lightning==2.3.0 torch deepspeed==0.14.3 wandb ninja --upgrade
```

you can download the minipile binidx via 

``` 
mkdir -p data
wget --continue -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
wget --continue -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
```

support for validation sets has been added
to download the minipile validation set you'll need to run the supplied `convert.py` and then obtain binidx conversion tool at `https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v5/make_data.py` and run that

## configuration

new config system allows you to specify one or more `-c CONFIG_PATH` in yaml or json format
later configs will override earlier ones
you can also list specific config parameters e.g. `--model.n_layer 12 --train.lr_init: 6e-4`

see configs.py for specific configuration settings in dataclasses

model.model_type is the variety of model
this supports two-in-one models separated by underscore, like `--model.model_type x060b2_goco` (goldfinch)

## running it

to create the starting initial state for a model run prepare.py with --train.train_stage 1:
`python train.py -c configs/L12D768minipile.yaml -c configs/goldfinch.yaml --train.proj_dir out/L12-D768-x060b2_goco-0 --train.train_stage 1"`

then to train the model:
`python train.py -c configs/L12D768minipile.yaml -c configs/goldfinch.yaml --train.proj_dir out/L12-D768-x060b2_goco-0"`

beware, it will continue from any numbered saved checkpoints still in the directory (if running again in the same dir)

there is also some lm_eval support in run_lm_eval.py, which also uses the same config system

and dragon_test3.py which can be used to run a quick inference test, also with the same system
