#!/bin/bash
#######################################################################################################################
#
# !!! Run demo-training-prepare.sh with the same MODEL_TYPE & N_LAYER & N_EMBD first !!!
# Or, rename your base model to rwkv-init.pth and put it in the output folder
#
# The trainer will load the last rwkv-*.pth in the folder, such that it can continue from a stopped run
# Therefore check the log (### Loading rwkv-xxx.pth... ###), and make sure you don't have extra rwkv-*.pth there
#
#######################################################################################################################

model_type=""
layer=0
emb=0
dim_att=0
dim_ffn=0
ctx_len=512 # !!! change magic_prime if you change ctx_len !!!
posemb='none'
suffix=""
lr_init="6e-4"
lr_final="6e-5"
n_gpu=1
m_bsz=16
grad_cp=1 # turn on grad_cp to save VRAM
save_period=10
accumulate_grad_batches=1
head_size=64

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_type) model_type="$2"; shift ;;
        --layer) layer="$2"; shift ;;
        --emb) emb="$2"; shift ;;
        --dim_att) dim_att="$2"; shift ;;
        --dim_ffn) dim_ffn="$2"; shift ;;
        --ctx_len) ctx_len="$2"; shift ;;
        --head_size) head_size="$2"; shift ;;
        --posemb) posemb="$2"; shift ;;
        --lr_init) lr_init="$2"; shift ;;
        --lr_final) lr_final="$2"; shift ;;
        --n_gpu) n_gpu="$2"; shift ;;
        --m_bsz) m_bsz="$2"; shift ;;
        --grad_cp) grad_cp="$2"; shift ;;
        --save_period) save_period="$2"; shift ;;
        --suffix) suffix="$2"; shift ;;
        --accumulate_grad_batches) accumulate_grad_batches="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "model_type: $model_type"
echo "layer: $layer"
echo "emb: $emb"
echo "posemb: $posemb"
echo "lr_init: $lr_init"
echo "lr_final: $lr_final"
echo "ctx_len: $ctx_len"
echo "n_gpu: $n_gpu"
echo "m_bsz: $m_bsz"
echo "accumulate_grad_batches: $accumulate_grad_batches"
echo "grad_cp: $grad_cp"
echo "save_period: $save_period"
echo "suffix: $suffix"

PROJ_DIR="out/L"$layer"-D"$emb"-"$model_type$suffix # set output folder
rm -f "$PROJ_DIR"/rwkv-{0..100}.pth # remove old checkpts in folder

#######################################################################################################################
#
# magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case) = 2926181 in this case
# use https://www.dcode.fr/prime-numbers-search
# 512 would be 2926181
# 1024 would be 1463027
# 2048 would be 731531
# 4096 would be 365759
# for 7583728687 tokens, 4096 would be 1851491
# for 100000000 tokens, 4096 would be 24405
# use https://www.dcode.fr/prime-numbers-search
#
N_NODE=1 # number of nodes
DS_BUCKET_MB=2 # set to 2 for consumer GPUs, set to 200 for A100 / H100 (affects speed & vram usage)
#
python train.py --load_model "0" --wandb "Linear_Attention_Arena" --proj_dir $PROJ_DIR --model_type $model_type \
 --ctx_len $ctx_len --train_stage 3 --epoch_count 999999 --epoch_begin 0 \
 --data_file "data/minipile" --validation_data_file "data/minipile_validation" --val_check_interval 100 --my_exit_tokens 1498226207 --magic_prime 2926181 \
 --num_nodes $N_NODE --micro_bsz $m_bsz --n_layer $layer --n_embd $emb --dim_att $dim_att --dim_ffn $dim_ffn --posemb $posemb \
 --lr_init $lr_init --lr_final $lr_final --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --data_type "binidx" --vocab_size 65536 \
 --weight_decay 0.001 --epoch_save $save_period --head_size_a $head_size \
 --accelerator gpu --devices $n_gpu --precision bf16 --strategy deepspeed_stage_2 --grad_cp $grad_cp --ds_bucket_mb $DS_BUCKET_MB \
 --accumulate_grad_batches $accumulate_grad_batches
