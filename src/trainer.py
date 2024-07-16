import os, math, time, datetime, subprocess
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only

from configs import TrainerCLI_Config

def my_save(config:TrainerCLI_Config, trainer:pl.Trainer, dd, ff):
    if 'deepspeed_stage_3' in config.train.strategy:
        trainer.save_checkpoint(ff, weights_only=True)
    else:
        torch.save(dd, ff)

class train_callback(pl.Callback):
    def __init__(self, config:TrainerCLI_Config):
        super().__init__()
        self.config = config

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        config = self.config
        # if config.cuda_cleanup > 0:
        #     torch.cuda.empty_cache()
        real_global_step = trainer.global_step + config.train.epoch_begin * config.runtime.epoch_global_steps

        # LR schedule
        w_steps = config.train.warmup_steps
        if config.train.lr_final == config.train.lr_init or config.runtime.epoch_count == 0:
            lr = config.train.lr_init
        else:
            decay_total = config.runtime.epoch_count * config.runtime.epoch_global_steps
            progress = (real_global_step - w_steps + 1) / (decay_total - w_steps)
            progress = min(1, max(0, progress))

            if config.train.lr_final == 0 or config.train.lr_init == 0:  # linear decay
                lr = config.train.lr_init + (config.train.lr_final - config.train.lr_init) * progress
            else:  # exp decay
                lr = config.train.lr_init * math.exp(math.log(config.train.lr_final / config.train.lr_init) * pow(progress, 1))
            # if trainer.is_global_zero:
            #     print(trainer.global_step, decay_step, decay_total, w_step, progress, lr)

        if config.train.my_exit_tokens != 0: # cosine decay
            real_tokens = real_global_step * config.model.ctx_len * config.runtime.global_step_bsz
            warmup_tokens = w_steps * config.model.ctx_len * config.runtime.global_step_bsz
            progress = (real_tokens - warmup_tokens) / (abs(config.train.my_exit_tokens) - warmup_tokens)
            progress = max(0, min(1, progress))
            lr_final_factor = config.train.lr_final / config.train.lr_init                
            lr_mult = (0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)
            if config.train.my_exit_tokens > 0:
                lr = config.train.lr_init * lr_mult
            else:
                lr = (lr + config.train.lr_init * lr_mult) / 2
            if progress >= 1:
                if (trainer.is_global_zero) or ('deepspeed_stage_3' in config.train.strategy):
                    my_save(
                        config, trainer,
                        pl_module.state_dict(),
                        f"{config.train.proj_dir}/rwkv-final.pth",
                    )
                    print("!!!TRAINING COMPLETE!!!")
                    exit(0)
        if trainer.global_step < w_steps:
            lr = lr * (0.2 + 0.8 * trainer.global_step / w_steps)

        if config.train.weight_decay_final > 0:
            wd_now = config.train.weight_decay * math.exp(math.log(config.train.weight_decay_final / config.train.weight_decay) * progress)
        else:
            wd_now = config.train.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            if config.train.layerwise_lr > 0:
                param_group["lr"] = lr * param_group["my_lr_scale"]
                # print(param_group["lr"], param_group["my_lr_scale"])
            else:
                param_group["lr"] = lr

        trainer.my_lr = lr
        trainer.my_wd = wd_now
        # rank_zero_info(f"{real_global_step} {lr}")

        if trainer.global_step == 0 and batch_idx == 0:
            if trainer.is_global_zero:  # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(config.train.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {config.runtime.my_timestamp}\n{vars(self.config)}\n")
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                if len(config.train.wandb) > 0:
                    print("Login to wandb...")
                    import wandb
                    wandb.init(
                        project=config.train.wandb,
                        name=config.runtime.run_name + " " + config.runtime.my_timestamp,
                        config=config,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        config = self.config
        tokens_per_micro_step = config.model.ctx_len * config.runtime.global_step_bsz / config.train.accumulate_grad_batches
        real_global_step = trainer.global_step + config.train.epoch_begin * config.runtime.epoch_global_steps
        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = tokens_per_micro_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            if pl.__version__[0]=='2':
                micro_step_loss = outputs["loss"] * trainer.accumulate_grad_batches
            else:
                micro_step_loss = trainer.my_loss_all.float().mean().item() * trainer.accumulate_grad_batches
            trainer.my_loss_sum += micro_step_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            # self.log("s", real_global_step, prog_bar=True, on_step=True)

            # if len(config.train.wandb) > 0:
            #     lll = {"loss": micro_step_loss, "lr": trainer.my_lr, "wd": trainer.my_wd, "Gtokens": real_global_step * tokens_per_micro_step / 1e9}
            #     if kt_s > 0:
            #         lll["kt/s"] = kt_s
            #     trainer.my_wandb.log(lll, step=int(real_global_step))
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in config.train.strategy): # save pth
            if config.train.magic_prime > 0:
                expand_factor = 1
                if int(real_global_step) == int(config.train.magic_prime * expand_factor // self.config.runtime.global_step_bsz) - 1:
                    to_save_dict = pl_module.state_dict()
                    my_save(
                        config, trainer,
                        to_save_dict,
                        f"{config.train.proj_dir}/rwkv-final.pth",
                    )
                

    def on_train_epoch_start(self, trainer, pl_module):
        config = self.config
        if pl.__version__[0]=='2':
            dataset = trainer.train_dataloader.dataset
        else:
            dataset = trainer.train_dataloader.dataset.datasets
        assert "MyDataset" in str(dataset)
        # print(f'########## world_size {trainer.world_size} global_rank {trainer.global_rank} real_epoch {trainer.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        config = self.config
        to_save_dict = {}
        real_current_epoch = trainer.current_epoch
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in config.train.strategy):  # save pth
            if (config.train.epoch_save > 0 and (real_current_epoch+1) % config.train.epoch_save == 0) or (real_current_epoch == config.runtime.epoch_count - 1):
                if config.train.data_type == 'wds_img':
                    raw_dict = pl_module.state_dict()
                    for k in raw_dict:
                        if k.startswith('encoder.') or k.startswith('decoder.'):
                            to_save_dict[k] = raw_dict[k]
                else:
                    to_save_dict = pl_module.state_dict()
                try:
                    my_save(
                        config, trainer,
                        to_save_dict,
                        f"{config.train.proj_dir}/rwkv-{trainer.current_epoch}.pth",
                    )
                except Exception as e:
                    print('Error\n\n', e, '\n\n')

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{real_current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {real_current_epoch - config.train.epoch_begin}\n")
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0

@rank_zero_only
def generate_init_weight(model, config:TrainerCLI_Config, init_weight_name):
    mm = model.generate_init_weight()

    if config.train.train_stage == 1:
        if len(config.train.load_model) > 0:
            print(f"Combine weights from {config.train.load_model}...")
            load_dict = torch.load(config.train.load_model, map_location="cpu")
            for k in load_dict:
                try:
                    assert k in mm
                except:
                    print('missing', k)
                    exit(0)
                src = load_dict[k]
                try:
                    mm[k] = src.reshape(mm[k].shape)
                except:
                    tmp = mm[k].squeeze().clone()
                    print(k, src.shape, '-->', mm[k].shape)
                    ss = src.shape[0]
                    dd = tmp.shape[0]
                    for i in range(dd):
                        pos = i / dd * ss
                        if pos >= ss - 1:
                            tmp[i] = src[ss-1]
                        else:
                            p0 = int(math.floor(pos))
                            ii = pos - p0
                            tmp[i] = src[p0] * (1-ii) + src[p0+1] * (ii)
                    mm[k] = tmp.reshape(mm[k].shape)
                    sss = src.squeeze().float().cpu().numpy()
                    print(sss[:10], '...', sss[-10:])
                    mmm = mm[k].squeeze().float().cpu().numpy()
                    print(mmm[:10], '...', mmm[-10:])

    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)

    if config.train.train_stage == 1:
        print("Done. Now go for stage 2.")
        exit(0)
