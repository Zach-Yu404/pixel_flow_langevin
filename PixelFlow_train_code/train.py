from setproctitle import setproctitle
setproctitle("MSPflow::IP_prior")
import os

import argparse
import copy
from collections import OrderedDict
from datetime import datetime, timedelta
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import math

from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils.logger import setup_logger
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything
from pixelflow.data_in1k import build_imagenet_loader
from pixelflow.evaluate_utils import ValidationEvaluator


def get_args_parser():
    parser = argparse.ArgumentParser(description='Action Tokenizer', add_help=False)
    parser.add_argument('config', type=str, help='config')
    parser.add_argument('--output-dir', default='./exp0001', help='Output directory')
    parser.add_argument('--logging-steps', type=int, default=10, help='Logging steps')
    parser.add_argument('--checkpoint-steps', type=int, default=100000, help='Checkpoint steps')
    parser.add_argument('--pretrained-model', type=str, default=None, help='Pretrained model')
    parser.add_argument('--report-to', type=str, default=None, help='Report to, eg. wandb')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')

    return parser


def get_training_lr(step, warmup_steps, base_lr, total_steps):
    """Warmup + cosine decay schedule."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def main(args):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(hours=2))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    exp_name = "{}".format(datetime.now().strftime('%Y%m%d-%H%M%S'))

    # When resuming, find the existing log file to append to
    resume_log_file = None
    if args.resume is not None and rank == 0:
        existing_logs = sorted(
            [f for f in os.listdir(args.output_dir) if f.endswith('.log')],
            key=lambda f: os.path.getmtime(os.path.join(args.output_dir, f)),
        )
        if existing_logs:
            resume_log_file = os.path.join(args.output_dir, existing_logs[-1])

    logger = setup_logger(args.output_dir, exp_name, rank, __name__, log_file=resume_log_file)

    config = OmegaConf.load(args.config)

    rank_seed = config.seed * world_size + rank
    seed_everything(rank_seed, deterministic_ops=False, allow_tf32=False)
    logger.info(f"Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size} Seed: {rank_seed}")

    # save args and config to output_dir (rank 0 only)
    if rank == 0:
        with open(os.path.join(args.output_dir, "args.txt"), "w") as f:
            f.write(str(args))
        with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(config))

    logger.info(f"Config: {config}")
    model = config_utils.instantiate_from_config(config.model).to(device)

    # Dual EMA: short (half-life ~1 epoch) and medium (half-life ~5 epochs)
    ema_short_decay = float(getattr(config.train, "ema_short_decay", 0.99994455))
    ema_medium_decay = float(getattr(config.train, "ema_medium_decay", 0.99998891))
    ema_short = copy.deepcopy(model).to(device)
    ema_medium = copy.deepcopy(model).to(device)
    for param in ema_short.parameters():
        param.requires_grad = False
    for param in ema_medium.parameters():
        param.requires_grad = False
    logger.info(f"Dual EMA: short decay={ema_short_decay}, medium decay={ema_medium_decay}")

    logger.info(f"Num of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    noise_scheduler = PixelFlowScheduler(
        num_train_timesteps=config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    if args.pretrained_model is not None:
        ckpt = torch.load(args.pretrained_model, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    data_loader, sampler = build_imagenet_loader(config, noise_scheduler_copy)

    total_steps = config.train.epochs * len(data_loader)
    global_step = 0
    first_epoch = 0

    # Resume from checkpoint
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.module.load_state_dict(ckpt["model"], strict=True)
        # Load dual EMA (fall back to single "ema" key for old checkpoints)
        if "ema_short" in ckpt:
            ema_short.load_state_dict(ckpt["ema_short"], strict=True)
            ema_medium.load_state_dict(ckpt["ema_medium"], strict=True)
        elif "ema" in ckpt:
            ema_short.load_state_dict(ckpt["ema"], strict=True)
            ema_medium.load_state_dict(ckpt["ema"], strict=True)
            logger.info("Loaded old single EMA into both ema_short and ema_medium")
        optimizer.load_state_dict(ckpt["opt"])
        global_step = ckpt["global_step"]
        first_epoch = ckpt["epoch"]
        logger.info(f"***** Resumed from {args.resume} at step {global_step}, epoch {first_epoch} *****")
    else:
        # Prepare models for training:
        update_ema(ema_short, model.module, decay=0)
        update_ema(ema_medium, model.module, decay=0)

    warmup_steps = int(getattr(config.train, "warmup_steps", 0))

    # Validation setup
    do_validate = bool(getattr(config.train, "validate", False))
    val_every_n_epochs = int(getattr(config.train, "val_every_n_epochs", 10))
    checkpoint_every_n_epochs = int(getattr(config.train, "checkpoint_every_n_epochs", 0))
    evaluator = None
    if do_validate and rank == 0:
        val_root = str(config.train.val_root)
        evaluator = ValidationEvaluator(
            config=config,
            val_root=val_root,
            device=device,
            num_samples=int(getattr(config.train, "val_num_samples", 500)),
            batch_size=int(getattr(config.train, "val_batch_size", 8)),
            num_inference_steps=int(getattr(config.train, "val_num_inference_steps", 10)),
            guidance_scale=float(getattr(config.train, "val_guidance_scale", 4.0)),
        )
        logger.info(f"Validation enabled: every {val_every_n_epochs} epochs, {evaluator.num_samples} samples")

    logger.info(f"***** Running training ***** Total steps: {total_steps}")
    model.train()
    ema_short.eval()
    ema_medium.eval()

    # Calculate how many steps to skip in the resumed epoch
    steps_per_epoch = len(data_loader)
    skip_steps = global_step - first_epoch * steps_per_epoch if args.resume else 0

    for epoch in range(first_epoch, config.train.epochs):
        sampler.set_epoch(epoch)
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        for step, batch in enumerate(data_loader):
            if skip_steps > 0:
                skip_steps -= 1
                continue
            lr = get_training_lr(global_step, warmup_steps, config.train.lr, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.zero_grad()
            target = batch["target_values"].to(device)
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                model_output = model(
                    hidden_states=batch["pixel_values"].to(device),
                    encoder_hidden_states=None,
                    class_labels=torch.tensor(batch["input_ids"], device=device),
                    timestep=batch["timesteps"].to(device),
                    latent_size=batch["batch_latent_size"].to(device),
                    pos_embed=batch["pos_embed"].to(device),
                    cu_seqlens_q=batch["cumsum_q_len"].to(device),
                    cu_seqlens_k=None,
                    seqlen_list_q=batch["seqlen_list_q"],
                    seqlen_list_k=None,
                )

            loss = (model_output.float() - target.float()) ** 2
            loss_split = torch.split(loss, batch["seqlen_list_q"], dim=0)
            loss_items = torch.stack([x.mean() for x in loss_split])
            if "padding_size" in batch and batch["padding_size"] is not None and batch["padding_size"] > 0:
                loss_items = loss_items[:-1]

            loss = loss_items.mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema_short, model.module, decay=ema_short_decay)
            update_ema(ema_medium, model.module, decay=ema_medium_decay)

            epoch_loss_sum += loss.item()
            epoch_loss_count += 1

            if global_step % args.logging_steps == 0:
                logger.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Grad Norm: {grad_norm.item()}, LR: {lr:.6f}")

            if global_step % args.checkpoint_steps == 0 and global_step > 0:
                if rank == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "ema": ema_short.state_dict(),
                            "ema_short": ema_short.state_dict(),
                            "ema_medium": ema_medium.state_dict(),
                            "opt": optimizer.state_dict(),
                            "args": args,
                            "global_step": global_step,
                            "epoch": epoch,
                        },
                        os.path.join(args.output_dir, f"model_{global_step}.pt"))
                    logger.info(f"Model saved at step {global_step}")
                dist.barrier()

            global_step += 1

        # Log epoch average loss
        if epoch_loss_count > 0:
            avg_loss = epoch_loss_sum / epoch_loss_count
            logger.info(f"[Epoch {epoch}] Average Loss: {avg_loss:.6f} ({epoch_loss_count} steps)")

        # End-of-epoch: checkpoint and validation
        ckpt_dict = {
            "model": model.module.state_dict(),
            "ema": ema_short.state_dict(),
            "ema_short": ema_short.state_dict(),
            "ema_medium": ema_medium.state_dict(),
            "opt": optimizer.state_dict(),
            "args": args,
            "global_step": global_step,
            "epoch": epoch + 1,
        }

        if checkpoint_every_n_epochs > 0 and (epoch + 1) % checkpoint_every_n_epochs == 0:
            if rank == 0:
                torch.save(ckpt_dict, os.path.join(args.output_dir, f"model_epoch{epoch + 1}.pt"))
                logger.info(f"Epoch checkpoint saved at epoch {epoch + 1}")
            dist.barrier()

        if do_validate and (epoch + 1) % val_every_n_epochs == 0:
            if rank == 0 and evaluator is not None:
                val_save_dir = os.path.join(args.output_dir, f"val_epoch{epoch + 1}")
                evaluator.compute_metrics(
                    ema_short, epoch + 1, logger=logger, save_dir=val_save_dir,
                )
            dist.barrier(device_ids=[local_rank])
            model.train()

    # save final checkpoint
    if rank == 0:
        torch.save(
            {
                "model": model.module.state_dict(),
                "ema": ema_short.state_dict(),
                "ema_short": ema_short.state_dict(),
                "ema_medium": ema_medium.state_dict(),
                "opt": optimizer.state_dict(),
                "args": args,
                "global_step": global_step,
                "epoch": epoch + 1,
            },
            os.path.join(args.output_dir, f"model_final.pt"))
        logger.info(f"Final model saved at step {global_step}")
    dist.barrier()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
