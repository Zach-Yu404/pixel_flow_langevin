"""Standalone FID evaluation for a single checkpoint."""
import argparse
import os

import torch
from omegaconf import OmegaConf

from pixelflow.evaluate_utils import ValidationEvaluator
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val_root", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--ema", default="ema_short", choices=["ema_short", "ema_medium", "ema", "model"],
                   help="Which weights to load: ema_short, ema_medium, ema (legacy), or model")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    seed_everything(42)

    config = OmegaConf.load(args.config)
    model = config_utils.instantiate_from_config(config.model).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        # Priority: requested key -> ema_short -> ema -> model
        for key in [args.ema, "ema_short", "ema", "model"]:
            if key in ckpt:
                state = ckpt[key]
                print(f"Loading weights from key '{key}'")
                break
        else:
            state = ckpt
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[{args.ckpt}] loaded on GPU {args.gpu}")

    val_root = args.val_root or str(config.train.val_root)
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.ckpt),
                                           f"fid_{os.path.basename(args.ckpt).replace('.pt','')}_cfg{args.guidance_scale}")

    evaluator = ValidationEvaluator(
        config=config,
        val_root=val_root,
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )

    metrics = evaluator.compute_metrics(model, epoch=0, logger=None, save_dir=out_dir)
    ckpt_name = os.path.basename(args.ckpt)
    fid = metrics.get("fid")
    fid_str = f"{fid:.4f}" if fid is not None else "N/A"
    print(f"=== RESULT: {ckpt_name} | CFG={args.guidance_scale} | FID={fid_str} ===")


if __name__ == "__main__":
    main()
