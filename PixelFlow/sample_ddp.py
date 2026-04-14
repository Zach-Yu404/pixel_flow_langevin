import argparse
import datetime
import numpy as np
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
import math

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist

from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.pipeline_pixelflow import PixelFlowPipeline
from pixelflow.utils import config as config_utils


def get_args_parser():
    parser = argparse.ArgumentParser(description='sample 50k images for FID evaluation', add_help=False)
    parser.add_argument('--pretrained', type=str, required=True, help='Pretrained model path')

    parser.add_argument("--sample-dir", type=str, default="evaluate_256pix_folder")
    parser.add_argument("--cfg", type=float, default=2.4)
    parser.add_argument("--num-steps-per-stage", type=int, default=30)
    parser.add_argument("--use-ode-dopri5", action="store_true")
    parser.add_argument("--local-batch-size", type=int, default=16)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--global-seed", type=int, default=10)
    return parser


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(hours=1))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    print(args)

    # create and load model
    config = OmegaConf.load(f"{args.pretrained}/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(device)
    ckpt = torch.load(f"{args.pretrained}/model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    resolution = config.data.resolution

    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps, num_stages=config.scheduler.num_stages, gamma=-1/3
    )
    num_steps_per_stage = [args.num_steps_per_stage for _ in range(config.scheduler.num_stages)]
    pipeline = PixelFlowPipeline(scheduler, model)

    # Create folder to save samples:
    sample_folder_dir = args.sample_dir
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    local_batch_size = args.local_batch_size
    global_batch_size = local_batch_size * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"

    # Number of images per class is equal (and pad 0)
    class_list_global = torch.zeros((total_samples,), device=device)
    num_classes = args.num_classes
    class_list = torch.arange(0, num_classes).repeat(args.num_fid_samples // num_classes)
    class_list_global[:class_list.shape[0]] = class_list

    local_samples = int(total_samples // dist.get_world_size())
    assert local_samples % local_batch_size == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(local_samples // local_batch_size)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        cur_index = torch.arange(local_batch_size) * dist.get_world_size() + rank + total
        cur_class_list = class_list_global[cur_index]

        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            samples = pipeline(
                prompt=cur_class_list,
                num_inference_steps=list(num_steps_per_stage),
                height=resolution,
                width=resolution,
                guidance_scale=args.cfg,
                device=device,
                use_ode_dopri5=args.use_ode_dopri5,
            )
        samples = (samples * 255).round().astype("uint8")
        image_list = [Image.fromarray(sample) for sample in samples]

        for img_ind, image in enumerate(image_list):
            index = img_ind * dist.get_world_size() + rank + total
            if index < args.num_fid_samples:
                image.save(f"{sample_folder_dir}/{index:06d}.png")

        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
