"""Evaluation utilities: FID, SSIM, LPIPS for PixelFlow validation."""

import os
import copy
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms

from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.pipeline_pixelflow import PixelFlowPipeline


def compute_ssim_batch(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """Compute SSIM between two batches of images (B, C, H, W) in [0, 1]."""
    channel = img1.shape[1]
    kernel = _gaussian_kernel(window_size, 1.5).to(img1.device, img1.dtype)
    kernel = kernel.expand(channel, 1, window_size, window_size)

    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, kernel, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, kernel, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=[1, 2, 3])  # per-image SSIM


def _gaussian_kernel(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


class ValidationEvaluator:
    """Generates samples and computes FID, SSIM, LPIPS against a validation set."""

    def __init__(self, config, val_root, device, num_samples=500, batch_size=8,
                 num_inference_steps=10, guidance_scale=4.0):
        self.config = config
        self.val_root = val_root
        self.device = device
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.resolution = int(getattr(config.data, "resolution", 256))
        self.num_stages = config.scheduler.num_stages
        self.normalize = bool(getattr(config.data, "normalize", True))

        # Load LPIPS model lazily
        self._lpips_model = None

        # Build validation image list with class labels
        self.val_dataset = ImageFolder(
            val_root,
            transform=transforms.Compose([
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
            ]),
        )

    @property
    def lpips_model(self):
        if self._lpips_model is None:
            import lpips
            self._lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self._lpips_model.eval()
        return self._lpips_model

    @torch.no_grad()
    def generate_samples(self, model, save_dir=None):
        """Generate samples using the pipeline, returns list of (generated_tensor, class_label)."""
        scheduler = PixelFlowScheduler(
            self.config.scheduler.num_train_timesteps,
            num_stages=self.num_stages,
            gamma=-1 / 3,
        )
        pipeline = PixelFlowPipeline(
            scheduler, model, text_encoder=None, tokenizer=None,
        )

        num_classes = len(self.val_dataset.classes)
        samples_per_class = self.num_samples // num_classes
        steps = [self.num_inference_steps] * self.num_stages

        generated = []  # list of (B, C, H, W) tensors in [0, 1]
        labels = []

        for cls_idx in range(num_classes):
            remaining = samples_per_class
            while remaining > 0:
                bs = min(self.batch_size, remaining)
                raw = pipeline(
                    prompt=[cls_idx] * bs,
                    height=self.resolution,
                    width=self.resolution,
                    num_inference_steps=steps,
                    guidance_scale=self.guidance_scale,
                    device=self.device,
                    shift=1.0,
                )
                # raw shape: (B, H, W, C) numpy
                # Convert to tensor (B, C, H, W) in [0, 1]
                imgs = torch.from_numpy(raw).permute(0, 3, 1, 2).float()
                if self.normalize:
                    imgs = imgs * 0.5 + 0.5  # denorm from [-1, 1] to [0, 1]
                imgs = imgs.clamp(0, 1)
                generated.append(imgs)
                labels.extend([cls_idx] * bs)
                remaining -= bs

        generated = torch.cat(generated, dim=0)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            for i in range(generated.shape[0]):
                img = (generated[i].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                Image.fromarray(img).save(os.path.join(save_dir, f"gen_{labels[i]}_{i:04d}.png"))

        return generated, labels

    @torch.no_grad()
    def compute_metrics(self, model, epoch, logger=None, save_dir=None):
        """Generate samples and compute FID, SSIM, LPIPS. Returns dict of metrics."""
        model.eval()

        gen_dir = os.path.join(save_dir, "generated") if save_dir else None
        generated, gen_labels = self.generate_samples(model, save_dir=gen_dir)

        # --- SSIM & LPIPS: compare against real images of matching class ---
        # Build a mapping: class_idx -> list of real image tensors
        real_by_class = {}
        for idx in range(len(self.val_dataset)):
            img, cls_idx = self.val_dataset[idx]
            if cls_idx not in real_by_class:
                real_by_class[cls_idx] = []
            real_by_class[cls_idx].append(img)

        ssim_scores = []
        lpips_scores = []
        for i in range(generated.shape[0]):
            cls_idx = gen_labels[i]
            if cls_idx not in real_by_class or len(real_by_class[cls_idx]) == 0:
                continue
            # Pick a random real image from the same class
            real_idx = i % len(real_by_class[cls_idx])
            real_img = real_by_class[cls_idx][real_idx].unsqueeze(0).to(self.device)
            gen_img = generated[i].unsqueeze(0).to(self.device)

            # Resize if needed
            if real_img.shape[-2:] != gen_img.shape[-2:]:
                real_img = F.interpolate(real_img, size=gen_img.shape[-2:], mode='bilinear', align_corners=False)

            # SSIM
            ssim_val = compute_ssim_batch(gen_img, real_img).item()
            ssim_scores.append(ssim_val)

            # LPIPS (expects [-1, 1])
            lpips_val = self.lpips_model(gen_img * 2 - 1, real_img * 2 - 1).item()
            lpips_scores.append(lpips_val)

        # --- FID: use clean-fid ---
        fid_score = None
        if gen_dir is not None:
            try:
                from cleanfid import fid
                fid_score = fid.compute_fid(gen_dir, self.val_root,
                                            mode="clean", num_workers=4)
            except Exception as e:
                if logger:
                    logger.info(f"FID computation failed: {e}")

        metrics = {
            "fid": fid_score,
            "ssim": float(np.mean(ssim_scores)) if ssim_scores else None,
            "lpips": float(np.mean(lpips_scores)) if lpips_scores else None,
        }

        if logger:
            fid_str = f"{metrics['fid']:.4f}" if metrics['fid'] is not None else "N/A"
            ssim_str = f"{metrics['ssim']:.4f}" if metrics['ssim'] is not None else "N/A"
            lpips_str = f"{metrics['lpips']:.4f}" if metrics['lpips'] is not None else "N/A"
            logger.info(f"[Validation Epoch {epoch}] FID: {fid_str}, SSIM: {ssim_str}, LPIPS: {lpips_str}")

        model.train()
        return metrics
