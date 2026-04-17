"""Evaluation utilities: FID (and optional SSIM/LPIPS) for PixelFlow validation.

For MRI (2-channel real/imag) mode:
  * validation set is read via MRIDataset (.pt files)
  * each slice is converted to a magnitude image with per-image min-max
    normalization (matching app2.py visualization), saved once as 3-channel
    grayscale PNGs, then compressed into a clean-fid "custom stats" cache
    (Inception feature mean/cov) so subsequent validation runs only pay the
    generation cost, not the real-set feature extraction cost.
  * generated samples go through the exact same magnitude->normalize->PNG
    pipeline so the two distributions are directly comparable.

For RGB (ImageFolder) mode we keep the original behaviour.
"""
import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms

from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.pipeline_pixelflow import PixelFlowPipeline


# --------------------------------------------------------------------------- #
# SSIM helpers (unchanged)                                                    #
# --------------------------------------------------------------------------- #
def compute_ssim_batch(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
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
    return ssim_map.mean(dim=[1, 2, 3])


def _gaussian_kernel(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


# --------------------------------------------------------------------------- #
# MRI helpers: 2-ch tensor -> 8-bit grayscale RGB PIL image                    #
# --------------------------------------------------------------------------- #
def _mri_tensor_to_mag_uint8(img: torch.Tensor) -> np.ndarray:
    """(2, H, W) float tensor -> (H, W) uint8 per-image min-max normalized."""
    assert img.ndim == 3 and img.shape[0] == 2, f"expected (2,H,W), got {tuple(img.shape)}"
    real, imag = img[0], img[1]
    mag = torch.sqrt(real ** 2 + imag ** 2)
    mn, mx = mag.min(), mag.max()
    if (mx - mn) > 1e-8:
        mag = (mag - mn) / (mx - mn)
    else:
        mag = torch.zeros_like(mag)
    arr = (mag.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    return arr


def _mri_np_to_mag_uint8(samples: np.ndarray) -> np.ndarray:
    """(B, H, W, 2) numpy -> (B, H, W) uint8."""
    mag = np.sqrt(samples[..., 0] ** 2 + samples[..., 1] ** 2)
    out = np.zeros_like(mag, dtype=np.uint8)
    for i in range(mag.shape[0]):
        m = mag[i]
        mn, mx = m.min(), m.max()
        if (mx - mn) > 1e-8:
            m = (m - mn) / (mx - mn)
        out[i] = (np.clip(m, 0, 1) * 255).round().astype(np.uint8)
    return out


def _save_gray_as_rgb(arr2d: np.ndarray, path: str) -> None:
    Image.fromarray(arr2d, mode="L").convert("RGB").save(path)


# --------------------------------------------------------------------------- #
# ValidationEvaluator                                                         #
# --------------------------------------------------------------------------- #
class ValidationEvaluator:
    """Generates samples and computes FID (+ optional SSIM/LPIPS)."""

    def __init__(self, config, val_root, device,
                 num_samples=500, batch_size=8,
                 num_inference_steps=10, guidance_scale=4.0,
                 stats_cache_dir=None, compute_ssim_lpips=False):
        self.config = config
        self.val_root = str(val_root)
        self.device = device
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.resolution = int(getattr(config.data, "resolution", 256))
        self.num_stages = config.scheduler.num_stages
        self.normalize = bool(getattr(config.data, "normalize", False))
        self.is_mri = str(getattr(config.data, "dataset", "")).lower() == "mri"
        self.compute_ssim_lpips = bool(compute_ssim_lpips)
        self._lpips_model = None

        if self.is_mri:
            from pixelflow.datasets.mri_dataset import MRIDataset
            self.val_dataset = MRIDataset(
                root=self.val_root,
                pt_key=str(getattr(config.data, "pt_key", "slices")),
                recursive=bool(getattr(config.data, "recursive", True)),
                target_mode="class_index",
                skip_broken_files=bool(getattr(config.data, "skip_broken_files", False)),
                verbose=False,
            )
        else:
            self.val_dataset = ImageFolder(
                self.val_root,
                transform=transforms.Compose([
                    transforms.Resize(self.resolution,
                                      interpolation=transforms.InterpolationMode.LANCZOS),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                ]),
            )

        cache_root = Path(stats_cache_dir) if stats_cache_dir else Path(self.val_root).parent
        self._stats_cache_dir = cache_root / "_fid_stats_cache"
        self._stats_cache_dir.mkdir(parents=True, exist_ok=True)
        tag = hashlib.md5(self.val_root.encode()).hexdigest()[:10]
        self._stats_name = f"pixelflow_val_{tag}_res{self.resolution}"
        self._real_png_dir = self._stats_cache_dir / f"real_pngs_{tag}_res{self.resolution}"

    # ---------- Real-side FID stats: build once, reuse forever -------------- #
    def prepare_val_stats(self, logger=None):
        """Dump validation set as PNGs and build clean-fid custom stats cache."""
        from cleanfid import fid as _fid
        if _fid.test_stats_exists(self._stats_name, mode="clean"):
            if logger:
                logger.info(f"[FID] Reusing cached val stats '{self._stats_name}'")
            return

        if self._real_png_dir.exists():
            shutil.rmtree(self._real_png_dir)
        self._real_png_dir.mkdir(parents=True, exist_ok=True)

        if logger:
            logger.info(f"[FID] Exporting {len(self.val_dataset)} val slices -> {self._real_png_dir}")

        for idx in range(len(self.val_dataset)):
            img, cls_idx = self.val_dataset[idx]
            if self.is_mri:
                gray = _mri_tensor_to_mag_uint8(img)
            else:
                # ImageFolder path: img is (3, H, W) in [0, 1]
                arr = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                gray = arr if arr.ndim == 2 else arr
            out = self._real_png_dir / f"real_{cls_idx}_{idx:06d}.png"
            if gray.ndim == 2:
                _save_gray_as_rgb(gray, str(out))
            else:
                Image.fromarray(gray).save(out)

        _fid.make_custom_stats(self._stats_name, str(self._real_png_dir), mode="clean")
        if logger:
            logger.info(f"[FID] Built custom stats '{self._stats_name}'")

    # ---------- Generation ------------------------------------------------- #
    @torch.no_grad()
    def generate_samples(self, model, save_dir):
        scheduler = PixelFlowScheduler(
            self.config.scheduler.num_train_timesteps,
            num_stages=self.num_stages,
            gamma=-1 / 3,
        )
        pipeline = PixelFlowPipeline(scheduler, model, text_encoder=None, tokenizer=None)

        num_classes = len(self.val_dataset.classes)
        samples_per_class = max(self.num_samples // num_classes, 1)
        steps = [self.num_inference_steps] * self.num_stages

        os.makedirs(save_dir, exist_ok=True)
        saved = 0
        labels_out = []
        for cls_idx in range(num_classes):
            remaining = samples_per_class
            while remaining > 0:
                bs = min(self.batch_size, remaining)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw = pipeline(
                        prompt=[cls_idx] * bs,
                        height=self.resolution,
                        width=self.resolution,
                        num_inference_steps=steps,
                        guidance_scale=self.guidance_scale,
                        device=self.device,
                        shift=1.0,
                    )
                # raw: (B, H, W, C) numpy
                if self.is_mri:
                    gray_batch = _mri_np_to_mag_uint8(raw)
                    for k in range(gray_batch.shape[0]):
                        _save_gray_as_rgb(gray_batch[k],
                                          os.path.join(save_dir, f"gen_{cls_idx}_{saved:06d}.png"))
                        labels_out.append(cls_idx)
                        saved += 1
                else:
                    arr = raw
                    if self.normalize:
                        arr = arr * 0.5 + 0.5
                    arr = np.clip(arr, 0, 1)
                    arr = (arr * 255).round().astype(np.uint8)
                    for k in range(arr.shape[0]):
                        Image.fromarray(arr[k]).save(
                            os.path.join(save_dir, f"gen_{cls_idx}_{saved:06d}.png"))
                        labels_out.append(cls_idx)
                        saved += 1
                remaining -= bs
        return saved, labels_out

    # ---------- Main entry called from train.py --------------------------- #
    @torch.no_grad()
    def compute_metrics(self, model, epoch, logger=None, save_dir=None):
        model.eval()

        # 1. Make sure real-side stats are ready (lazy; cheap on repeat).
        self.prepare_val_stats(logger=logger)

        # 2. Generate samples to save_dir/generated.
        assert save_dir is not None, "save_dir is required for FID computation"
        gen_dir = os.path.join(save_dir, "generated")
        os.makedirs(gen_dir, exist_ok=True)
        n_gen, _ = self.generate_samples(model, save_dir=gen_dir)

        # 3. FID against cached custom stats.
        fid_score = None
        try:
            from cleanfid import fid as _fid
            fid_score = _fid.compute_fid(
                gen_dir,
                dataset_name=self._stats_name,
                dataset_res=self.resolution,
                mode="clean",
                dataset_split="custom",
                num_workers=4,
            )
        except Exception as e:
            if logger:
                logger.info(f"[FID] computation failed: {e}")

        metrics = {"fid": fid_score, "num_generated": n_gen}

        # 4. Optional SSIM / LPIPS (disabled by default — expensive, paired).
        if self.compute_ssim_lpips and not self.is_mri:
            metrics.update(self._compute_ssim_lpips_rgb(gen_dir))

        if logger:
            fid_str = f"{fid_score:.4f}" if fid_score is not None else "N/A"
            logger.info(f"[Validation Epoch {epoch}] FID: {fid_str} (n={n_gen})")

        model.train()
        return metrics

    # ---------- Optional paired SSIM/LPIPS (RGB only) --------------------- #
    @property
    def lpips_model(self):
        if self._lpips_model is None:
            import lpips
            self._lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self._lpips_model.eval()
        return self._lpips_model

    def _compute_ssim_lpips_rgb(self, gen_dir):
        real_by_class = {}
        for idx in range(len(self.val_dataset)):
            img, cls = self.val_dataset[idx]
            real_by_class.setdefault(cls, []).append(img)

        ssim_scores, lpips_scores = [], []
        for p in sorted(Path(gen_dir).glob("gen_*.png")):
            parts = p.stem.split("_")
            cls_idx = int(parts[1])
            if cls_idx not in real_by_class:
                continue
            real = real_by_class[cls_idx][0].unsqueeze(0).to(self.device)
            gen = (torch.from_numpy(np.array(Image.open(p).convert("RGB")))
                   .permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0)
            if real.shape[-2:] != gen.shape[-2:]:
                real = F.interpolate(real, size=gen.shape[-2:], mode='bilinear', align_corners=False)
            ssim_scores.append(compute_ssim_batch(gen, real).item())
            lpips_scores.append(self.lpips_model(gen * 2 - 1, real * 2 - 1).item())
        return {
            "ssim": float(np.mean(ssim_scores)) if ssim_scores else None,
            "lpips": float(np.mean(lpips_scores)) if lpips_scores else None,
        }
