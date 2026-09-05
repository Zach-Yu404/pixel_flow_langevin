"""Forward operators, masks and measurement construction for the five tasks -- numerically identical to the
reference chain (IP_package/demo_runner.py + pipeline.py + inpaintingStart.py + Algorithm2/measurement.py).

Conventions kept on purpose:
  * random-inpainting masks are seeded with Python's hash(short_name)  -> PYTHONHASHSEED=0 is a hard contract;
  * box inpainting uses a centred box (mask_len_range[0]) and adds N(0, sigma_n^2) noise to y (seed sha256 of
    task/image/measurement_seed); blur/SR seeds are the legacy constants (kernel_std*1e4+12345, etc.);
  * the motion-blur kernel is DAPS's random-walk PSF (kernel_size 61, intensity 0.5, seed 42), stored as a data
    file so the DAPS repository is not needed at run time (scripts/make_motion_kernel.py regenerates it).
"""
import hashlib
import os
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import make_exact_AT

INPAINT_TASKS = ("box_inpainting", "random_inpainting")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ── operators ───────────────────────────────────────────────────────────────
class InpaintingOperator:
    """y = mask * x with a fixed [1,1,H,W] mask (1 = observed). measure() adds N(0, sigma^2) noise."""

    def __init__(self, mask, sigma, resolution=256):
        self.base_mask = mask
        self.mask = mask
        self.sigma = float(sigma)
        self.resolution = int(resolution)

    def _mask_for(self, x):
        mask = self.base_mask
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")
        return mask.to(device=x.device, dtype=x.dtype)

    def get_mask(self, x=None):
        return self.base_mask if x is None else self._mask_for(x)

    def __call__(self, x):
        return self._mask_for(x) * x

    def measure(self, x):
        y0 = self(x)
        return y0 + self.sigma * torch.randn_like(y0)


class Blurkernel(nn.Module):
    """Depthwise 2-D convolution with reflection padding (the DPS/DAPS convention)."""

    def __init__(self, kernel_size, device=None):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size // 2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3),
        )
        self.device = device

    def forward(self, x):
        return self.seq(x)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for _, f in self.named_parameters():
            f.data.copy_(k)


def gaussian_kernel(kernel_size, std):
    n = np.zeros((kernel_size, kernel_size))
    n[kernel_size // 2, kernel_size // 2] = 1
    return torch.from_numpy(scipy.ndimage.gaussian_filter(n, sigma=std))


class GaussianBlurOperator:
    def __init__(self, kernel_size, kernel_std, device, sigma=0.05, resolution=256):
        self.resolution, self.sigma, self.device = int(resolution), float(sigma), device
        self.kernel = gaussian_kernel(int(kernel_size), float(kernel_std))
        self.conv = Blurkernel(kernel_size, device=device).to(device)
        self.conv.update_weights(self.kernel.type(torch.float32))
        self.conv.requires_grad_(False)

    def __call__(self, x):
        return self.conv(x)

    blur = __call__

    def get_mask(self, x=None):
        shape = (1 if x is None else x.shape[0], 1, self.resolution, self.resolution)
        return torch.ones(shape, device=self.device if x is None else x.device)


class MotionBlurOperator:
    def __init__(self, kernel, device, sigma=0.05, resolution=256):
        self.resolution, self.sigma, self.device = int(resolution), float(sigma), device
        K = torch.as_tensor(np.asarray(kernel), dtype=torch.float32, device=device)
        self.kernel = K
        self.conv = Blurkernel(K.shape[-1], device=device).to(device)
        self.conv.update_weights(K)
        self.conv.requires_grad_(False)
        self._K_flipped = torch.flip(K, dims=(-2, -1)).clone()
        self.conv_T = Blurkernel(K.shape[-1], device=device).to(device)
        self.conv_T.update_weights(self._K_flipped)
        self.conv_T.requires_grad_(False)

    def __call__(self, x):
        return self.conv(x)

    blur = __call__

    def blur_T(self, x):
        return self.conv_T(x)

    def get_mask(self, x=None):
        shape = (1 if x is None else x.shape[0], 1, self.resolution, self.resolution)
        return torch.ones(shape, device=self.device if x is None else x.device)


class SROperator:
    def __init__(self, resolution=256, scale_factor=4, device="cuda", sigma=0.05, antialias=False):
        self.resolution, self.scale_factor = int(resolution), int(scale_factor)
        self.target_size = self.resolution // self.scale_factor
        self.sigma, self.device, self.antialias = float(sigma), device, bool(antialias)

    def __call__(self, x):
        return F.interpolate(x, size=(self.target_size, self.target_size), mode="bicubic",
                             align_corners=False, antialias=self.antialias)

    def get_mask(self, x=None):
        shape = (1 if x is None else x.shape[0], 1, self.resolution, self.resolution)
        return torch.ones(shape, device=self.device if x is None else x.device)


def load_motion_kernel(kernel_size, intensity, kernel_seed):
    p = os.path.join(DATA_DIR, f"motion_kernel_k{int(kernel_size)}_i{float(intensity)}_seed{int(kernel_seed)}.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found; generate it with scripts/make_motion_kernel.py (needs the DAPS repo)")
    return np.load(p)


# ── A_k = A . U^(K-1-k) with exact adjoints, per stage resolution ─────────────
def _interpolate_adjoint(y, target_size):
    with torch.enable_grad():
        xp = torch.zeros(*y.shape[:2], *target_size, device=y.device, dtype=y.dtype, requires_grad=True)
        Ux = F.interpolate(xp, size=y.shape[-2:], mode="bilinear", align_corners=False)
        (grad,) = torch.autograd.grad(Ux, xp, grad_outputs=y.detach())
    return grad.detach()


def make_Ak_fns_inpaint(operator, y, stage_shape, device):
    mask = operator.get_mask(x=y).to(device).float()
    full_h, full_w = mask.shape[-2:]
    stage_h, stage_w = stage_shape[-2:]
    need_resize = (stage_h != full_h) or (stage_w != full_w)

    def A_k(x):
        x_up = F.interpolate(x, size=(full_h, full_w), mode="bilinear", align_corners=False) if need_resize else x
        return mask * x_up

    def AT_k(r):
        masked = mask * r
        return _interpolate_adjoint(masked, (stage_h, stage_w)) if need_resize else masked

    return A_k, AT_k


def _make_Ak_blur(operator, y, stage_shape, device):
    """A_k for the blur operators: upsample to full resolution, then blur. The adjoint is taken exactly by
    autograd through the whole chain (make_exact_AT), as the reference full_ip run does."""
    stage_h, stage_w = stage_shape[-2:]
    full_h = full_w = operator.resolution
    need_resize = (stage_h != full_h) or (stage_w != full_w)

    def A_k(x):
        x_full = F.interpolate(x, size=(full_h, full_w), mode="bilinear", align_corners=False) if need_resize else x
        return operator.blur(x_full)

    return A_k, make_exact_AT(A_k, tuple(stage_shape))


def make_Ak_fns_sr(operator, y, stage_shape, device):
    stage_h, stage_w = stage_shape[-2:]
    full_h = full_w = operator.resolution
    target_h = target_w = operator.target_size
    need_resize = (stage_h != full_h) or (stage_w != full_w)
    aa = bool(getattr(operator, "antialias", False))

    def A_k(x):
        x_full = F.interpolate(x, size=(full_h, full_w), mode="bilinear", align_corners=False) if need_resize else x
        return F.interpolate(x_full, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=aa)

    def AT_k(r):
        prev = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            with torch.enable_grad():
                xp = torch.zeros((r.shape[0], r.shape[1], full_h, full_w), device=r.device, dtype=r.dtype, requires_grad=True)
                ds = F.interpolate(xp, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=aa)
                (grad,) = torch.autograd.grad(ds, xp, grad_outputs=r.detach())
            if not need_resize:
                return grad.detach()
            with torch.enable_grad():
                xs = torch.zeros((r.shape[0], r.shape[1], stage_h, stage_w), device=r.device, dtype=r.dtype, requires_grad=True)
                up = F.interpolate(xs, size=(full_h, full_w), mode="bilinear", align_corners=False)
                (grad2,) = torch.autograd.grad(up, xs, grad_outputs=grad.detach())
            return grad2.detach()
        finally:
            torch.use_deterministic_algorithms(prev)

    return A_k, AT_k


# ── masks ──────────────────────────────────────────────────────────────────
def make_random_mask(short_name, prob, resolution, device):
    """`prob` = fraction MISSING; seed = hash(short_name) (stable only under PYTHONHASHSEED=0)."""
    seed = int(hash(short_name) & 0xFFFFFFFF)
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = (torch.rand(1, 1, resolution, resolution, generator=g) >= prob).float()
    return m.to(device)


def make_center_box_mask(resolution, box_len, device):
    m = torch.ones(1, 1, resolution, resolution, device=device)
    t = l = (resolution - box_len) // 2
    m[..., t:t + box_len, l:l + box_len] = 0.0
    return m


def noise_seed_for(task, short_name, measurement_seed):
    key = f"{task}/{short_name}/{int(measurement_seed)}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


# ── measurement construction ────────────────────────────────────────────────
def build_measurement(task, op_cfg, demo, sigma_n, resolution, device, measurement_seed):
    """Returns dict(op, mask [1,1,H,W] 1=observed, y, make_Ak_fns, hole [1,1,H,W] or None)."""
    gt = demo["gt"].unsqueeze(0).to(device)
    sigma_n = float(sigma_n)

    if task == "motion_blur":
        K = load_motion_kernel(op_cfg["kernel_size"], op_cfg["kernel_intensity"], op_cfg["kernel_seed"])
        op = MotionBlurOperator(K, device, sigma=sigma_n, resolution=resolution)
        seed = int(float(op_cfg["kernel_intensity"]) * 1e4) + 11111 + 12345
        torch.manual_seed(seed)
        y_clean = op(gt).detach()
        y = (y_clean + torch.randn_like(y_clean) * sigma_n).detach()
        return dict(op=op, mask=op.get_mask(gt), y=y, make_Ak_fns=_make_Ak_blur, hole=None)

    if task == "gaussian_blur":
        op = GaussianBlurOperator(int(op_cfg["kernel_size"]), float(op_cfg["kernel_std"]), device,
                                  sigma=sigma_n, resolution=resolution)
        seed = int(float(op_cfg["kernel_std"]) * 1e4) + 12345
        torch.manual_seed(seed)
        y_clean = op(gt).detach()
        y = (y_clean + torch.randn_like(y_clean) * sigma_n).detach()
        return dict(op=op, mask=op.get_mask(gt), y=y, make_Ak_fns=_make_Ak_blur, hole=None)

    if task == "superresolution":
        op = SROperator(resolution=resolution, scale_factor=int(op_cfg.get("scale_factor", 4)), device=device,
                        sigma=sigma_n, antialias=bool(op_cfg.get("antialias", False)))
        torch.manual_seed(4_200_000)
        y_clean = op(gt).detach()
        y = y_clean + sigma_n * torch.randn_like(y_clean)
        return dict(op=op, mask=op.get_mask(gt), y=y, make_Ak_fns=make_Ak_fns_sr, hole=None)

    if task in INPAINT_TASKS:
        if task == "random_inpainting":
            mask = make_random_mask(demo["short_name"], float(op_cfg.get("mask_prob", 0.7)), resolution, device)
        else:
            if not bool(op_cfg.get("center", True)):
                raise ValueError("box_inpainting: only the centred box is supported in the clean project")
            mask = make_center_box_mask(resolution, int(op_cfg.get("mask_len_range", [128, 129])[0]), device)
        op = InpaintingOperator(mask, sigma_n, resolution)
        mode = op_cfg.get("measurement_mode", "measure")
        rng_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(noise_seed_for(task, demo["short_name"], measurement_seed))
            y = (op.measure(gt) if mode == "measure" else op(gt)).detach()
        finally:
            torch.set_rng_state(rng_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        hole = (1.0 - mask).to(device).float()
        return dict(op=op, mask=mask, y=y, make_Ak_fns=make_Ak_fns_inpaint, hole=hole if float(hole.sum()) > 0 else None)

    raise ValueError(f"unknown task {task!r}")
