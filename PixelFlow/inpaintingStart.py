from abc import ABC, abstractmethod
# from .resizer import Resizer
import torch.nn.functional as F
# from .fastmri_utils import fft2c_new
# from .motionblur.motionblur import Kernel

import torch
import torch.nn as nn
import scipy
import numpy as np
import yaml
import warnings
from torch.autograd import grad

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            if __OPERATOR__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __OPERATOR__[name] = cls
        cls.name = name
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class Operator(ABC):
    """
    Abstract base class for operators in diffusion processes.

    Attributes:
        sigma (float): Standard deviation of measurement noise.
    """
    def __init__(self, sigma=0.05):
        """
        Initializes the operator with a noise standard deviation.

        Args:
            sigma (float, optional): Measurement noise level. Defaults to 0.05.
        """
        self.sigma = sigma

    @abstractmethod
    def __call__(self, x):
        """
        Abstract method: apply operator to input data.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Output after applying the operator.
        """
        pass
        
    # @abstractmethod
    # def adjoint(self, x):
    #     """
    #     Applies the adjoint (transpose) of the forward operator Aᵀ.
    
    #     For a linear measurement model:
    #         y = A x + ε ,   ε ~ N(0, σ² I)
    
    #     this method computes:
    #         Aᵀ x
    
    #     This is used when forming gradients of the data-consistency term:
    #         ∇ₓ ||A x − y||² = Aᵀ (A x − y)
    
    #     Args:
    #         x (torch.Tensor): Input tensor in measurement space.
    
    #     Returns:
    #         torch.Tensor: Result after applying the adjoint operator.
    #     """
    #     pass
    
    
    # @abstractmethod
    # def precondition(self, x, t):
    #     """
    #     Applies the preconditioning matrix used in diffusion-based
    #     posterior sampling.
    
    #     In DPS/DAPS, this corresponds to applying:
    #         (Aᵀ A / σ² + I / t²)⁻¹ x
    
    #     where:
    #         - A is the forward operator
    #         - σ² is the measurement noise variance
    #         - t is the diffusion noise level
    
    #     This improves conditioning and accelerates convergence,
    #     especially for ill-posed inverse problems.
    
    #     Args:
    #         x (torch.Tensor): Input tensor.
    #         t (float or torch.Tensor): Diffusion noise scale at the current step.
    
    #     Returns:
    #         torch.Tensor: Preconditioned tensor.
    #     """
    #     pass
    
    
    # @abstractmethod
    # def noise_modulation(self, t):
    #     """
    #     Computes the noise modulation factor for Langevin / posterior sampling.
    
    #     This corresponds to:
    #         sqrt((Aᵀ A / σ² + I / t²)⁻¹)
    
    #     and is used to correctly scale injected Gaussian noise so that
    #     samples follow the target posterior distribution.
    
    #     Args:
    #         t (float or torch.Tensor): Diffusion noise level.
    
    #     Returns:
    #         torch.Tensor: Noise modulation tensor.
    #     """
    #     pass


    def measure(self, x):
        """
        Measures input data by applying the operator and adding Gaussian noise.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Noisy measurement result.
        """
        y0 = self(x)
        return y0 + self.sigma * torch.randn_like(y0)

    def loss(self, x, y):
        """
        Computes squared-error loss between operator output and observed data.

        Args:
            x (torch.Tensor): Input data tensor.
            y (torch.Tensor): Observed measurement tensor.

        Returns:
            torch.Tensor: Loss values (one per sample).
        """
        return ((self(x) - y) ** 2).flatten(1).sum(-1)

    def gradient(self, x, y, return_loss=False):
        """
        Computes gradient of the loss with respect to input x.

        Args:
            x (torch.Tensor): Input tensor requiring gradient.
            y (torch.Tensor): Observed measurements.
            return_loss (bool, optional): If True, returns both gradient and loss. Defaults to False.

        Returns:
            torch.Tensor or tuple: Gradient tensor (and optionally the loss value).
        """
        x_tmp = x.clone().detach().requires_grad_(True)
        loss = self.loss(x_tmp, y).sum()
        x_grad = torch.autograd.grad(loss, x_tmp)[0]
        if return_loss:
            return x_grad, loss
        return x_grad

    def log_likelihood(self, x, y):
        return -self.loss(x, y) / 2 / self.sigma ** 2

    def likelihood(self, x, y):
        return torch.exp(self.log_likelihood(x, y))


# Linear Operator
def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t + h, l:l + w] = 0

    return mask, t, t + h, l, l + w


class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(32, 32)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                                            mask_shape=(mask_h, mask_w),
                                            image_size=self.image_size,
                                            margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask


# @register_operator(name='inpainting')
# class Inpainting(Operator):
#     def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None, resolution=256, device='cuda',
#                  sigma=0.05):
#         super().__init__(sigma)
#         self.mask_gen = mask_generator(mask_type, mask_len_range, mask_prob_range, resolution)
#         self.mask = None  # [B, 1, H, W]
#         self.sigma = sigma

#     def __call__(self, x):
#         if self.mask is None:
#             self.mask = self.mask_gen(x)
#             self.mask = self.mask[0:1, 0:1, :, :]
#         return self.mask * x

@register_operator(name='inpainting')
class Inpainting(Operator):
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None, resolution=256, device='cuda',
                 sigma=0.05):
        super().__init__(sigma)
        self.mask_gen = mask_generator(mask_type, mask_len_range, mask_prob_range, resolution)
        self.mask = None
        self.base_mask = None
        self.resolution = int(resolution)
        self.device = device
        self.sigma = sigma

    def _init_mask(self, x=None):
        if self.base_mask is not None:
            return

        if x is not None:
            mask_source = torch.zeros(
                1, 3, self.resolution, self.resolution, device=x.device, dtype=x.dtype
            )
        else:
            mask_source = torch.zeros(
                1, 3, self.resolution, self.resolution, device=self.device, dtype=torch.float32
            )
        mask = self.mask_gen(mask_source)
        self.base_mask = mask[0:1, 0:1, :, :]
        self.mask = self.base_mask

    def _resolve_target_size(self, x, downsample=None):
        if downsample is None:
            return int(x.shape[-2]), int(x.shape[-1])

        if isinstance(downsample, (int, float)):
            factor = int(downsample)
            if factor <= 0:
                raise ValueError(f"downsample factor must be positive, got {downsample}")
            size = max(1, self.resolution // factor)
            return size, size

        if isinstance(downsample, (tuple, list)) and len(downsample) == 2:
            h, w = int(downsample[0]), int(downsample[1])
            if h <= 0 or w <= 0:
                raise ValueError(f"downsample target size must be positive, got {downsample}")
            return h, w

        raise TypeError(
            f"downsample must be None, int/float factor, or (h, w) tuple/list, got {type(downsample)}"
        )

    @staticmethod
    def _interp(x, size, mode):
        if mode in ("linear", "bilinear", "bicubic", "trilinear"):
            return F.interpolate(x, size=size, mode=mode, align_corners=False)
        return F.interpolate(x, size=size, mode=mode)

    def _downsample_iterative(self, x, target_h, target_w, mode):
        cur_h, cur_w = int(x.shape[-2]), int(x.shape[-1])
        if (cur_h, cur_w) == (target_h, target_w):
            return x

        # Fallback for upsampling or mixed resize requests.
        if target_h > cur_h or target_w > cur_w:
            return self._interp(x, size=(target_h, target_w), mode=mode)

        out = x
        while out.shape[-2] > target_h or out.shape[-1] > target_w:
            h, w = int(out.shape[-2]), int(out.shape[-1])
            next_h = max(target_h, h // 2) if h > target_h else h
            next_w = max(target_w, w // 2) if w > target_w else w
            if (next_h, next_w) == (h, w):
                break
            out = self._interp(out, size=(next_h, next_w), mode=mode)

        if out.shape[-2:] != (target_h, target_w):
            out = self._interp(out, size=(target_h, target_w), mode=mode)
        return out

    def _get_mask_and_input(self, x, downsample=None):
        self._init_mask(x)
        target_h, target_w = self._resolve_target_size(x, downsample=downsample)

        if x.shape[-2:] != (target_h, target_w):
            x_use = self._downsample_iterative(x, target_h, target_w, mode="bilinear")
        else:
            x_use = x

        mask = self.base_mask
        if mask.shape[-2:] != (target_h, target_w):
            mask = self._downsample_iterative(mask, target_h, target_w, mode="nearest")
        mask = mask.to(device=x.device, dtype=x.dtype)
        self.mask = mask
        return x_use, mask

    def get_mask(self, x=None, downsample=None):
        if x is None:
            self._init_mask()
            self.mask = self.base_mask
            return self.base_mask
        _, mask = self._get_mask_and_input(x, downsample=downsample)
        return mask

    def __call__(self, x, downsample=None):
        x_use, mask = self._get_mask_and_input(x, downsample=downsample)
        return mask * x_use

    def measure(self, x, downsample=None):
        y0 = self(x, downsample=downsample)
        return y0 + self.sigma * torch.randn_like(y0)

    def adjoint(self, x):
        x_use, mask = self._get_mask_and_input(x)
        return mask * x_use
        
    def precondition(self, x, t):
        _, mask = self._get_mask_and_input(x)
        bmat = 1.0/(mask/self.sigma**2 + 1.0/t**2)
        bmat = bmat.to(x.device)
        return bmat*x
        
    def noise_modulation(self, t):
        self._init_mask()
        mask = self.mask
        bmat = 1.0/(mask/self.sigma**2 + 1.0/t**2)
        bmat = torch.sqrt(bmat)
        return bmat

@register_operator(name='down_sampling')
class DownSampling(Operator):
    def __init__(self, resolution=256, scale_factor=4, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.scale_factor = scale_factor
        self.resolution = resolution
        self.target_size = resolution // scale_factor

    def __call__(self, x):
        return F.interpolate(x, size=(self.target_size, self.target_size),
                             mode='bicubic', align_corners=False)

class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size // 2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2, self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            # Generate a random motion blur kernel
            k = self._generate_motion_kernel(self.kernel_size, self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    @staticmethod
    def _generate_motion_kernel(kernel_size, intensity):
        """Generate a motion blur kernel without external dependency."""
        k = np.zeros((kernel_size, kernel_size), dtype=np.float64)
        center = kernel_size // 2
        angle = np.random.uniform(0, np.pi)
        length = int(kernel_size * intensity)
        length = max(1, min(length, kernel_size))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for i in range(length):
            t = i - length // 2
            r = center + int(round(t * sin_a))
            c = center + int(round(t * cos_a))
            if 0 <= r < kernel_size and 0 <= c < kernel_size:
                k[r, c] = 1.0
        if k.sum() == 0:
            k[center, center] = 1.0
        k /= k.sum()
        return k

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k


@register_operator(name='gaussian_blur')
class GaussianBlur(Operator):
    def __init__(self, kernel_size, intensity, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))
        self.conv.requires_grad_(False)

    def __call__(self, data):
        return self.conv(data)


@register_operator(name='motion_blur')
class MotionBlur(Operator):
    def __init__(self, kernel_size, intensity, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)

        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))
        self.conv.requires_grad_(False)

    def __call__(self, data):
        return self.conv(data)


# Non-linear Operator
# @register_operator(name='phase_retrieval')
# class PhaseRetrieval(Operator):
#     def __init__(self, oversample=0.0, resolution=256, sigma=0.05):
#         super().__init__(sigma)
#         self.pad = int((oversample / 8.0) * resolution)

#     def __call__(self, x):
#         x = x * 0.5 + 0.5  # [-1, 1] -> [0, 1]
#         x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
#         if not torch.is_complex(x):
#             x = x.type(torch.complex64)
#         fft2_m = torch.view_as_complex(fft2c_new(torch.view_as_real(x)))
#         amplitude = fft2_m.abs()
#         # amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min())
#         return amplitude


# @register_operator(name='nonlinear_blur')
# class NonlinearBlur(Operator):
#     def __init__(self, opt_yml_path, device='cuda', sigma=0.05):
#         super().__init__(sigma)
#         self.device = device
#         self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)
#         self.blur_model.requires_grad_(False)

#         np.random.seed(0)
#         kernel_np = np.random.randn(1, 512, 2, 2) * 1.2
#         random_kernel = (torch.from_numpy(kernel_np)).float().to(self.device)
#         self.random_kernel = random_kernel

#     def prepare_nonlinear_blur_model(self, opt_yml_path):
#         from .bkse.models.kernel_encoding.kernel_wizard import KernelWizard

#         with open(opt_yml_path, "r") as f:
#             opt = yaml.safe_load(f)["KernelWizard"]
#             model_path = opt["pretrained"]
#         blur_model = KernelWizard(opt)
#         blur_model.eval()
#         blur_model.load_state_dict(torch.load(model_path))
#         blur_model = blur_model.to(self.device)
#         self.random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
#         return blur_model

#     def call_old(self, data):
#         # random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
#         data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]
#         blurred = []
#         for i in range(data.shape[0]):
#             single_blurred = self.blur_model.adaptKernel(data[i:i + 1], kernel=self.random_kernel)
#             blurred.append(single_blurred)
#         blurred = torch.cat(blurred, dim=0)
#         blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
#         return blurred

#     def __call__(self, data):
#         data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]

#         random_kernel = self.random_kernel.repeat(data.shape[0], 1, 1, 1)
#         blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
#         blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]

#         # blurred = []
#         # for i in range(data.shape[0]):
#         #     single_blurred = self.blur_model.adaptKernel(data[i:i + 1], kernel=self.random_kernel)
#         #     blurred.append(single_blurred)
#         # blurred = torch.cat(blurred, dim=0)
#         # blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
#         return blurred


@register_operator(name='high_dynamic_range')
class HighDynamicRange(Operator):
    def __init__(self, device='cuda', scale=2, sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.scale = scale

    def __call__(self, data):
        return torch.clip((data * self.scale), -1, 1)


class LatentWrapper(Operator):
    def __init__(self, op, model):
        super().__init__(sigma=op.sigma)
        self.op = op
        self.model = model

    def __call__(self, x):
        decoded = self.model.decode(x)
        return self.op(decoded)


    def loss(self, pred, observation):
        decoded = self.model.decode(pred)
        return self.op.loss(decoded.float(), observation)

    def gradient(self, pred, observation, return_loss=False):
        pred_tmp = pred.clone().detach().requires_grad_(True)
        loss = self.loss(pred_tmp, observation).sum()
        pred_grad = grad(loss, pred_tmp)[0]
        pred_grad = pred_grad.to(pred.dtype)
        # clip the gradient
        pred_grad = torch.clamp(pred_grad, -1, 1)
        if return_loss:
            return pred_grad, loss
        else:
            return pred_grad
        
