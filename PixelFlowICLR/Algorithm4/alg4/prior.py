"""The prior covariance surrogate S of (12)/(22): per-class spectral S from ImageNet-val statistics.

S_k = F^H diag(P_k) F, where P_k (H x W, strictly positive) is the floored, centred mean power spectrum of the
image's own ImageNet synset at stage k (computed by scripts/compute_s_stats.py). S is fixed, not a config option.
"""
import os
import copy
import numpy as np
import torch


class SOperator:
    """Interface Block 1 needs: S^-1 x and S^-1/2 x (both exact for operators diagonal in an orthonormal basis)."""
    scalar_equiv = float("nan")
    meta = {}

    def apply_S_inv(self, x):
        raise NotImplementedError

    def apply_S_inv_sqrt(self, x):
        raise NotImplementedError


class SpectralSOp(SOperator):
    """S diagonal in the orthonormal 2-D Fourier basis: S(w) = power[w], shared across the three channels."""

    def __init__(self, power, meta=None):
        if not torch.is_tensor(power) or power.dim() != 2:
            raise ValueError("spectral power must be a 2-D tensor (H, W)")
        if not bool((power > 0).all()):
            raise ValueError("spectral power must be strictly positive (floor it first)")
        self.power = power.detach().to(torch.float32)
        self.scalar_equiv = float(power.mean())
        self.meta = dict(meta or {}, mode="spectral", scalar_equiv=self.scalar_equiv)

    def _apply(self, x, p):
        X = torch.fft.fft2(x, norm="ortho")
        return torch.fft.ifft2(X / p.to(x.device), norm="ortho").real

    def apply_S_inv(self, x):
        return self._apply(x, self.power)

    def apply_S_inv_sqrt(self, x):
        return self._apply(x, self.power.sqrt())

    def inv_diag_mean(self):
        """Exact constant diagonal of S^-1 = mean_w 1/P(w) (used to build the Jacobi probe correctly)."""
        return float(self.power.reciprocal().mean())


class ClassSpectralPrior:
    """s2_fn(stage_idx, sigma_tau) -> SpectralSOp for the bound ImageNet class.

    Unbound until ``bind(class_idx)`` (returns a cheap view sharing the npz handle and operator cache).
    npz keys: f"{synset}_stage{k}" (per class) -- produced by scripts/compute_s_stats.py.
    """

    def __init__(self, npz_path, synset_map_path, num_stages):
        self.npz_path, self.K = npz_path, int(num_stages)
        with open(synset_map_path) as f:
            self.synsets = [ln.split()[0] for ln in f if ln.strip()]
        if len(self.synsets) != 1000:
            raise ValueError(f"{synset_map_path}: expected 1000 synsets, got {len(self.synsets)}")
        self.npz = np.load(npz_path)      # lazy per key: only the bound class's arrays are read
        self._cache = {}
        self.class_idx = None

    def bind(self, class_idx):
        class_idx = int(class_idx)
        if not 0 <= class_idx < len(self.synsets):
            raise ValueError(f"class_idx {class_idx} outside 0..{len(self.synsets) - 1}")
        b = copy.copy(self)
        b.class_idx = class_idx
        return b

    def _ops(self):
        if self.class_idx is None:
            raise RuntimeError("ClassSpectralPrior is unbound: call .bind(class_idx) first")
        syn = self.synsets[self.class_idx]
        if syn not in self._cache:
            miss = [k for k in range(self.K) if f"{syn}_stage{k}" not in self.npz.files]
            if miss:
                raise KeyError(f"{self.npz_path}: no '{syn}_stage{{k}}' for stages {miss}")
            self._cache[syn] = {
                k: SpectralSOp(torch.from_numpy(self.npz[f"{syn}_stage{k}"]),
                               meta=dict(source=os.path.basename(self.npz_path), key=f"{syn}_stage{k}"))
                for k in range(self.K)}
        return self._cache[syn]

    def __call__(self, k, sigma_tau):
        return self._ops()[int(k)]

    def describe(self):
        if self.class_idx is None:
            return "spectral[class=<unbound>]"
        ops = self._ops()
        return (f"spectral[class={self.synsets[self.class_idx]}] Tr(S_k)/D = "
                f"{[round(float(ops[k].scalar_equiv), 6) for k in range(self.K)]}")
