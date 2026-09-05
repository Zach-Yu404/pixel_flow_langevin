#!/usr/bin/env python
"""Regenerate data/motion_kernel_k<size>_i<intensity>_seed<seed>.npy from the DAPS motion-blur PSF generator
(baselines/DAPS/forward_operator/motionblur/motionblur.py). Only needed if the data file is missing.
    python scripts/make_motion_kernel.py --daps /path/to/DAPS [--size 61 --intensity 0.5 --seed 42]"""
import argparse, os, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--daps", required=True); ap.add_argument("--size", type=int, default=61)
ap.add_argument("--intensity", type=float, default=0.5); ap.add_argument("--seed", type=int, default=42); a = ap.parse_args()
sys.path.insert(0, a.daps)
from forward_operator.motionblur.motionblur import Kernel
np.random.seed(a.seed)
K = np.asarray(Kernel(size=(a.size, a.size), intensity=a.intensity).kernelMatrix, dtype=np.float32)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", f"motion_kernel_k{a.size}_i{a.intensity}_seed{a.seed}.npy")
np.save(out, K); print(out, K.shape, "sum", float(K.sum()))
