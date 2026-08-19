#!/usr/bin/env python
"""Test-side measurement overrides — nothing outside test/ is modified.

Two settings differ between our pipeline and the eight baselines on the two
inpainting tasks (see .research/tasks/measurement-alignment-inpainting.md):

  noise  demo_runner builds inpainting y as ``op(gt)`` (the operator's forward
         is ``mask * x``; the sigma handed to its constructor is stored and
         never used), so our y is noiseless while every baseline adds sigma:
         0.05 for DPS/FPS/DAPS/PSLD/ReSample, 0.10 for DDRM/DDNM/DiffPIR.
  box    our box sits at a random position; the shared demo15 protocol and
         DPS/FPS/DDRM/DDNM/DiffPIR use a centred 128x128 box.

Both are realigned here by composing functions that already exist:
``op.measure`` (inpaintingStart.py), ``make_box_mask(short_name=None)`` and
``_force_mask`` (demo_runner.py / pipeline.py). The PixelFlow scheduler and
``set_timesteps`` are deliberately untouched.

``apply()`` rebinds the name inside main.py and utils.py, so every existing
runner picks the new measurement up without being edited.
"""

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A2 = os.path.dirname(HERE)
if A2 not in sys.path:
    sys.path.insert(0, A2)

import utils                                                   # noqa: E402
import main as alg2                                            # noqa: E402
import torch                                                   # noqa: E402
from demo_runner import make_box_mask                          # noqa: E402
from pipeline import _force_mask                               # noqa: E402

INPAINT_TASKS = ("box_inpainting", "random_inpainting")
_ORIG_BUILD = alg2.build_setup_and_measurement
_STATE = {"inpaint_noise": False, "box_center": False, "noise_seed": 0}


def noise_seed_for(task, short_name, base_seed):
    """Stable across processes — unlike Python's hash(), which the legacy mask
    seeding uses and which is why PYTHONHASHSEED=0 is a hard contract here."""
    digest = hashlib.sha256(f"{task}/{short_name}/{base_seed}".encode()).hexdigest()
    return int(digest[:8], 16)


def patched_build_setup_and_measurement(task, op_cfg, demo, sigma_n, resolution,
                                        device, gpu_seed_offset=0):
    op, mask, y, label, panel, mkA, trep = _ORIG_BUILD(
        task, op_cfg, demo, sigma_n, resolution, device, gpu_seed_offset)

    want_center = _STATE["box_center"] and task == "box_inpainting"
    want_noise = _STATE["inpaint_noise"] and task in INPAINT_TASKS
    if not (want_center or want_noise):
        return op, mask, y, label, panel, mkA, trep

    gt = demo["gt"].unsqueeze(0).to(device)
    if want_center:
        box_len = int(op_cfg.get("mask_len_range", [128, 129])[0])
        mask = make_box_mask(resolution, box_len, device, short_name=None)
        _force_mask(op, mask)                       # operator and mask stay in sync
        label = f"Meas (box {box_len}x{box_len} centred · GT)"
        panel = mask * gt + (1 - mask) * (-torch.ones_like(gt))

    # y must be rebuilt whenever the mask moved, noise or not.
    rng_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(noise_seed_for(task, demo["short_name"],
                                         _STATE["noise_seed"]))
        y = (op.measure(gt) if want_noise else op(gt)).detach()
    finally:
        torch.set_rng_state(rng_state)              # leave the global stream as found
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return op, mask, y, label, panel, mkA, trep


def apply(inpaint_noise, box_center, noise_seed):
    """Rebind in both modules that imported the name. Returns what took effect."""
    _STATE.update(inpaint_noise=bool(inpaint_noise), box_center=bool(box_center),
                  noise_seed=int(noise_seed))
    alg2.build_setup_and_measurement = patched_build_setup_and_measurement
    utils.build_setup_and_measurement = patched_build_setup_and_measurement
    return dict(_STATE)


def restore():
    alg2.build_setup_and_measurement = _ORIG_BUILD
    utils.build_setup_and_measurement = _ORIG_BUILD
    _STATE.update(inpaint_noise=False, box_center=False)
