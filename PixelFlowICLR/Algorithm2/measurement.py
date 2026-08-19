#!/usr/bin/env python
"""Measurement construction — the one place the forward operator's settings
take effect, driven entirely by config.json's "tasks_setup".

demo_runner.build_setup_and_measurement hardcodes two settings that put our
inpainting results on a different footing from every published baseline:

  noise  the inpainting branches build y as ``op(gt)``; the operator's forward
         is ``mask * x`` and the sigma handed to its constructor is stored but
         never used, so our y is noiseless. Blur and SR do add sigma. All eight
         baselines add noise to inpainting too (0.05 for DPS/FPS/DAPS/PSLD/
         ReSample, 0.10 for DDRM/DDNM/DiffPIR), so noiseless inpainting is a
         strictly easier problem than the one they solved.
  box    the box sits at a random position, while the shared demo15 protocol
         and DPS/FPS/DDRM/DDNM/DiffPIR centre it.

Both are now config keys — "measurement_mode" per task and "center" in the box
operator block — applied here on top of demo_runner's output using functions
that already exist (``op.measure``, ``make_box_mask(short_name=None)``,
``_force_mask``). main.py and utils.py import the name from this module, so
every runner picks the configured measurement up through its normal call.

The PixelFlow scheduler is not involved and is left untouched.
"""

import hashlib

import torch

from demo_runner import build_setup_and_measurement as _raw_build, make_box_mask
from pipeline import _force_mask

INPAINT_TASKS = ("box_inpainting", "random_inpainting")

# Populated by main() from config.json — no defaults in code.
_CFG = {"tasks_setup": {}, "seed": None}


def configure(tasks_setup, measurement_seed):
    _CFG["tasks_setup"] = tasks_setup
    _CFG["seed"] = int(measurement_seed)


def noise_seed_for(task, short_name):
    """Stable across processes. Python's hash() is not (the legacy mask seeding
    uses it, which is why PYTHONHASHSEED=0 is a hard contract)."""
    key = f"{task}/{short_name}/{_CFG['seed']}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def build_setup_and_measurement(task, op_cfg, demo, sigma_n, resolution, device,
                                gpu_seed_offset=0):
    """Same signature as demo_runner's, so existing call sites are unchanged."""
    op, mask, y, label, panel, mkA, trep = _raw_build(
        task, op_cfg, demo, sigma_n, resolution, device, gpu_seed_offset)

    spec = _CFG["tasks_setup"].get(task, {})
    mode = spec.get("measurement_mode", "measure")
    if task not in INPAINT_TASKS:
        # demo_runner always adds sigma here; honouring "call" would mean
        # rebuilding y and silently changing every blur/SR result.
        if mode != "measure":
            raise ValueError(
                f'tasks_setup."{task}".measurement_mode={mode!r}: only "measure" '
                "is available for blur/SR, whose y already carries sigma_n.")
        return op, mask, y, label, panel, mkA, trep
    if mode not in ("measure", "call"):
        raise ValueError(f'tasks_setup."{task}".measurement_mode={mode!r} '
                         'must be "measure" (noisy, baseline-aligned) or "call".')

    center = bool(op_cfg.get("center", False))
    if mode == "call" and not center:
        return op, mask, y, label, panel, mkA, trep      # legacy measurement

    gt = demo["gt"].unsqueeze(0).to(device)
    if center:
        box_len = int(op_cfg.get("mask_len_range", [128, 129])[0])
        mask = make_box_mask(resolution, box_len, device, short_name=None)
        _force_mask(op, mask)                    # operator and mask stay in sync
        label = f"Meas (box {box_len}x{box_len} centred · GT)"
        panel = mask * gt + (1 - mask) * (-torch.ones_like(gt))

    # y is rebuilt whenever the mask moved or noise is wanted. The global RNG is
    # restored afterwards so the sampler's own noise stream is untouched.
    rng_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(noise_seed_for(task, demo["short_name"]))
        y = (op.measure(gt) if mode == "measure" else op(gt)).detach()
    finally:
        torch.set_rng_state(rng_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return op, mask, y, label, panel, mkA, trep
