#!/usr/bin/env python
"""Run the FINAL article version sampler and compare to article baseline."""
import os, sys, time, copy, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

# Import final sampler's main
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "final_sampler",
    os.path.join(os.path.dirname(__file__), "ms_posterior_sampling_article_version_final.py")
)
final_sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(final_sampler)

# Prepare config
import json
cfg_path = "debug_IP2/ms_posterior_sampling_article_version_final.json"
t0 = time.time()
final_sampler.main(cfg_path)
print(f"Final sampler done in {time.time()-t0:.1f}s")
