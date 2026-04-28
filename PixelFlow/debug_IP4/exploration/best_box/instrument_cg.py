"""Instrument cg_solve to log how often tol is hit vs max_iter.

Runs ONE batch (B=1 image) through the E1_W1+lam20 config with cg_max_iter=50
and tol=1e-5 (the production defaults), capturing per-call iteration count
and final relative residual.
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "debug_IP4"))
os.chdir(REPO)

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf

import ms_posterior_sampling_article_version_final_utils as utils
from inpaintingStart import get_operator
from pixelflow.utils import config as config_utils
from ms_sampler_v5 import run_ip4

LOG = []  # (iters_used, rel_residual, hit_tol)

_orig_cg = utils.cg_solve

def cg_solve_logged(A_fn, b, x0=None, tol=1e-5, max_iter=50):
    Bsize = b.shape[0]
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - A_fn(x)
    p = r.clone()
    rs_old = (r * r).reshape(Bsize, -1).sum(dim=1)
    b_norm = (b * b).reshape(Bsize, -1).sum(dim=1).sqrt().clamp(min=1e-12)
    iters_used = max_iter
    hit_tol = False
    final_rel = None
    for it in range(max_iter):
        Ap = A_fn(p)
        pAp = (p * Ap).reshape(Bsize, -1).sum(dim=1).clamp(min=1e-12)
        alpha = (rs_old / pAp).reshape(Bsize, 1, 1, 1)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).reshape(Bsize, -1).sum(dim=1)
        rel = (rs_new.sqrt() / b_norm).max().item()
        if rel < tol:
            iters_used = it + 1
            hit_tol = True
            final_rel = rel
            break
        beta = (rs_new / rs_old.clamp(min=1e-12)).reshape(Bsize, 1, 1, 1)
        p = r + beta * p
        rs_old = rs_new
    if final_rel is None:
        final_rel = rel
    LOG.append((iters_used, final_rel, hit_tol))
    return x


# Monkey-patch in *both* import paths the codebase uses.
utils.cg_solve = cg_solve_logged
import langevin_v5
langevin_v5.cg_solve = cg_solve_logged


# ---------------------------------------------------------------- run a sample
DEVICE = "cuda:0"
print("loading model + image ...", flush=True)
config = OmegaConf.load("pretrained_models/c2img/config.yaml")
model = config_utils.instantiate_from_config(config.model).to(DEVICE)
ckpt = torch.load("pretrained_models/c2img/model.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt, strict=True); model.eval()

VAL = "/data/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/val"
fn = sorted(os.listdir(VAL))[0]   # any image
img = Image.open(os.path.join(VAL, fn)).convert("RGB")

def cca(p, sz=256):
    while min(*p.size) >= 2 * sz:
        p = p.resize(tuple(x // 2 for x in p.size), Image.BOX)
    sc = sz / min(*p.size)
    p = p.resize(tuple(round(x * sc) for x in p.size), Image.BICUBIC)
    a = np.array(p)
    cy = (a.shape[0] - sz) // 2; cx = (a.shape[1] - sz) // 2
    return Image.fromarray(a[cy:cy+sz, cx:cx+sz])

tf = transforms.Compose([
    transforms.Lambda(lambda p: cca(p, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
gt = tf(img).unsqueeze(0).to(DEVICE)
torch.manual_seed(0)
op = get_operator("inpainting", resolution=256, device=DEVICE, sigma=0.05,
                  mask_type="box", mask_len_range=(128,129), mask_prob_range=None)
y = op(gt).detach()

# E1_W1+lam20 config (production defaults: cg_tol=1e-5, cg_max_iter=50)
KW = json.load(open(os.path.join(HERE, "configs/E1_W1+lam20.json")))["kw"]
KW.pop("class_label", None); KW.pop("seed", None)

print(f"running run_ip4 (E1, B=1, cg_tol={KW['cg_tol']}, cg_max_iter={KW['cg_max_iter']}) ...", flush=True)
t0 = time.time()
xf, _, _, res, _, _ = run_ip4(
    model, config, gt, y, op, 0.05, DEVICE,
    class_label=[10], seed=0, **KW,
)
print(f"done in {time.time()-t0:.0f}s   total CG calls = {len(LOG)}", flush=True)


# ------------------------------------------------------------------ stats
iters = np.array([r[0] for r in LOG])
rels  = np.array([r[1] for r in LOG])
tol_h = np.array([r[2] for r in LOG])

print("\n" + "="*60)
print(f"Total CG calls           : {len(LOG)}")
print(f"Hit tol (rel<1e-5)       : {tol_h.sum():4d}  ({100*tol_h.mean():.1f}%)")
print(f"Hit max_iter ({KW['cg_max_iter']}) : {(~tol_h).sum():4d}  ({100*(~tol_h).mean():.1f}%)")
print(f"Iters used  -- min/median/mean/max = {iters.min()}/{int(np.median(iters))}/{iters.mean():.1f}/{iters.max()}")
print(f"Final rel-resid: median={np.median(rels):.2e}  max={rels.max():.2e}")

print("\nIter-count histogram (max_iter=50):")
bins = [0,2,5,10,15,20,25,30,35,40,45,50]
for i in range(len(bins)-1):
    n = ((iters > bins[i]) & (iters <= bins[i+1])).sum()
    bar = "#" * int(40 * n / len(LOG))
    print(f"  ({bins[i]:>2}, {bins[i+1]:>2}]  {n:4d}  {bar}")
print("="*60)
