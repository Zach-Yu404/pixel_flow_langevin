"""Pre-build 8 deterministic GT images + 8 per-image random masks (mask_prob=0.7).

All 18 OAT configs share this exact data.pt for fair comparison.

Saves: data.pt with keys
  fnames: list[str]   (8 ImageNet val filenames)
  labels: list[int]   (8 class indices, per-image)
  gts:    Tensor[8, 3, 256, 256] in [-1, 1]
  masks:  Tensor[8, 1, 256, 256] (1=observed, 0=hole; ~30% observed per image)
  mask_seed: int
"""
import os, sys, json, csv

HERE = os.path.dirname(os.path.abspath(__file__))
DBG  = os.path.dirname(HERE)
REPO = os.path.dirname(DBG)
for p in (HERE, DBG, REPO): sys.path.insert(0, p)
os.chdir(REPO)

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from inpaintingStart import get_operator


VAL_ROOT     = "/data/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/val"
VAL_SOLUTION = "/data/Zach_dataset/imageNet256/LOC_val_solution.csv"
SYNSET_MAP   = "/data/Zach_dataset/imageNet256/LOC_synset_mapping.txt"

N         = 8
RES       = 256
MASK_PROB = 0.7      # fraction of pixels to MASK OUT (= 70% holes, 30% observed)
MASK_SEED = 2026
SAMPLE_SEED = 20260428
SIGMA_N   = 0.05


def cca(p, sz=256):
    while min(*p.size) >= 2 * sz:
        p = p.resize(tuple(x // 2 for x in p.size), Image.BOX)
    sc = sz / min(*p.size)
    p = p.resize(tuple(round(x * sc) for x in p.size), Image.BICUBIC)
    a = np.array(p); cy = (a.shape[0]-sz)//2; cx = (a.shape[1]-sz)//2
    return Image.fromarray(a[cy:cy+sz, cx:cx+sz])


def build_label_map():
    syn2idx = {}
    with open(SYNSET_MAP) as f:
        for i, line in enumerate(f):
            syn2idx[line.strip().split(" ", 1)[0]] = i
    img2syn = {}
    with open(VAL_SOLUTION) as f:
        r = csv.reader(f); next(r)
        for row in r:
            img2syn[row[0]] = row[1].strip().split(" ")[0]
    return {iid: syn2idx[s] for iid, s in img2syn.items()}


def main():
    DEVICE = "cuda:0"
    label_map = build_label_map()
    all_fnames = sorted(f for f in os.listdir(VAL_ROOT)
                        if f.lower().endswith((".jpeg", ".jpg")))
    print(f"[val] {len(all_fnames)} images")

    rng = np.random.RandomState(seed=SAMPLE_SEED)
    sel = rng.choice(len(all_fnames), size=N, replace=False)
    fnames = [all_fnames[i] for i in sel]
    labels = [label_map[os.path.splitext(fn)[0]] for fn in fnames]

    tf = transforms.Compose([
        transforms.Lambda(lambda p: cca(p, RES)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    gts = torch.stack([tf(Image.open(os.path.join(VAL_ROOT, f)).convert("RGB"))
                       for f in fnames]).to(DEVICE)
    print(f"  GTs: {tuple(gts.shape)}, labels={labels}")

    # Per-image random masks (8 distinct), 70% holes each
    np.random.seed(MASK_SEED); torch.manual_seed(MASK_SEED); torch.cuda.manual_seed_all(MASK_SEED)
    operator = get_operator(
        "inpainting", resolution=RES, device=DEVICE, sigma=SIGMA_N,
        mask_type="random", mask_len_range=None, mask_prob_range=(MASK_PROB, MASK_PROB + 1e-6),
    )
    masks = []
    for b in range(N):
        m = operator.get_mask(x=gts[b:b+1]).float()  # [1, 1, H, W]
        # WARNING: _retrieve_random returns shape [3, H, W] reshaped to [1, 1, H, W] via _init_mask
        masks.append(m[0])
    masks = torch.stack(masks).to(DEVICE)  # [8, 1, 256, 256]

    # Sanity: every mask has ~30% observed pixels
    obs_frac = masks.sum(dim=(1,2,3)) / (RES * RES)
    print(f"  observed fraction per img: {obs_frac.cpu().numpy().round(4).tolist()}")
    print(f"  mean observed: {obs_frac.mean().item():.4f}  (expected ≈ {1-MASK_PROB:.2f})")
    assert (obs_frac > 0.25).all() and (obs_frac < 0.35).all(), "mask prob mismatch"

    out = dict(fnames=fnames, labels=labels, gts=gts.cpu(), masks=masks.cpu(),
               mask_seed=MASK_SEED, mask_prob=MASK_PROB, sigma_n=SIGMA_N)
    p = os.path.join(HERE, "data.pt")
    torch.save(out, p)
    print(f"[done] saved {p}")


if __name__ == "__main__":
    main()
