"""Prepare 5 disjoint validation groups of 100 ImageNet val images each.

Outputs per group g (0..4):
  groups/group_{g}.pt with keys:
    - fnames: list[str]                # 100 filenames
    - labels: list[int]                # 100 class indices (0..999)
    - gts:    Tensor[100, 3, 256, 256] # in [-1, 1]
    - masks:  Tensor[100, 1, 256, 256] # 1=observed, 0=hole; one 128x128 box per image
    - mask_seed: int                   # seed used for mask generation

Determinism:
  - image sampling: np.random.RandomState(seed=g) over sorted val filenames
  - class label:    LOC_val_solution.csv (filename->synset) + LOC_synset_mapping.txt (synset->idx via 1-based line)
  - mask:           np.random.seed(1000 + g) + torch.manual_seed(1000 + g) + cuda seed; same seed gives same 100 masks
"""
import os, sys, json, csv

HERE = os.path.dirname(os.path.abspath(__file__))      # val/
LB   = os.path.dirname(HERE)                           # larger_box
DBG  = os.path.dirname(LB)                             # debug_IP4
REPO = os.path.dirname(DBG)                            # PixelFlow
for p in (HERE, LB, DBG, REPO): sys.path.insert(0, p)
os.chdir(REPO)

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from inpaintingStart import get_operator


VAL_ROOT      = "/data/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/val"
VAL_SOLUTION  = "/data/Zach_dataset/imageNet256/LOC_val_solution.csv"
SYNSET_MAP    = "/data/Zach_dataset/imageNet256/LOC_synset_mapping.txt"

NUM_GROUPS    = 5
GROUP_SIZE    = 100
RES           = 256
BOX_LEN       = 128
MASK_SEED_OFFSET = 1000   # group g uses mask seed (1000 + g)
SIGMA_N       = 0.05      # measurement noise (matches sweep)


def cca(p, sz=256):
    while min(*p.size) >= 2 * sz:
        p = p.resize(tuple(x // 2 for x in p.size), Image.BOX)
    sc = sz / min(*p.size)
    p = p.resize(tuple(round(x * sc) for x in p.size), Image.BICUBIC)
    a = np.array(p); cy = (a.shape[0]-sz)//2; cx = (a.shape[1]-sz)//2
    return Image.fromarray(a[cy:cy+sz, cx:cx+sz])


def build_label_map():
    """Build {image_id (no ext) -> class_idx (0..999)}."""
    syn2idx = {}
    with open(SYNSET_MAP) as f:
        for i, line in enumerate(f):
            syn = line.strip().split(" ", 1)[0]
            syn2idx[syn] = i  # 0-based
    img2syn = {}
    with open(VAL_SOLUTION) as f:
        r = csv.reader(f)
        next(r)  # skip header
        for row in r:
            image_id = row[0]
            syn = row[1].strip().split(" ")[0]
            img2syn[image_id] = syn
    img2idx = {iid: syn2idx[s] for iid, s in img2syn.items()}
    assert len(syn2idx) == 1000, f"expected 1000 synsets, got {len(syn2idx)}"
    print(f"[label_map] {len(syn2idx)} synsets, {len(img2idx)} val images mapped")
    return img2idx


def main():
    label_map = build_label_map()
    all_fnames = sorted(os.listdir(VAL_ROOT))
    all_fnames = [f for f in all_fnames if f.lower().endswith(".jpeg") or f.lower().endswith(".jpg")]
    print(f"[val] {len(all_fnames)} images in {VAL_ROOT}")
    assert len(all_fnames) >= NUM_GROUPS * GROUP_SIZE

    # Disjoint sampling: one global RandomState picks all 500 indices, then split.
    rng = np.random.RandomState(seed=20260428)
    chosen_idx = rng.choice(len(all_fnames), size=NUM_GROUPS * GROUP_SIZE, replace=False)
    print(f"[sample] picked {len(chosen_idx)} disjoint indices from val (seed=20260428)")

    tf = transforms.Compose([
        transforms.Lambda(lambda p: cca(p, RES)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    DEVICE = "cuda:0"
    summary_rows = []

    for g in range(NUM_GROUPS):
        sel = chosen_idx[g*GROUP_SIZE : (g+1)*GROUP_SIZE]
        fnames = [all_fnames[i] for i in sel]
        # class labels per image
        labels = []
        for fn in fnames:
            iid = os.path.splitext(fn)[0]
            assert iid in label_map, f"missing label for {iid}"
            labels.append(label_map[iid])

        # load + transform GTs (stack)
        gts = []
        for fn in fnames:
            img = Image.open(os.path.join(VAL_ROOT, fn)).convert("RGB")
            gts.append(tf(img))
        gts = torch.stack(gts).to(DEVICE)  # [100, 3, 256, 256] in [-1, 1]
        assert gts.shape == (GROUP_SIZE, 3, RES, RES)

        # masks: deterministic 100 box masks
        mask_seed = MASK_SEED_OFFSET + g
        np.random.seed(mask_seed)
        torch.manual_seed(mask_seed)
        torch.cuda.manual_seed_all(mask_seed)
        operator = get_operator(
            "inpainting", resolution=RES, device=DEVICE, sigma=SIGMA_N,
            mask_type="box", mask_len_range=(BOX_LEN, BOX_LEN+1), mask_prob_range=None,
        )
        masks = []
        for b in range(GROUP_SIZE):
            # _get_mask_and_input draws one fresh mask. Pass the b-th GT (for shape).
            m = operator.get_mask(x=gts[b:b+1]).float()  # [1, 1, 256, 256]
            assert m.shape == (1, 1, RES, RES), f"unexpected mask shape {m.shape}"
            masks.append(m[0])  # [1, 256, 256]
        masks = torch.stack(masks).to(DEVICE)  # [100, 1, 256, 256]
        # sanity: every mask should have exactly BOX_LEN*BOX_LEN holes
        n_holes = (1 - masks).sum(dim=(1,2,3))
        assert (n_holes == BOX_LEN*BOX_LEN).all().item(), \
            f"mask hole counts not all {BOX_LEN**2}: {n_holes.unique().tolist()}"

        out = {
            "fnames":  fnames,
            "labels":  labels,
            "gts":     gts.cpu(),
            "masks":   masks.cpu(),
            "mask_seed": mask_seed,
            "group_id": g,
        }
        out_path = os.path.join(HERE, "groups", f"group_{g}.pt")
        torch.save(out, out_path)
        print(f"[group {g}] saved {out_path}  fnames[0]={fnames[0]} label[0]={labels[0]} "
              f"mask_seed={mask_seed}  holes/img={int(n_holes[0].item())}")

        summary_rows.append(dict(
            group_id=g, num_images=GROUP_SIZE, mask_seed=mask_seed,
            first_fname=fnames[0], first_label=labels[0],
            last_fname=fnames[-1], last_label=labels[-1],
            label_min=int(min(labels)), label_max=int(max(labels)),
            label_unique=len(set(labels)),
        ))

    # write a small index for human inspection
    with open(os.path.join(HERE, "groups", "groups_index.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\n[done] wrote groups/groups_index.json  ({NUM_GROUPS} groups)")


if __name__ == "__main__":
    main()
