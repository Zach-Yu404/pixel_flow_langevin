#!/usr/bin/env python
"""Single-file, single-pass S statistics over the FULL ImageNet-256 val set.

Outputs (this directory):
  s_pooled_statistics_labelled.json   pooled s^2(k) per class (1000 x 4)
  s_pooled_statistics_all.json        pooled s^2(k) over all 50k images
  spectral_power_labelled.npz         P_k(w) per class, keys "<synset>_stage{k}"
  spectral_power_all.npz              P_k(w) over all,   keys "stage{k}"

Conventions (verbatim from the current implementation):
  * stage pyramid  : chain-of-bilinear-halving (onestep_mse_vs_t.gt_stage_pyramid)
  * pooled s^2     : main4 --mode measure_s2 — flatten pixels x channels per
                     stage, float64 n/sum/sum2, s2 = E[x^2]-E[x]^2, range [-1,1]
  * spectral P     : s_prior_methods.cmd_measure — center by the group MEAN
                     IMAGE, P = mean_{i,c} |fft2(z, norm="ortho")|^2,
                     floor = 1e-8 * max(P). One-pass via the exact identity
                     (1/N) sum|Fx|^2 - |F mu|^2 == (1/N) sum|F(x-mu)|^2.
  * preprocessing  : evaluate.cca (aspect-preserving BOX-halving + BICUBIC +
                     center-crop 256) + Normalize(0.5,0.5) -> [-1,1]
  * labels         : LOC_val_solution.csv first synset + LOC_synset_mapping.txt
Resume: per-class shards under ./_shards + checkpoint every 25 classes.
"""
import os, sys, csv, json, pickle, time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/CBIG-Standard-ECE/Zach_dataset/Zach_dataset/imageNet256"
VAL_ROOT = BASE + "/ILSVRC/Data/CLS-LOC/val"
SYNSET_MAP = BASE + "/LOC_synset_mapping.txt"
VAL_SOLUTION = BASE + "/LOC_val_solution.csv"
K = 4
RES = 256
FLOOR_REL = 1e-8
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
LIMIT_CLASSES = int(os.environ.get("SSTATS_LIMIT", "0"))  # 0 = all


def cca(p, sz=256):
    """verbatim: evaluate.cca (aspect-preserving resize + center crop)."""
    while min(*p.size) >= 2 * sz:
        p = p.resize(tuple(x // 2 for x in p.size), Image.BOX)
    sc = sz / min(*p.size)
    p = p.resize(tuple(round(x * sc) for x in p.size), Image.BICUBIC)
    a = np.array(p)
    cy = (a.shape[0] - sz) // 2
    cx = (a.shape[1] - sz) // 2
    return Image.fromarray(a[cy:cy + sz, cx:cx + sz])


def gt_stage_pyramid(gt, num_stages):
    """verbatim: onestep_mse_vs_t.gt_stage_pyramid (bilinear halving chain)."""
    pyr = {num_stages - 1: gt}
    cur = gt
    for k in range(num_stages - 2, -1, -1):
        cur = F.interpolate(cur, size=(cur.shape[-2] // 2, cur.shape[-1] // 2),
                            mode="bilinear")
        pyr[k] = cur
    return pyr


def build_maps():
    syn2idx, syn_order = {}, []
    with open(SYNSET_MAP) as f:
        for i, line in enumerate(f):
            s = line.strip().split(" ", 1)[0]
            syn2idx[s] = i
            syn_order.append(s)
    img2syn = {}
    with open(VAL_SOLUTION) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            img2syn[row[0]] = row[1].split(" ")[0]
    return syn2idx, syn_order, img2syn


class ValSet(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fn, syn = self.files[i]
        for att in range(6):
            try:
                p = Image.open(os.path.join(VAL_ROOT, fn)).convert("RGB")
                break
            except OSError:
                time.sleep(5)
        x = torch.from_numpy(np.array(cca(p, RES))).permute(2, 0, 1).float() / 255.0
        x = (x - 0.5) / 0.5
        return x, syn


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    return xs, [b[1] for b in batch]


class Acc:
    """Per-group accumulators, float64/complex128 on GPU."""

    def __init__(self):
        self.n = 0
        self.ps = {k: None for k in range(K)}   # pooled: [n_elems, sum, sum2]
        self.sF = {}                            # complex sum of fft2 per (k)
        self.sP = {}                            # sum |fft2|^2 per (k)

    def add(self, pyr, nimg):
        self.n += nimg
        for k in range(K):
            x = pyr[k].double()
            v = x.reshape(-1)
            if self.ps[k] is None:
                self.ps[k] = [0, 0.0, 0.0]
            self.ps[k][0] += int(v.numel())
            self.ps[k][1] += float(v.sum())
            self.ps[k][2] += float((v * v).sum())
            X = torch.fft.fft2(x, norm="ortho")          # (B,3,H,W) complex128
            f = X.sum(dim=0)
            p = X.abs().pow(2).sum(dim=0)
            if k not in self.sF:
                self.sF[k], self.sP[k] = f, p
            else:
                self.sF[k] += f
                self.sP[k] += p

    def pooled(self):
        out = {}
        for k in range(K):
            n, s, sq = self.ps[k]
            m = s / n
            out[str(k)] = sq / n - m * m
        return out

    def spectra(self):
        out = {}
        for k in range(K):
            mu = self.sF[k] / self.n                     # F(mean image), (3,H,W)
            P = (self.sP[k] / self.n - mu.abs().pow(2)).mean(dim=0)  # mean over C
            P = P.clamp_min(0)                           # fp guard
            floor = FLOOR_REL * float(P.max())
            out[k] = (P.clamp_min(floor).float().cpu().numpy(), floor,
                      int((P < floor).sum()))
        return out

    def merge_raw(self, other):
        self.n += other.n
        for k in range(K):
            if other.ps[k] is None:
                continue
            if self.ps[k] is None:
                self.ps[k] = [0, 0.0, 0.0]
            for j in range(3):
                self.ps[k][j] += other.ps[k][j]
            if k not in self.sF:
                self.sF[k] = other.sF[k].clone()
                self.sP[k] = other.sP[k].clone()
            else:
                self.sF[k] += other.sF[k]
                self.sP[k] += other.sP[k]


def main():
    t0 = time.time()
    syn2idx, syn_order, img2syn = build_maps()
    files = sorted((fn, img2syn[os.path.splitext(fn)[0]])
                   for fn in os.listdir(VAL_ROOT)
                   if fn.lower().endswith((".jpeg", ".jpg")))
    files.sort(key=lambda t: (t[1], t[0]))               # class-ordered
    if LIMIT_CLASSES:
        keep = set(sorted({s for _, s in files})[:LIMIT_CLASSES])
        files = [t for t in files if t[1] in keep]
    shards = os.path.join(HERE, "_shards")
    os.makedirs(shards, exist_ok=True)
    ckpt_p = os.path.join(shards, "checkpoint.pkl")
    done_classes, g = set(), Acc()
    if os.path.exists(ckpt_p):
        with open(ckpt_p, "rb") as f:
            ck = pickle.load(f)
        done_classes = ck["done"]
        g.n = ck["gn"]
        g.ps = ck["gps"]
        g.sF = {k: torch.from_numpy(v).to(DEV) for k, v in ck["gsF"].items()}
        g.sP = {k: torch.from_numpy(v).to(DEV) for k, v in ck["gsP"].items()}
        print(f"[resume] {len(done_classes)} classes done", flush=True)
    files = [t for t in files if t[1] not in done_classes]
    dl = DataLoader(ValSet(files), batch_size=50, num_workers=8,
                    shuffle=False, collate_fn=collate)
    cur_syn, cur = None, None
    n_done = len(done_classes)

    def flush_class(syn, acc):
        nonlocal n_done
        sp = acc.spectra()
        rec = dict(synset=syn, class_idx=syn2idx[syn], n_images=acc.n,
                   pooled_s2=acc.pooled(),
                   spectral_floor={str(k): sp[k][1] for k in range(K)},
                   spectral_floored_bins={str(k): sp[k][2] for k in range(K)})
        with open(os.path.join(shards, "labelled.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        np.savez(os.path.join(shards, f"{syn}.npz"),
                 **{f"stage{k}": sp[k][0] for k in range(K)})
        g.merge_raw(acc)
        done_classes.add(syn)
        n_done += 1
        if n_done % 25 == 0:
            with open(ckpt_p, "wb") as f:
                pickle.dump(dict(done=done_classes, gn=g.n, gps=g.ps,
                                 gsF={k: v.cpu().numpy() for k, v in g.sF.items()},
                                 gsP={k: v.cpu().numpy() for k, v in g.sP.items()}), f)
            print(f"[ckpt] {n_done} classes, {g.n} images, "
                  f"{time.time()-t0:.0f}s", flush=True)

    for xs, syns in dl:
        xs = xs.to(DEV)
        i = 0
        while i < len(syns):
            j = i
            while j < len(syns) and syns[j] == syns[i]:
                j += 1
            if syns[i] != cur_syn:
                if cur_syn is not None:
                    flush_class(cur_syn, cur)
                cur_syn, cur = syns[i], Acc()
            cur.add(gt_stage_pyramid(xs[i:j], K), j - i)
            i = j
    if cur_syn is not None:
        flush_class(cur_syn, cur)

    # ---- final outputs ----
    meta = dict(
        source=VAL_ROOT, n_images=g.n, n_classes=len(done_classes),
        preprocessing="evaluate.cca (BOX-halving + BICUBIC + center-crop 256) "
                      "+ Normalize(0.5,0.5) -> [-1,1]",
        pyramid="chain-of-bilinear-halving (gt_stage_pyramid), stages 32/64/128/256",
        pooled_formula="per stage: flatten pixels x channels x images, "
                       "s2 = E[x^2] - E[x]^2 (float64 accumulation)",
        spectral_formula="P_k(w) = mean_{i,c} |fft2(x_i - mu_group, "
                         "norm='ortho')|^2, computed one-pass via "
                         "(1/N)sum|Fx|^2 - |F mu|^2 (exact); "
                         f"floor = {FLOOR_REL} * max(P_k); channels shared",
        script="s_stats/compute_s_stats.py", elapsed_s=int(time.time() - t0))
    recs = [json.loads(l) for l in open(os.path.join(shards, "labelled.jsonl"))]
    recs = {r["synset"]: r for r in recs}
    json.dump(dict(meta=meta, classes=recs),
              open(os.path.join(HERE, "s_pooled_statistics_labelled.json"), "w"),
              indent=1)
    sp = g.spectra()
    json.dump(dict(meta=meta, pooled_s2=g.pooled(),
                   spectral_floor={str(k): sp[k][1] for k in range(K)},
                   spectral_floored_bins={str(k): sp[k][2] for k in range(K)}),
              open(os.path.join(HERE, "s_pooled_statistics_all.json"), "w"),
              indent=1)
    np.savez(os.path.join(HERE, "spectral_power_all.npz"),
             **{f"stage{k}": sp[k][0] for k in range(K)})
    lab = {}
    for syn in sorted(done_classes):
        z = np.load(os.path.join(shards, f"{syn}.npz"))
        for k in range(K):
            lab[f"{syn}_stage{k}"] = z[f"stage{k}"]
    np.savez(os.path.join(HERE, "spectral_power_labelled.npz"), **lab)
    print(f"S STATS DONE n={g.n} classes={len(done_classes)} "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
