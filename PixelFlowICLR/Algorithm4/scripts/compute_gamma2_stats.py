#!/usr/bin/env python
"""gamma^2(k, tau) on ImageNet val (all 50 images per class): the per-stage, per-tau mean squared error of the
network velocity against the exact interpolant velocity (draft eq. 20 / Cor. 8),
    x_tau = H_tau x1 + sigma_tau x0,   d_exact = B_k x1 - (e_k - s_k) x0,   gamma^2 = mean ||v_theta(x_tau) - d_exact||^2,
with x1 the stage pyramid of the val image, x0 ~ N(0,I) seeded per (image, stage), CFG and class embedding as in
sampling. Output: data/gamma2_all.json (drop-in for config paths.gamma2_table) + data/gamma2_labelled.json.
Multi-GPU: run one process per GPU (classes are claimed atomically); then --merge.
    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=<g> python scripts/compute_gamma2_stats.py [--val-root DIR --synset-map F --val-solution F]
    python scripts/compute_gamma2_stats.py --merge
TF32 matmul is enabled by default (relative effect on gamma^2 ~1e-5; --fp32 to disable)."""
import argparse, csv, json, math, os, sys, time, zlib
import numpy as np, torch
from PIL import Image
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from alg4.data import center_crop_arr
from alg4.model import load_model
from alg4.ops import apply_H_tau, apply_B, compute_sigma_tau, gt_stage_pyramid
from alg4.sampler import make_velocity_fn, rope_for, stage_schedule

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
ap.add_argument("--val-root", default="/CBIG-Standard-ECE/Zach_dataset/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/val")
ap.add_argument("--val-solution", default="/CBIG-Standard-ECE/Zach_dataset/Zach_dataset/imageNet256/LOC_val_solution.csv")
ap.add_argument("--synset-map", default=os.path.join(ROOT, "data", "synsets.txt"))
ap.add_argument("--per-class", type=int, default=50); ap.add_argument("--chunk", type=int, default=50)
ap.add_argument("--shards", default=os.path.join(ROOT, "data", "gamma2_shards"))
ap.add_argument("--fp32", action="store_true"); ap.add_argument("--merge", action="store_true"); a = ap.parse_args()


def build_maps():
    syn_order = [l.split()[0] for l in open(a.synset_map) if l.strip()]
    img2syn = {}
    with open(a.val_solution) as f:
        r = csv.reader(f); next(r)
        for row in r: img2syn[row[0]] = row[1].split(" ")[0]
    return syn_order, img2syn


def eps_for(names, stage_idx, shape):
    outs = []
    for name in names:
        seed = (zlib.crc32(f"{name}|stage{stage_idx}".encode()) ^ 42) & 0x7FFFFFFF
        g = torch.Generator(device="cpu").manual_seed(seed); outs.append(torch.randn(shape[1:], generator=g))
    return torch.stack(outs, dim=0)


def load_images(fns):
    xs = []
    for fn in fns:
        p = Image.open(os.path.join(a.val_root, fn)).convert("RGB")
        x = torch.from_numpy(np.array(center_crop_arr(p, 256))).permute(2, 0, 1).float() / 255.0
        xs.append((x - 0.5) / 0.5)
    return torch.stack(xs)


def claim(cdir, syn):
    os.makedirs(cdir, exist_ok=True); p = os.path.join(cdir, syn)
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd); return True
    except FileExistsError:
        return False


def measure():
    cfg = json.load(open(a.config)); device = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = not a.fp32
    model_dir = cfg["paths"]["model_dir"]; model_dir = model_dir if os.path.isabs(model_dir) else os.path.normpath(os.path.join(ROOT, model_dir))
    config, model = load_model(model_dir, device); K = int(config.scheduler.num_stages); ncls = int(model.num_classes)
    kw = cfg["sampler_kw"]; gs = float(kw["guidance_scale"]); do_cfg = gs > 0
    syn_order, img2syn = build_maps(); syn2idx = {s: i for i, s in enumerate(syn_order)}
    per = {s: [] for s in syn_order}
    for fn in sorted(os.listdir(a.val_root)):
        s = img2syn.get(os.path.splitext(fn)[0])
        if s in per and len(per[s]) < a.per_class: per[s].append(fn)
    sched = {si: stage_schedule(config, si, kw["ode_steps_per_stage"], float(kw["shift"]), device) for si in range(K)}
    os.makedirs(a.shards, exist_ok=True); t0 = time.time(); n = 0
    for syn in syn_order:
        out = os.path.join(a.shards, f"{syn}.json")
        if os.path.exists(out) or not claim(a.shards + "_claims", syn): continue
        fns = per[syn]; gt = load_images(fns).to(device); pyr = gt_stage_pyramid(gt, K)
        pe = torch.full((len(fns),), syn2idx[syn], dtype=torch.int32, device=device)
        rec = dict(synset=syn, class_idx=syn2idx[syn], n=len(fns), files=fns, table={}, per_image_sq_mean={},
                   precision="fp32" if a.fp32 else "tf32")
        for si in range(K):
            sc = sched[si]; s_k, e_k = float(sc.start_t[si]), float(sc.end_t[si]); x1 = pyr[si]
            x0 = eps_for([os.path.splitext(f)[0] for f in fns], si, x1.shape).to(device)
            d_exact = apply_B(x1, s_k, e_k) - (e_k - s_k) * x0
            size_tensor, rope_pos = rope_for(model, x1.shape[-2], x1.shape[-1], device)
            rec["table"][str(si)] = {}; rec["per_image_sq_mean"][str(si)] = {}
            for step in range(len(sc.Timesteps)):
                T = sc.Timesteps[step]; tau = float(sc.t[step]); sigma_tau = compute_sigma_tau(tau, s_k, e_k)
                x_tau = apply_H_tau(x1, tau, s_k, e_k) + sigma_tau * x0
                vs = []
                for lo in range(0, x_tau.shape[0], a.chunk):
                    hi = min(lo + a.chunk, x_tau.shape[0]); pe_c = pe[lo:hi]
                    emb = torch.cat([ncls * torch.ones_like(pe_c), pe_c], 0) if do_cfg else pe_c
                    vfn = make_velocity_fn(model, T, emb, size_tensor, rope_pos, do_cfg, gs, si)
                    with torch.no_grad(): vs.append(vfn(x_tau[lo:hi]))
                err = ((torch.cat(vs, 0) - d_exact) ** 2).mean(dim=(1, 2, 3))
                rec["table"][str(si)][f"{round(tau, 6)}"] = float(err.mean())
                rec["per_image_sq_mean"][str(si)][f"{round(tau, 6)}"] = float((err ** 2).mean())
        tmp = out + f".tmp{os.getpid()}"; json.dump(rec, open(tmp, "w")); os.replace(tmp, out); n += 1
        print(f"[class] {syn} ({syn2idx[syn]}) n={len(fns)} done={n} elapsed={(time.time()-t0)/60:.0f}min", flush=True)
    print("SHARD DONE", flush=True)


def merge():
    syn_order, _ = build_maps(); recs = {}
    for syn in syn_order:
        p = os.path.join(a.shards, f"{syn}.json")
        if os.path.exists(p): recs[syn] = json.load(open(p))
    stages = sorted(next(iter(recs.values()))["table"], key=int); taus = {k: list(next(iter(recs.values()))["table"][k]) for k in stages}
    N = sum(r["n"] for r in recs.values()); table, std = {}, {}
    for k in stages:
        table[k], std[k] = {}, {}
        for t in taus[k]:
            m = sum(r["n"] * r["table"][k][t] for r in recs.values()) / N
            m2 = sum(r["n"] * r["per_image_sq_mean"][k][t] for r in recs.values()) / N
            table[k][t] = m; std[k][t] = math.sqrt(max(m2 - m * m, 0.0))
    meta = dict(n_images=N, n_classes=len(recs), per_class=a.per_class, val_root=a.val_root,
                definition="gamma2(k,tau) = mean_images mean_pixels ||v_theta(x_tau) - (B_k x1 - (e_k-s_k) x0)||^2",
                precision={r.get("precision", "fp32") for r in recs.values()}.__repr__())
    json.dump(dict(note="Algorithm-4 gamma^2 table (eq. 20), ImageNet val", meta=meta, table=table, std_across_images=std),
              open(os.path.join(ROOT, "data", "gamma2_all.json"), "w"), indent=1)
    json.dump(dict(meta=meta, classes={s: dict(class_idx=r["class_idx"], n=r["n"], table=r["table"]) for s, r in recs.items()}),
              open(os.path.join(ROOT, "data", "gamma2_labelled.json"), "w"), indent=1)
    print(f"merged {len(recs)} classes / {N} images -> data/gamma2_all.json, data/gamma2_labelled.json")


merge() if a.merge else measure()
