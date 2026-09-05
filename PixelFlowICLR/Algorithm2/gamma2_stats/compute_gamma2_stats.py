#!/usr/bin/env python
"""gamma^2(k, tau) on ImageNet val: all 50 images per class (1000 classes = 50k images), TF32 matmul (G2_TF32=0 for fp32),
all-classes table + per-class table. Single file; resumable; multi-GPU by class sharding with atomic claims.

Definition (draft eq. 20 / Cor. 8, exactly as main.py's onestep loop and gamma2_meas_alg4.json):
    x1 = stage-k GT pyramid (evaluate.cca preprocessing + Normalize(0.5,0.5), chain-of-bilinear-halving),
    x0 = eps_for(image, k)   deterministic N(0,I) per (image, stage): seed crc32(f"{name}|stage{k}") ^ 42,
    x_tau = H_tau x1 + sigma_tau x0,   v = v_theta(x_tau, tau, k)  (CFG, guidance_scale from config sampler_kw,
                                       class embedding = the image's own ImageNet class; REAL G, eff_si=None),
    d_exact = B_k x1 - (e_k - s_k) x0,
    gamma^2(k, tau) = mean_images mean_pixels ||v - d_exact||^2 .
tau grid = the sampler's own per-stage schedule (PixelFlowScheduler, ode_steps_per_stage, shift), keyed
str(round(tau, 6)) like the existing table so run_posterior_sampling_alg4's lookup works unchanged.

Usage (one process per GPU, they share the class list through claims):
    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=<g> python gamma2_stats/compute_gamma2_stats.py
    PYTHONHASHSEED=0 python gamma2_stats/compute_gamma2_stats.py --merge      # after all shards finish
Env: G2_PER_CLASS (default 50 = all), G2_CHUNK (50), G2_LIMIT_CLASSES (0 = all), G2_TF32 (1), G2_ONLY (synsets), G2_SHARDS (dir)."""
import os, sys, csv, json, time, zlib, math, argparse
import numpy as np, torch
from PIL import Image
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); ALG2 = os.path.dirname(HERE)
sys.path.insert(0, ALG2); os.chdir(ALG2)
BASE = "/CBIG-Standard-ECE/Zach_dataset/Zach_dataset/imageNet256"
VAL_ROOT = BASE + "/ILSVRC/Data/CLS-LOC/val"; SYNSET_MAP = BASE + "/LOC_synset_mapping.txt"; VAL_SOLUTION = BASE + "/LOC_val_solution.csv"
PER_CLASS = int(os.environ.get("G2_PER_CLASS", "50"))   # 50 = every val image (user 2026-09-04: 全量 5w)
LIMIT = int(os.environ.get("G2_LIMIT_CLASSES", "0"))
SHARDS = os.path.abspath(os.environ.get("G2_SHARDS", os.path.join(HERE, "shards"))); CLAIMS = SHARDS + "_claims"   # abspath: utils import chdir()s to IP_package
ONLY = [x for x in os.environ.get("G2_ONLY", "").split(",") if x]   # restrict to these synsets (precision cross-check)
RES = 256
TF32 = os.environ.get("G2_TF32", "1") == "1"    # 1 = TF32 matmul (default since 2026-09-04), 0 = strict fp32
CHUNK = int(os.environ.get("G2_CHUNK", "50"))   # images per forward pass (CFG doubles it); user 2026-09-04: larger batch (gain <1%, stage3 peak ~25 GB)

def cca(p, sz=256):
    """verbatim: evaluate.cca (aspect-preserving resize + center crop)."""
    while min(*p.size) >= 2 * sz:
        p = p.resize(tuple(x // 2 for x in p.size), Image.BOX)
    sc = sz / min(*p.size); p = p.resize(tuple(round(x * sc) for x in p.size), Image.BICUBIC)
    a = np.array(p); cy = (a.shape[0] - sz) // 2; cx = (a.shape[1] - sz) // 2
    return Image.fromarray(a[cy:cy + sz, cx:cx + sz])

def gt_stage_pyramid(gt, num_stages):
    """verbatim: onestep_mse_vs_t.gt_stage_pyramid (bilinear halving chain)."""
    pyr = {num_stages - 1: gt}; cur = gt
    for k in range(num_stages - 2, -1, -1):
        cur = F.interpolate(cur, size=(cur.shape[-2] // 2, cur.shape[-1] // 2), mode="bilinear"); pyr[k] = cur
    return pyr

def eps_for(names, stage_idx, shape):
    """verbatim scheme of onestep_mse_vs_t.eps_for: one N(0,I) per (image name, stage)."""
    outs = []
    for name in names:
        seed = (zlib.crc32(f"{name}|stage{stage_idx}".encode()) ^ 42) & 0x7FFFFFFF
        g = torch.Generator(device="cpu").manual_seed(seed); outs.append(torch.randn(shape[1:], generator=g))
    return torch.stack(outs, dim=0)

def build_maps():
    syn2idx, syn_order = {}, []
    with open(SYNSET_MAP) as f:
        for i, line in enumerate(f):
            s = line.strip().split(" ", 1)[0]; syn2idx[s] = i; syn_order.append(s)
    img2syn = {}
    with open(VAL_SOLUTION) as f:
        r = csv.reader(f); next(r)
        for row in r: img2syn[row[0]] = row[1].split(" ")[0]
    return syn2idx, syn_order, img2syn

def load_images(fns):
    xs = []
    for fn in fns:
        for att in range(8):
            try:
                p = Image.open(os.path.join(VAL_ROOT, fn)).convert("RGB"); break
            except OSError:
                time.sleep(5)
        x = torch.from_numpy(np.array(cca(p, RES))).permute(2, 0, 1).float() / 255.0
        xs.append((x - 0.5) / 0.5)
    return torch.stack(xs)

def retry(fn, tries=12, delay=10, what=""):
    """ceph EIO (errno 121) retry wrapper for directory listings / small file ops."""
    for att in range(tries):
        try:
            return fn()
        except OSError as e:
            if att == tries - 1: raise
            print(f"[eio-retry {att}] {what}: {e}", flush=True); time.sleep(delay)

def class_files(syn_order, img2syn):
    """first PER_CLASS val files (sorted by filename) of every synset -> {syn: [fn, ...]};
    cached in gamma2_stats/val_subset.json so the 50k-entry listdir happens once."""
    cache = os.path.join(HERE, f"val_subset_{PER_CLASS}.json")
    if os.path.exists(cache):
        return retry(lambda: json.load(open(cache)), what="read subset cache")
    per = {s: [] for s in syn_order}
    for fn in sorted(retry(lambda: os.listdir(VAL_ROOT), what="listdir val")):
        s = img2syn.get(os.path.splitext(fn)[0])
        if s in per and len(per[s]) < PER_CLASS: per[s].append(fn)
    tmp = cache + f".tmp{os.getpid()}"; json.dump(per, open(tmp, "w")); os.replace(tmp, cache)
    return per

def claim(syn):
    return retry(lambda: _claim(syn), what=f"claim {syn}")

def _claim(syn):
    os.makedirs(CLAIMS, exist_ok=True); p = os.path.join(CLAIMS, syn)
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd); return True
    except FileExistsError:
        if time.time() - os.path.getmtime(p) > 3600:   # stale (> 1 h) -> reclaim
            try:
                os.remove(p); fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd); return True
            except (FileExistsError, FileNotFoundError):
                return False
        return False

def measure():
    import main4, s_prior_methods as SP, onestep_mse_vs_t as base
    from utils import make_velocity_fn, apply_H_tau, apply_B, compute_sigma_tau
    from omegaconf import OmegaConf
    device = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = TF32   # user 2026-09-04: TF32 matmul (v rel. err 3e-3, gamma2 bias <1e-3 rel.)
    SP._init_globals()
    config = OmegaConf.load(os.path.join(main4.PATHS["model_dir"], "config.yaml"))
    K = int(config.scheduler.num_stages); kw = dict(main4.SAMPLER_KW)
    gs = float(kw["guidance_scale"]); do_cfg = gs > 0; shift = float(kw["shift"]); ode = int(kw["ode_steps_per_stage"])
    model = main4._load_model(config, device); ncls = int(model.num_classes)
    print(f"[setup] K={K} ode_steps={ode} shift={shift} guidance_scale={gs} cfg={do_cfg} per_class={PER_CLASS} "
          f"tf32(matmul)={torch.backends.cuda.matmul.allow_tf32} dtype={next(model.parameters()).dtype}", flush=True)
    syn2idx, syn_order, img2syn = retry(build_maps, what="build_maps"); per = class_files(syn_order, img2syn)
    classes = ONLY if ONLY else (syn_order[:LIMIT] if LIMIT else syn_order)
    scheds = {}
    for si in range(K):
        sc = main4._stage_schedule(config, si, ode, shift, device)
        scheds[si] = dict(sc=sc, s_k=float(sc.start_t[si]), e_k=float(sc.end_t[si]),
                          rope=base.rope_for(model, 32 * 2 ** si, 32 * 2 ** si, device))
    os.makedirs(SHARDS, exist_ok=True); t_start = time.time(); n_done = 0
    for syn in classes:
        out = os.path.join(SHARDS, f"{syn}.json")
        if retry(lambda: os.path.exists(out), what="exists") or not claim(syn): continue
        t0 = time.time(); fns = per[syn]
        gt = retry(lambda: load_images(fns), what=f"load {syn}").to(device); pyr = gt_stage_pyramid(gt, K)
        pe = torch.full((len(fns),), syn2idx[syn], dtype=torch.int32, device=device)
        rec = dict(synset=syn, class_idx=syn2idx[syn], files=fns, n=len(fns), table={}, per_image_sq_mean={},
                   precision="tf32" if TF32 else "fp32")
        for si in range(K):
            S = scheds[si]; sc = S["sc"]; x1 = pyr[si]; names = [os.path.splitext(f)[0] for f in fns]
            x0 = eps_for(names, si, x1.shape).to(device)
            d_exact = apply_B(x1, S["s_k"], S["e_k"], None) - (S["e_k"] - S["s_k"]) * x0
            size_tensor, rope_pos = S["rope"]; rec["table"][str(si)] = {}; rec["per_image_sq_mean"][str(si)] = {}
            for step in range(len(sc.Timesteps)):
                T = sc.Timesteps[step]; tau = float(sc.t[step]); sigma_tau = compute_sigma_tau(tau, S["s_k"], S["e_k"])
                x_tau = apply_H_tau(x1, tau, S["s_k"], S["e_k"], None) + sigma_tau * x0
                vs = []
                for lo in range(0, x_tau.shape[0], CHUNK):
                    hi = min(lo + CHUNK, x_tau.shape[0]); pe_c = pe[lo:hi]
                    emb_c = torch.cat([ncls * torch.ones_like(pe_c), pe_c], 0) if do_cfg else pe_c
                    vfn = make_velocity_fn(model, T, emb_c, size_tensor, rope_pos, do_cfg, gs, si)
                    with torch.no_grad():
                        vs.append(vfn(x_tau[lo:hi]))
                v = torch.cat(vs, 0)
                err = ((v - d_exact) ** 2).mean(dim=(1, 2, 3))          # per image
                key = f"{round(tau, 6)}"
                rec["table"][str(si)][key] = float(err.mean())
                rec["per_image_sq_mean"][str(si)][key] = float((err ** 2).mean())   # for across-image std at merge
        def _write():
            tmp = out + f".tmp{os.getpid()}"; json.dump(rec, open(tmp, "w")); os.replace(tmp, out)
        retry(_write, what=f"write {syn}")
        n_done += 1; el = time.time() - t_start
        print(f"[class] {syn} ({syn2idx[syn]}) n={len(fns)} {time.time()-t0:.0f}s  stage3 tau0/last: "
              f"{list(rec['table'][str(K-1)].values())[0]:.4f}/{list(rec['table'][str(K-1)].values())[-1]:.4f}  "
              f"done_by_me={n_done} elapsed={el/60:.0f}min", flush=True)
    print("G2 SHARD DONE", flush=True)

def merge():
    syn2idx, syn_order, _ = build_maps()
    recs = {}
    for syn in syn_order:
        p = os.path.join(SHARDS, f"{syn}.json")
        if os.path.exists(p): recs[syn] = json.load(open(p))
    if not recs: raise SystemExit("no shards")
    stages = sorted(next(iter(recs.values()))["table"].keys(), key=int)
    taus = {k: list(next(iter(recs.values()))["table"][k].keys()) for k in stages}
    N = sum(r["n"] for r in recs.values())
    prec = {}
    for r in recs.values(): prec[r.get("precision", "fp32")] = prec.get(r.get("precision", "fp32"), 0) + 1
    table, std, rows = {}, {}, []
    for k in stages:
        table[k], std[k] = {}, {}
        for t in taus[k]:
            m = sum(r["n"] * r["table"][k][t] for r in recs.values()) / N
            m2 = sum(r["n"] * r["per_image_sq_mean"][k][t] for r in recs.values()) / N
            table[k][t] = m; std[k][t] = math.sqrt(max(m2 - m * m, 0.0))
            rows.append(dict(stage=k, tau=t, gamma2_all=m, std_across_images=std[k][t], n_images=N))
    meta = dict(source=VAL_ROOT, n_images=N, n_classes=len(recs), per_class=PER_CLASS,
                subset="first PER_CLASS val files of each synset, sorted by filename",
                definition="gamma2(k,tau) = mean_images mean_pixels ||v_theta(x_tau) - (B_k x1 - (e_k-s_k) x0)||^2; "
                           "x_tau = H_tau x1 + sigma_tau x0; x0 ~ eps_for(name, k); CFG guidance from config sampler_kw; "
                           "class embedding = own class; REAL G (eff_si=None)",
                precision=dict(classes_by_precision=prec, note="matmul TF32 (v rel. err ~3e-3 vs fp32); classes without "
                               "a 'precision' field were measured in strict fp32 before the 2026-09-04 switch"),
                tau_grid="sampler per-stage schedule (PixelFlowScheduler, ode_steps_per_stage, shift); keys str(round(tau,6))",
                preprocessing="evaluate.cca (BOX-halving + BICUBIC + center-crop 256) + Normalize(0.5,0.5) -> [-1,1]",
                pyramid="chain-of-bilinear-halving (gt_stage_pyramid), stages 32/64/128/256",
                script="gamma2_stats/compute_gamma2_stats.py", reference_table="gamma2_meas_alg4.json (7 demo images)")
    json.dump(dict(note="Algorithm-4 gamma^2 table (eq. 20) measured on the ImageNet-val subset; drop-in for gamma2_meas_alg4.json",
                   meta=meta, table=table, std_across_images=std), open(os.path.join(HERE, "gamma2_all.json"), "w"), indent=1)
    json.dump(dict(meta=meta, classes={s: dict(class_idx=r["class_idx"], n=r["n"], table=r["table"]) for s, r in recs.items()}),
              open(os.path.join(HERE, "gamma2_labelled.json"), "w"), indent=1)
    with open(os.path.join(HERE, "gamma2_all.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    ref = json.load(open(os.path.join(ALG2, "gamma2_meas_alg4.json")))["table"]
    print(f"merged {len(recs)} classes / {N} images")
    for k in stages:
        print(f"stage {k}: " + "  ".join(f"{t}:{table[k][t]:.4f}(ref {ref[k].get(t, float('nan')):.4f})" for t in taus[k]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--merge", action="store_true"); a = ap.parse_args()
    merge() if a.merge else measure()
