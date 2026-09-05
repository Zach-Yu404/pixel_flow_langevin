"""gamma^2-table comparison under the default S (spectral_class): baseline (gamma2_meas_alg4.json, 7 demo images)
vs gamma2_all (val 50k) vs gamma2_labelled (val, the image's own class). Grid: 5 tasks x 7 images
(6 grid images + junco) x seeds 42-44, [2,2,1,1] forced (as all_img_tests / S4). Metrics: MSE full/hole, PSNR, SSIM,
LPIPS (piq-VGG + official alex) via rerun_imageNet/metrics.py. Work-stealing over cells through atomic claims."""
import os, sys, json, csv, time
import numpy as np, torch
A = "/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2"; sys.path.insert(0, A); os.chdir(A)
sys.path.insert(0, "/CBIG-Standard-ECE/Zach/MSFlow/PixelFlow/IP_package/rerun_imageNet")
import s_prior_methods as SP, main4, utils
import metrics as MET
from metrics import lpips_alex
OUT = os.path.join(A, "gamma2_stats", "test"); CLAIMS = os.path.join(OUT, "_claims"); CSV = os.path.join(OUT, "results.csv")
device = "cuda:0"
TASKS = ["box_inpainting", "random_inpainting", "gaussian_blur", "motion_blur", "superresolution"]
IMAGES = ["junco", "breastplate_armor", "crane_structure", "ibex_horns", "lakeside_beach", "sea_anemone", "shetland_sheepdog"]
SEEDS = (42, 43, 44); ARMS = ["baseline", "gamma2_all", "gamma2_labelled", "baseline_rerun"]   # rerun = same table+seed: GPU noise floor
G_BASE = json.load(open(os.path.join(A, "gamma2_meas_alg4.json")))["table"]
G_ALL = json.load(open(os.path.join(A, "gamma2_stats", "gamma2_all.json")))["table"]
G_LAB = json.load(open(os.path.join(A, "gamma2_stats", "gamma2_labelled.json")))["classes"]
SYN = [l.split()[0] for l in open("/CBIG-Standard-ECE/Zach_dataset/Zach_dataset/imageNet256/LOC_synset_mapping.txt")]
def retry(fn, tries=12, delay=10):
    for att in range(tries):
        try: return fn()
        except OSError as e:
            if att == tries - 1: raise
            print(f"[eio-retry {att}] {e}", flush=True); time.sleep(delay)
def claim(key):
    os.makedirs(CLAIMS, exist_ok=True); p = os.path.join(CLAIMS, key)
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd); return True
    except FileExistsError:
        if time.time() - os.path.getmtime(p) > 1800:
            try:
                os.remove(p); fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd); return True
            except (FileExistsError, FileNotFoundError): return False
        return False
def done_keys():
    if not os.path.exists(CSV): return set()
    out = set()
    for r in csv.DictReader(open(CSV)):
        try: out.add((r["arm"], r["task"], r["image"], int(r["seed"])))
        except (ValueError, TypeError, KeyError): continue      # tolerate a duplicated header / partial line
    return out
def to01(x): return (x.clamp(-1, 1) + 1) / 2
def mets(x, gt, hole):
    r01, g01 = to01(x), to01(gt)
    mse = float(((x - gt) ** 2).mean())
    mh = float((((x - gt) ** 2) * hole).sum() / (hole.sum() * 3)) if hole is not None and float(hole.sum()) > 0 else float("nan")
    return dict(mse_full=mse, mse_hole=mh, psnr=MET.psnr(r01, g01), ssim=MET.ssim(r01, g01),
                lpips_piq=MET.lpips(r01, g01, device), lpips_alex=lpips_alex(r01, g01, device))
SP._init_globals()
config, model, _g2_junco, _ = SP._load_sampling(device); K = int(config.scheduler.num_stages)
s2_fn = main4.default_s2_fn(K)
print(f"[setup] S={main4.S_DESC}; arms={ARMS}; schedule [2,2,1,1] forced", flush=True)
FIELDS = ["arm", "task", "image", "seed", "class_idx", "synset", "mse_full", "mse_hole", "psnr", "ssim", "lpips_piq", "lpips_alex", "cg_bad", "secs"]
setups = {}
for task in TASKS:
    for image in IMAGES:
        for arm in ARMS:
            for seed in SEEDS:
                key = f"{arm}__{task}__{image}__{seed}"
                if (arm, task, image, seed) in retry(done_keys) or not retry(lambda: claim(key)): continue
                if (task, image) not in setups:
                    setups[(task, image)] = retry(lambda: main4._task_setup(task, image, device, config), tries=8, delay=20)
                S = setups[(task, image)]; cidx = int(S["demo"]["class_idx"]); syn = SYN[cidx]
                assert int(G_LAB[syn]["class_idx"]) == cidx, (syn, cidx)
                g2 = {"baseline": G_BASE, "baseline_rerun": G_BASE, "gamma2_all": G_ALL, "gamma2_labelled": G_LAB[syn]["table"]}[arm]
                S2 = dict(S); kw = dict(S["kw"]); kw["num_langevin"] = [2, 2, 1, 1]; S2["kw"] = kw
                t0 = time.time()
                x1, rows, _ = main4._run_once(model, config, S2, device, s2_fn=s2_fn, gamma2_tab=g2, seed=seed)
                hole = S["hole"] if "inpaint" in task else None
                m = mets(x1, S["gt"], hole)
                row = dict(arm=arm, task=task, image=image, seed=seed, class_idx=cidx, synset=syn, **m,
                           cg_bad=int(sum(1 for r in rows if not r.get("blk1_cg_converged", 1))), secs=round(time.time() - t0, 1))
                def _append():
                    with open(CSV, "a", newline="") as f:
                        f.seek(0, 2); w = csv.DictWriter(f, fieldnames=FIELDS)
                        if f.tell() == 0: w.writeheader()          # header only when the file is empty at open time
                        w.writerow(row)
                retry(_append)
                print(f"[cell] {arm}/{task}/{image}/s{seed}: psnr={m['psnr']:.2f} ssim={m['ssim']:.3f} lpips_alex={m['lpips_alex']:.3f} [{row['secs']}s]", flush=True)
print("G2TEST SHARD DONE", flush=True)
