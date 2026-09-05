"""gamma^2-table comparison, estimator view (user 2026-09-05: "尝试预测full noise, no xi_0, no noise, full noise MMSE5,
full noise MMSE10"). Arms: baseline (7-image gamma2_meas_alg4.json) / gamma2_all / gamma2_labelled (own class), under the
default S (spectral_class), [2,2,1,1] forced. Per (arm, task, image): 10 full-noise samples (seeds 42-51) -> single_avg10
(metrics averaged over seeds), MMSE5 (pixel mean of first 5), MMSE10 (mean of 10); no_xi0 (Block-2 xi_0=0 whole process,
seed 42); no_noise (xi_y=xi_h=xi_s=0 whole process, seed 42). Metrics as run_s4_test.py. Work-stealing via claims."""
import os, sys, json, csv, time
import numpy as np, torch
A = "/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2"; sys.path.insert(0, A); os.chdir(A)
sys.path.insert(0, "/CBIG-Standard-ECE/Zach/MSFlow/PixelFlow/IP_package/rerun_imageNet")
import s_prior_methods as SP, main4, utils
import metrics as MET
from metrics import lpips_alex
OUT = os.path.join(A, "gamma2_stats", "test"); CLAIMS = os.path.join(OUT, "_claims_est"); CSV = os.path.join(OUT, "results_est.csv")
device = "cuda:0"
TASKS = ["box_inpainting", "random_inpainting", "gaussian_blur", "motion_blur", "superresolution"]
IMAGES = ["junco", "breastplate_armor", "crane_structure", "ibex_horns", "lakeside_beach", "sea_anemone", "shetland_sheepdog"]
SEEDS = list(range(42, 52)); ARMS = ["baseline", "gamma2_all", "gamma2_labelled"]
EST = ["single_avg10", "MMSE5", "MMSE10", "no_xi0", "no_noise"]
G_BASE = json.load(open(os.path.join(A, "gamma2_meas_alg4.json")))["table"]
G_ALL = json.load(open(os.path.join(A, "gamma2_stats", "gamma2_all.json")))["table"]
G_LAB = json.load(open(os.path.join(A, "gamma2_stats", "gamma2_labelled.json")))["classes"]
SYN = [l.split()[0] for l in open("/CBIG-Standard-ECE/Zach_dataset/Zach_dataset/imageNet256/LOC_synset_mapping.txt")]
FIELDS = ["arm", "task", "image", "estimator", "class_idx", "synset", "mse_full", "mse_hole", "psnr", "ssim", "lpips_piq", "lpips_alex", "secs"]
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
        if time.time() - os.path.getmtime(p) > 2400:
            try:
                os.remove(p); fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd); return True
            except (FileExistsError, FileNotFoundError): return False
        return False
def done_cells():
    if not os.path.exists(CSV): return set()
    out = set()
    for r in csv.DictReader(open(CSV)):
        try: out.add((r["arm"], r["task"], r["image"]))
        except (KeyError, TypeError): continue
    return out
def to01(x): return (x.clamp(-1, 1) + 1) / 2
def mets(x, gt, hole):
    r01, g01 = to01(x), to01(gt)
    mse = float(((x - gt) ** 2).mean())
    mh = float((((x - gt) ** 2) * hole).sum() / (hole.sum() * 3)) if hole is not None and float(hole.sum()) > 0 else float("nan")
    return dict(mse_full=mse, mse_hole=mh, psnr=MET.psnr(r01, g01), ssim=MET.ssim(r01, g01),
                lpips_piq=MET.lpips(r01, g01, device), lpips_alex=lpips_alex(r01, g01, device))
SP._init_globals()
config, model, _g2j, _ = SP._load_sampling(device); K = int(config.scheduler.num_stages)
s2_fn = main4.default_s2_fn(K)
print(f"[setup] S={main4.S_DESC}; arms={ARMS}; estimators={EST}; seeds {SEEDS[0]}-{SEEDS[-1]}; [2,2,1,1] forced", flush=True)
def run(S, g2, seed, noise_off=None):
    S2 = dict(S); kw = dict(S["kw"]); kw["num_langevin"] = [2, 2, 1, 1]
    kw.pop("diag_noise_off", None); kw.pop("diag_noise_off_from_stage", None)
    if noise_off: kw["diag_noise_off"] = list(noise_off); kw["diag_noise_off_from_stage"] = 0
    S2["kw"] = kw
    x1, _, _ = main4._run_once(model, config, S2, device, s2_fn=s2_fn, gamma2_tab=g2, seed=seed)
    return x1.detach()
setups = {}
for task in TASKS:
    for image in IMAGES:
        for arm in ARMS:
            key = f"{arm}__{task}__{image}"
            if (arm, task, image) in retry(done_cells) or not retry(lambda: claim(key)): continue
            if (task, image) not in setups:
                setups[(task, image)] = retry(lambda: main4._task_setup(task, image, device, config), tries=8, delay=20)
            S = setups[(task, image)]; cidx = int(S["demo"]["class_idx"]); syn = SYN[cidx]
            assert int(G_LAB[syn]["class_idx"]) == cidx
            g2 = {"baseline": G_BASE, "gamma2_all": G_ALL, "gamma2_labelled": G_LAB[syn]["table"]}[arm]
            gt = S["gt"]; hole = S["hole"] if "inpaint" in task else None
            t0 = time.time()
            finals = [run(S, g2, sd) for sd in SEEDS]
            x_noxi0 = run(S, g2, 42, noise_off=["xi_0"])
            x_nonoise = run(S, g2, 42, noise_off=["xi_y", "xi_h", "xi_s"])
            singles = [mets(f, gt, hole) for f in finals]
            rows = {"single_avg10": {k: float(np.nanmean([s[k] for s in singles])) for k in singles[0]},
                    "MMSE5": mets(torch.stack(finals[:5]).mean(0), gt, hole),
                    "MMSE10": mets(torch.stack(finals).mean(0), gt, hole),
                    "no_xi0": mets(x_noxi0, gt, hole), "no_noise": mets(x_nonoise, gt, hole)}
            secs = round(time.time() - t0, 1)
            def _append():
                with open(CSV, "a", newline="") as f:
                    f.seek(0, 2); w = csv.DictWriter(f, fieldnames=FIELDS)
                    if f.tell() == 0: w.writeheader()
                    for est, m in rows.items():
                        w.writerow(dict(arm=arm, task=task, image=image, estimator=est, class_idx=cidx, synset=syn, secs=secs, **m))
            retry(_append)
            print(f"[cell] {arm}/{task}/{image}: single={rows['single_avg10']['psnr']:.2f} mmse10={rows['MMSE10']['psnr']:.2f} "
                  f"noxi0={rows['no_xi0']['psnr']:.2f} nonoise={rows['no_noise']['psnr']:.2f} dB [{secs}s]", flush=True)
print("G2EST SHARD DONE", flush=True)
