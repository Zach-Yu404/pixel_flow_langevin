#!/usr/bin/env python
"""s_stats/test: 4 val-set S constructions x 3 S_it schedules x all_img_tests
grid (5 tasks x 6 images), estimators {single_avg10(allnoise), MMSE5, MMSE10,
no_noise(=3xi0 full, seed42)}, metrics MSE/PSNR/SSIM/LPIPS(piq+alex).

Shard via env SHARD="pooled_all,spectral_all" etc. Resume via CSV rows.
"""
import os, sys, json, csv, time
import numpy as np, torch
A = "/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2"
sys.path.insert(0, A); os.chdir(A)
sys.path.insert(0, "/CBIG-Standard-ECE/Zach/MSFlow/PixelFlow/IP_package/rerun_imageNet")
import s_prior_methods as SP, main4, utils
import metrics as MET
from metrics import lpips_alex
ST = A + "/s_stats"; O = ST + "/test"; os.makedirs(O, exist_ok=True)
CSV = os.path.join(O, "s4_results.csv")
HDR = ("S_type,schedule,task,image,estimator,mse_full,mse_hole,psnr,ssim,"
       "lpips_piq,lpips_alex\n")
if not os.path.exists(CSV):
    open(CSV, "w").write(HDR)
done = set()
for r in csv.DictReader(open(CSV)):
    done.add((r["S_type"], r["schedule"], r["task"], r["image"]))
SHARD = [s for s in os.environ.get("SHARD", "").split(",") if s]
SCHEDS = {"2222": 2, "2211": [2, 2, 1, 1], "2221": [2, 2, 2, 1]}
TASKS = ["box_inpainting", "random_inpainting", "gaussian_blur",
         "motion_blur", "superresolution"]
IMAGES = ["breastplate_armor", "crane_structure", "ibex_horns",
          "lakeside_beach", "sea_anemone", "shetland_sheepdog"]
SEEDS = list(range(42, 52))
SP._init_globals(); device = "cuda:0"
config, model, gamma2_tab, _ = SP._load_sampling(device)
K = int(config.scheduler.num_stages)
POOL_ALL = json.load(open(ST + "/s_pooled_statistics_all.json"))["pooled_s2"]
POOL_LAB = json.load(open(ST + "/s_pooled_statistics_labelled.json"))["classes"]
SPEC_ALL = np.load(ST + "/spectral_power_all.npz")
SPEC_LAB = np.load(ST + "/spectral_power_labelled.npz")
syn_order = [l.strip().split(" ", 1)[0] for l in open(
    "/CBIG-Standard-ECE/Zach_dataset/Zach_dataset/imageNet256/LOC_synset_mapping.txt")]
_spec_cache = {}
def make_s2fn(stype, synset):
    if stype == "pooled_all":
        return lambda k, sig: float(POOL_ALL[str(k)])
    if stype == "pooled_class":
        t = POOL_LAB[synset]["pooled_s2"]
        return lambda k, sig: float(t[str(k)])
    if stype == "spectral_all":
        key = ("all",)
        if key not in _spec_cache:
            _spec_cache[key] = {k: utils.SpectralSOp(
                torch.from_numpy(SPEC_ALL[f"stage{k}"])) for k in range(K)}
        ops = _spec_cache[key]
        return lambda k, sig: ops[k]
    if stype == "spectral_class":
        key = ("c", synset)
        if key not in _spec_cache:
            _spec_cache[key] = {k: utils.SpectralSOp(
                torch.from_numpy(SPEC_LAB[f"{synset}_stage{k}"])) for k in range(K)}
        ops = _spec_cache[key]
        return lambda k, sig: ops[k]
    raise KeyError(stype)
def to01(x): return (x.clamp(-1, 1) + 1) / 2
def mets(x, gt, hole):
    r01, g01 = to01(x), to01(gt)
    mse = float(((x - gt) ** 2).mean())
    mh = float((((x - gt) ** 2) * hole).sum() / (hole.sum() * 3)) \
        if hole is not None and float(hole.sum()) > 0 else float("nan")
    return (mse, mh, MET.psnr(r01, g01), MET.ssim(r01, g01),
            MET.lpips(r01, g01, device), lpips_alex(r01, g01, device))
for stype in SHARD:
    for sched_tag, sched in SCHEDS.items():
        for task in TASKS:
            for image in IMAGES:
                if (stype, sched_tag, task, image) in done:
                    print(f"[s4] skip {stype}/{sched_tag}/{task}/{image}", flush=True)
                    continue
                # atomic per-cell claim so several GPU processes work-steal
                # without duplicating cells; stale claims (>20 min, no CSV row)
                # are reclaimed.
                cdir = os.path.join(O, "_claims"); os.makedirs(cdir, exist_ok=True)
                cpath = os.path.join(cdir, f"{stype}__{sched_tag}__{task}__{image}")
                try:
                    fd = os.open(cpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(os.getpid()).encode()); os.close(fd)
                except FileExistsError:
                    if time.time() - os.path.getmtime(cpath) > 1200:
                        try:
                            os.remove(cpath)
                            fd = os.open(cpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                            os.write(fd, str(os.getpid()).encode()); os.close(fd)
                        except (FileExistsError, FileNotFoundError):
                            continue
                    else:
                        continue
                S = None
                for _att in range(8):
                    try:
                        S = main4._task_setup(task, image, device, config)
                        break
                    except OSError as e:
                        print(f"[s4] setup EIO retry {_att}: {e}", flush=True)
                        time.sleep(20)
                if S is None:
                    raise OSError("setup failed after 8 retries")
                synset = syn_order[int(S["demo"]["class_idx"])]
                s2fn = make_s2fn(stype, synset)
                gt = S["gt"]
                hole = S["hole"].to(device).float() if S["hole"] is not None else None
                S["kw"]["num_langevin"] = sched
                S["kw"].pop("diag_noise_off", None)
                S["kw"].pop("diag_noise_off_from_stage", None)
                t0 = time.time()
                finals = []
                for seed in SEEDS:
                    x1, _, _ = main4._run_once(model, config, S, device, s2_fn=s2fn,
                                               gamma2_tab=gamma2_tab, seed=seed)
                    finals.append(x1.detach())
                S["kw"]["diag_noise_off"] = ["xi_y", "xi_h", "xi_s"]
                S["kw"]["diag_noise_off_from_stage"] = 0
                xnn, _, _ = main4._run_once(model, config, S, device, s2_fn=s2fn,
                                            gamma2_tab=gamma2_tab, seed=42)
                S["kw"].pop("diag_noise_off"); S["kw"].pop("diag_noise_off_from_stage")
                singles = [mets(f, gt, hole) for f in finals]
                rows = {
                    "single_avg10": tuple(float(np.nanmean([s[i] for s in singles]))
                                          for i in range(6)),
                    "MMSE5": mets(torch.stack(finals[:5]).mean(0), gt, hole),
                    "MMSE10": mets(torch.stack(finals).mean(0), gt, hole),
                    "no_noise": mets(xnn, gt, hole),
                }
                with open(CSV, "a") as f:
                    for est, m in rows.items():
                        f.write(f"{stype},{sched_tag},{task},{image},{est},"
                                + ",".join(f"{v:.4f}" for v in m) + "\n")
                print(f"[s4] {stype}/{sched_tag}/{task}/{image} "
                      f"single={rows['single_avg10'][2]:.2f} "
                      f"mmse10={rows['MMSE10'][2]:.2f} "
                      f"nonoise={rows['no_noise'][2]:.2f}dB "
                      f"[{int(time.time()-t0)}s]", flush=True)
print("S4 SHARD DONE", flush=True)
