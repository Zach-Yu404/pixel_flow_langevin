"""Block-2 Langevin probe x noise conditions (user 2026-09-03: "no noise，和xi_0 = 0的设置，结合这几个h_0的选择测试一下").
Conditions: no_noise (xi_y=xi_h=xi_s=0 whole process), xi0_zero (Block-2 xi_0=0 whole process), all_zero (both; extra
reference). Arms: baseline (exact draw (23)) | h0 in {1.0,0.5,0.1,5e-2,1e-2,5e-3,1e-5}. junco box, seeds 42-44, [2,2,1,1],
default S. Outputs results/alg4_block2_langevin/noise_conditions/<cond>/<arm>/junco_s<seed>/..."""
import os, sys, json, time, math, csv, numpy as np, torch
ALG2 = "/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2"; sys.path.insert(0, ALG2); os.chdir(ALG2)
import main4, utils, s_prior_methods as SP
from PIL import Image
OUT = os.path.join(ALG2, "results", "alg4_block2_langevin", "noise_conditions"); os.makedirs(OUT, exist_ok=True)
SHARD, NSHARD = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
device = "cuda:0"
ARMS = [("baseline", None), ("h1.0", 1.0), ("h0.5", 0.5), ("h0.1", 0.1), ("h5e-2", 5e-2), ("h1e-2", 1e-2), ("h5e-3", 5e-3), ("h1e-5", 1e-5)]
CONDS = {"no_noise": dict(diag_noise_off=["xi_y", "xi_h", "xi_s"], diag_noise_off_from_stage=0),
         "xi0_zero": dict(diag_noise_off=["xi_0"], diag_noise_off_from_stage=0),
         "all_zero": dict(diag_noise_off=["xi_y", "xi_h", "xi_s", "xi_0"], diag_noise_off_from_stage=0)}
CELLS = [(c, a, sd) for sd in (42, 43, 44) for c in CONDS for a in ARMS]
CELLS = [x for i, x in enumerate(CELLS) if i % NSHARD == SHARD]
SP._init_globals()
config, model, gamma2_tab, _ = SP._load_sampling(device); K = int(config.scheduler.num_stages)
s2_fn = main4.default_s2_fn(K)
def to_img(x):
    x = x[0] if x.dim() == 4 else x
    return ((x.clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 0).cpu().numpy()
S = None
for att in range(8):
    try:
        S = main4._task_setup("box_inpainting", "junco", device, config); break
    except OSError as e:
        print(f"[setup EIO retry {att}] {e}", flush=True); time.sleep(20)
for cond, (name, h0), seed in CELLS:
    d = os.path.join(OUT, cond, name, f"junco_s{seed}"); fin = os.path.join(d, "final.json")
    if os.path.exists(fin):
        print(f"[skip] {cond}/{name}/s{seed}", flush=True); continue
    kw = dict(S["kw"]); kw["num_langevin"] = [2, 2, 1, 1]; kw.update(CONDS[cond])
    if h0 is not None:
        kw["diag_block2_langevin"] = h0
    S2 = dict(S); S2["kw"] = kw
    t0 = time.time(); utils.NFE["n"] = 0
    x1, rows, traj = main4._run_once(model, config, S2, device, s2_fn=s2_fn, gamma2_tab=gamma2_tab, seed=seed, record_trajectory=True)
    gt, hole = S["gt"], S["hole"]
    mse_full = float(((x1 - gt) ** 2).mean()); mse_hole = utils.mse_masked(x1, gt, hole); mse_obs = utils.mse_masked(x1, gt, 1.0 - hole)
    psnr = 10 * math.log10(4.0 / mse_full)
    os.makedirs(d, exist_ok=True)
    json.dump(dict(cond=cond, arm=name, h0=h0, image="junco", seed=seed, mse_full=mse_full, mse_hole=mse_hole, mse_obs=mse_obs,
                   psnr_range2=psnr, secs=time.time() - t0, nfe=utils.NFE["n"],
                   cg_bad=int(sum(1 for r in rows if not r.get("blk1_cg_converged", 1))),
                   x0_rms_last=float(rows[-1].get("x0_rms", float("nan")))), open(fin, "w"), indent=1)
    with open(os.path.join(d, "trajectory_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    Image.fromarray(to_img(x1)).save(os.path.join(d, "x1_final.png"))
    fr = np.stack([to_img(torch.nn.functional.interpolate(t[1][None], size=(256, 256), mode="bilinear", align_corners=False)) for t in traj])
    np.save(os.path.join(d, "traj_x1.npy"), fr[[0, 5, 10, 15, 20, 25, 30, 35, len(fr) - 1]])
    print(f"[cell] {cond}/{name}/s{seed}: hole={mse_hole:.4f} full={mse_full:.4f} psnr={psnr:.2f} x0rms={rows[-1].get('x0_rms', float('nan')):.3f} [{time.time()-t0:.0f}s]", flush=True)
print("COND SHARD DONE", flush=True)
