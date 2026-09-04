"""Block-2 Langevin probe (user 2026-09-03): Alg-2 l.11+l.15-17 on x0 in place of the exact draw (23).
Arms: baseline (exact draw) | h0 in {1.0, 0.5, 0.1, 5e-2, 1e-2, 5e-3, 1e-5}.
Cells: box_inpainting x {junco + 6 grid images} x seed 42 (+ seeds 43,44 on junco), [2,2,1,1], default S
(spectral_class). Outputs results/alg4_block2_langevin/<arm>/<image>_s<seed>/{final.json, trajectory_metrics.csv,
x1_final.png, traj.npy} and per-arm trajectory montage; aggregation in a separate step."""
import os, sys, json, time, math, numpy as np, torch
ALG2 = "/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2"; sys.path.insert(0, ALG2); os.chdir(ALG2)
import main4, utils, s_prior_methods as SP
from PIL import Image
OUT = os.path.join(ALG2, "results", "alg4_block2_langevin"); os.makedirs(OUT, exist_ok=True)
SHARD, NSHARD = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
device = "cuda:0"
ARMS = [("baseline", None), ("h0.1", 0.1), ("h0.5", 0.5), ("h1.0", 1.0),
        ("h5e-2", 5e-2), ("h1e-2", 1e-2), ("h5e-3", 5e-3), ("h1e-5", 1e-5)]   # user: + small h0; flip arms withdrawn
IMAGES = ["junco"]   # user 2026-09-03: junco box only
CELLS = [(a, "junco", sd) for sd in (42, 43, 44) for a in ARMS]
CELLS = [c for i, c in enumerate(CELLS) if i % NSHARD == SHARD]
SP._init_globals()
config, model, gamma2_tab, _ = SP._load_sampling(device); K = int(config.scheduler.num_stages)
s2_fn = main4.default_s2_fn(K)
def to_img(x):  # [-1,1] tensor (1,3,H,W) or (3,H,W) -> uint8 HxWx3
    x = x[0] if x.dim() == 4 else x
    return ((x.clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 0).cpu().numpy()
setups = {}
for (name, h0), image, seed in CELLS:
    d = os.path.join(OUT, name, f"{image}_s{seed}"); fin = os.path.join(d, "final.json")
    if os.path.exists(fin):
        print(f"[skip] {name}/{image}_s{seed}", flush=True); continue
    if image not in setups:
        for att in range(6):
            try:
                setups[image] = main4._task_setup("box_inpainting", image, device, config); break
            except OSError as e:
                print(f"[setup EIO retry {att}] {e}", flush=True); time.sleep(20)
    S = setups[image]
    kw = dict(S["kw"]); kw["num_langevin"] = [2, 2, 1, 1]
    if h0 is not None:
        kw["diag_block2_langevin"] = h0
    S2 = dict(S); S2["kw"] = kw
    t0 = time.time(); utils.NFE["n"] = 0
    x1, rows, traj = main4._run_once(model, config, S2, device, s2_fn=s2_fn, gamma2_tab=gamma2_tab,
                                     seed=seed, record_trajectory=True)
    gt, hole = S["gt"], S["hole"]
    mse_full = float(((x1 - gt) ** 2).mean()); mse_hole = utils.mse_masked(x1, gt, hole)
    mse_obs = utils.mse_masked(x1, gt, 1.0 - hole)
    psnr = 10 * math.log10(4.0 / mse_full)
    os.makedirs(d, exist_ok=True)
    json.dump(dict(arm=name, h0=h0, image=image, seed=seed, mse_full=mse_full, mse_hole=mse_hole,
                   mse_obs=mse_obs, psnr_range2=psnr, secs=time.time() - t0, nfe=utils.NFE["n"],
                   cg_bad=int(sum(1 for r in rows if not r.get("blk1_cg_converged", 1))),
                   x0_rms_last=float(rows[-1].get("x0_rms", float("nan")))), open(fin, "w"), indent=1)
    import csv
    with open(os.path.join(d, "trajectory_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    Image.fromarray(to_img(x1)).save(os.path.join(d, "x1_final.png"))
    # trajectory: x1 per frame, upsampled to 256 for the montage
    fr = np.stack([to_img(torch.nn.functional.interpolate(t[1][None], size=(256, 256), mode="bilinear", align_corners=False))
                   for t in traj])
    np.save(os.path.join(d, "traj_x1.npy"), fr[[0, 5, 10, 15, 20, 25, 30, 35, len(fr) - 1]])
    if not os.path.exists(os.path.join(OUT, f"{image}_gt.png")):
        Image.fromarray(to_img(gt)).save(os.path.join(OUT, f"{image}_gt.png"))
        y_img = gt * (1 - hole) + hole * 0  # masked view for reference
        Image.fromarray(to_img(y_img)).save(os.path.join(OUT, f"{image}_masked.png"))
    print(f"[cell] {name}/{image}_s{seed}: hole={mse_hole:.4f} full={mse_full:.4f} psnr={psnr:.2f} "
          f"x0rms={rows[-1].get('x0_rms', float('nan')):.3f} [{time.time()-t0:.0f}s]", flush=True)
print("BLOCK2 SHARD DONE", flush=True)
