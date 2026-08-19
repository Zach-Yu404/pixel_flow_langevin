#!/usr/bin/env python
"""Driver for the Algorithm-2 measurement-alignment experiments.

Nothing outside test/ is modified. The sampler, the metrics, the config key
contract and the PixelFlow scheduler are all the existing ones; this driver
only (a) swaps in the baseline-aligned measurement via overrides.apply() and
(b) writes derived configs that satisfy main.py's exact-key schema.

    PYTHONHASHSEED=0 python run_experiments.py --mode regression   # CPU guard
    PYTHONHASHSEED=0 python run_experiments.py --mode verify       # CPU V1-V9
    PYTHONHASHSEED=0 python run_experiments.py --mode selftest     # CPU S1/S2
    PYTHONHASHSEED=0 python run_experiments.py --mode full_ip      # 5 tasks x 3 images
    PYTHONHASHSEED=0 python run_experiments.py --mode ridge_sweep  # eps dependence

ridge_sweep fans out one subprocess per (ridge_rel, seed) and retries: the ceph
mount intermittently raises OSError 121 during import-time directory scans, and
a run's success is judged by its own output CSV existing, never by scraping a
shared log (a previous run's success line is easy to mistake for your own).
"""

import argparse
import copy
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A2 = os.path.dirname(HERE)
ORIG_CWD = os.getcwd()
if A2 not in sys.path:
    sys.path.insert(0, A2)

import overrides                                               # noqa: E402
import main as alg2                                            # noqa: E402
import torch                                                   # noqa: E402

TEST_CFG = os.path.join(HERE, "config.json")
BASE_CFG = os.path.join(A2, "config.json")
DERIVED = os.path.join(HERE, "_derived")


def load_cfgs():
    with open(TEST_CFG) as f:
        test = json.load(f)
    with open(BASE_CFG) as f:
        base = json.load(f)
    return test, base


def derive(base, test, mode, out, *, seed=None, ridge_rel=None, image=None,
           self_test=False):
    """A config that main.py's contract accepts: same keys, different values."""
    cfg = copy.deepcopy(base)
    cfg.pop("_comment", None)
    cfg["mode"] = mode
    cfg["images"] = list(test["images"])
    cfg["traj_image"] = test["traj_image"]
    if seed is not None:
        cfg["algorithm"]["seed"] = int(seed)
    if ridge_rel is not None:
        cfg["algorithm"]["ridge_rel"] = float(ridge_rel)
    if "out" in cfg[mode]:                  # verify has no out (schema: res only)
        cfg[mode]["out"] = out
    if mode == "onestep":
        cfg["onestep"]["self_test"] = bool(self_test)
    if mode == "full_ip":
        cfg["full_ip"]["smoke"] = False
    if mode == "debug_box":
        cfg["debug_box"].update(image=image or test["ridge_sweep"]["image"],
                                trw_values=None, anchors=None,
                                x0_steps=[1], gamma2_scales=[1.0], h0=[0.1])
    os.makedirs(DERIVED, exist_ok=True)
    path = os.path.join(DERIVED, f"{mode}_{os.path.basename(out)}.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1)
    return path


def run_main(cfg_path, mode):
    argv = sys.argv
    sys.argv = ["main.py", "--config", cfg_path, "--mode", mode]
    try:
        return alg2.main()
    finally:
        sys.argv = argv


# ── regression guard ────────────────────────────────────────────────────────
def mode_regression(test, base):
    """With the overrides live, the three noisy tasks must be untouched and the
    two inpainting tasks must change in exactly the documented way."""
    from demo_runner import load_demo_images
    orig = overrides._ORIG_BUILD
    dev = "cpu"
    demos = {d["short_name"]: d for d in
             load_demo_images(resolution=256, demo_dir=_demo_dir(base))}
    demo = demos[test["traj_image"]]
    m = test["measurement"]
    overrides.apply(m["inpaint_noise"], m["box_center"], m["noise_seed"])

    ok = True
    for task, spec in base["tasks_setup"].items():
        sn = float(spec["sigma_n"])
        o_op, o_mask, o_y, *_ = orig(task, spec["operator"], demo, sn, 256, dev)
        n_op, n_mask, n_y, *_ = overrides.patched_build_setup_and_measurement(
            task, spec["operator"], demo, sn, 256, dev)
        same_mask = torch.equal(o_mask, n_mask)
        same_y = o_y.shape == n_y.shape and torch.equal(o_y, n_y)
        if task in overrides.INPAINT_TASKS:
            want_mask_same = not (m["box_center"] and task == "box_inpainting")
            good = (same_mask == want_mask_same) and not same_y
            delta = float((n_y - o_y).std())
            print(f"  {task:18s} mask_same={same_mask} (want {want_mask_same})  "
                  f"y_changed={not same_y}  std(y_new-y_old)={delta:.4f} "
                  f"(sigma_n={sn})  {'OK' if good else '** FAIL **'}")
        else:
            good = same_mask and same_y
            print(f"  {task:18s} untouched: mask_same={same_mask} y_same={same_y}"
                  f"  {'OK' if good else '** FAIL **'}")
        ok &= good
    print("REGRESSION GUARD:", "PASS" if ok else "** FAIL **")
    return 0 if ok else 1


def _demo_dir(base):
    return alg2.resolve_path(base["paths"]["demo_dir"]
                             .replace("{IP_PACKAGE}", alg2.base.IP_PACKAGE)
                             .replace("{HERE}", A2))


# ── eps-dependence sweep ────────────────────────────────────────────────────
def mode_ridge_sweep(test, base):
    sw = test["ridge_sweep"]
    root = os.path.join(test["out_root"], "ridge_sweep")
    todo = [(r, s) for r in sw["ridge_rel"] for s in sw["seeds"]]
    print(f"[sweep] {len(todo)} runs = {len(sw['ridge_rel'])} ridge x "
          f"{len(sw['seeds'])} seeds", flush=True)
    done, failed = [], []
    for ridge, seed in todo:
        tag = f"r{ridge:g}_s{seed}"
        out = f"{root}/{tag}"
        csv = os.path.join(A2, out, "final_metrics.csv")
        if os.path.exists(csv):
            os.remove(csv)
        for attempt in range(1, 7):
            cfg_path = derive(base, test, "debug_box", out,
                              seed=seed, ridge_rel=ridge)
            env = dict(os.environ, PYTHONHASHSEED="0")
            subprocess.run([sys.executable, __file__, "--mode", "_one",
                            "--config-path", cfg_path], env=env, cwd=HERE)
            if os.path.exists(csv):        # this run's own artefact, not a log line
                done.append((ridge, seed, csv))
                break
            print(f"  retry {tag} (attempt {attempt} produced no CSV)", flush=True)
        else:
            failed.append(tag)
    print(f"[sweep] ok={len(done)} failed={failed}", flush=True)
    _summarise_sweep(done, root)
    return 1 if failed else 0


def _summarise_sweep(done, root):
    import csv as csvmod
    import statistics as st
    by_ridge = {}
    for ridge, seed, path in done:
        with open(path) as f:
            row = list(csvmod.DictReader(f))[0]
        by_ridge.setdefault(ridge, []).append((seed, float(row["post_hole"]),
                                               float(row["post_obs"])))
    lines = ["ridge_rel,n,hole_mean,hole_sd,hole_spread_pct,obs_mean"]
    print(f"\n{'ridge_rel':>10} {'n':>2} {'hole mean':>10} {'sd':>8} "
          f"{'spread%':>8} {'obs mean':>9}")
    for ridge in sorted(by_ridge):
        hs = [h for _, h, _ in by_ridge[ridge]]
        os_ = [o for _, _, o in by_ridge[ridge]]
        sd = st.stdev(hs) if len(hs) > 1 else 0.0
        spread = (max(hs) - min(hs)) / st.mean(hs) * 100 if len(hs) > 1 else 0.0
        print(f"{ridge:>10.0e} {len(hs):>2} {st.mean(hs):>10.4f} {sd:>8.4f} "
              f"{spread:>8.1f} {st.mean(os_):>9.5f}")
        lines.append(f"{ridge:g},{len(hs)},{st.mean(hs):.6f},{sd:.6f},"
                     f"{spread:.2f},{st.mean(os_):.6f}")
    out = os.path.join(A2, root, "eps_dependence.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[sweep] -> {out}")


# ── entry ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True,
                    choices=["regression", "verify", "selftest", "full_ip",
                             "ridge_sweep", "_one"])
    ap.add_argument("--config-path", default=None,
                    help="internal: derived config for a single _one run")
    cli = ap.parse_args()
    test, base = load_cfgs()
    m = test["measurement"]

    if cli.mode == "_one":                       # one child of the sweep
        overrides.apply(m["inpaint_noise"], m["box_center"], m["noise_seed"])
        return run_main(cli.config_path, "debug_box")

    if cli.mode == "regression":
        return mode_regression(test, base)

    eff = overrides.apply(m["inpaint_noise"], m["box_center"], m["noise_seed"])
    print(f"[test] measurement override in effect: {eff}", flush=True)

    if cli.mode == "ridge_sweep":
        return mode_ridge_sweep(test, base)
    if cli.mode == "verify":
        return run_main(derive(base, test, "verify",
                               f"{test['out_root']}/verify"), "verify")
    if cli.mode == "selftest":
        return run_main(derive(base, test, "onestep",
                               f"{test['out_root']}/selftest", self_test=True),
                        "onestep")
    if cli.mode == "full_ip":
        return run_main(derive(base, test, "full_ip",
                               f"{test['out_root']}/full_ip"), "full_ip")


if __name__ == "__main__":
    raise SystemExit(main())
