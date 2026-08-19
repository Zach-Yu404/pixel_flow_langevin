#!/usr/bin/env python
"""Summarise the Block-2 integrator A/B into results/block2_ab/summary.csv.

Four arms (ula|exponential x final-draw off|on) over the same seeds. The seed
floor on the hole metric is ~5%, so only the per-arm MEAN is interpretable and
an arm only counts as better if it clears that floor.
"""

import csv
import glob
import os
import statistics as st
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "results", "block2_ab")
ARMS = ["ula_nofinal", "exp_nofinal", "ula_final", "exp_final"]
LABEL = {"ula_nofinal": "ULA, no final draw (paper)",
         "exp_nofinal": "exponential, no final draw",
         "ula_final": "ULA + final Block-1 draw",
         "exp_final": "exponential + final Block-1 draw"}


def main():
    # The mount intermittently returns a short directory listing, which silently
    # drops arms from the table. Glob until two consecutive passes agree.
    per_arm, prev = {}, None
    for _ in range(8):
        found = {}
        for arm in ARMS:
            for path in sorted(glob.glob(os.path.join(ROOT, arm + "_s*",
                                                      "final_metrics.csv"))):
                seed = int(os.path.basename(os.path.dirname(path)).rsplit("_s", 1)[1])
                with open(path) as f:
                    row = list(csv.DictReader(f))[0]
                found.setdefault(arm, []).append(
                    (seed, float(row["post_hole"]), float(row["post_obs"])))
        key = {a: sorted(s for s, _, _ in v) for a, v in found.items()}
        if key == prev:
            per_arm = found
            break
        prev, per_arm = key, found
        time.sleep(1)

    base = per_arm.get("ula_nofinal")
    base_mean = st.mean([h for _, h, _ in base]) if base else None
    base_sd = st.stdev([h for _, h, _ in base]) if base and len(base) > 1 else None
    lines = ["arm,n,hole_mean,hole_sd,hole_spread_pct,obs_mean,vs_baseline_pct,"
             "var_ratio_vs_base,var_ratio_significant"]
    print(f"{'arm':<32} {'n':>2} {'hole mean':>10} {'sd':>7} {'spread%':>8} "
          f"{'obs mean':>9} {'vs base':>9} {'var ratio':>10}")
    for arm in ARMS:
        rows = per_arm.get(arm)
        if not rows:
            continue
        hs = [h for _, h, _ in rows]
        obs = [o for _, _, o in rows]
        sd = st.stdev(hs) if len(hs) > 1 else 0.0
        spread = (max(hs) - min(hs)) / st.mean(hs) * 100 if len(hs) > 1 else 0.0
        rel = ((st.mean(hs) - base_mean) / base_mean * 100) if base_mean else 0.0
        # Variance ratio against the baseline arm. With n=4 per arm the F(3,3)
        # 5% critical value is 9.28, so anything under that is not evidence.
        vr = (base_sd / sd) ** 2 if (base_sd and sd) else float("nan")
        sig = "yes" if vr == vr and vr > 9.28 else "no"
        print(f"{LABEL[arm]:<32} {len(hs):>2} {st.mean(hs):>10.4f} {sd:>7.4f} "
              f"{spread:>8.1f} {st.mean(obs):>9.5f} {rel:>+8.1f}% "
              f"{vr:>9.1f}x{'*' if sig == 'yes' else ' '}")
        lines.append(f"{arm},{len(hs)},{st.mean(hs):.6f},{sd:.6f},{spread:.2f},"
                     f"{st.mean(obs):.6f},{rel:+.2f},{vr:.2f},{sig}")
    out = os.path.join(ROOT, "summary.csv")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n-> {out}")
    if base_mean:
        floor = 5.0
        print(f"\nseed floor ~{floor:.0f}%: an arm only counts as a MEAN improvement "
              f"if it beats the baseline by more than that.\n"
              f"var ratio = baseline_var / arm_var; * marks > 9.28, the F(3,3) 5% "
              f"critical value for n=4 per arm (weak evidence either way at n=4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
