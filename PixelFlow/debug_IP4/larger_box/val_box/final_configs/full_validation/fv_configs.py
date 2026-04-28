"""36-config sweep generated from complete_configs/.

Reads the 9 complete-spec JSON files in complete_configs/ (which have ALL
run_ip4 kwargs explicit) and crosses with {fd=False/True} × {tr=0.0/1.0}.

Total: 9 * 2 * 2 = 36 runs.
"""
import os, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
COMPLETE = os.path.join(HERE, "complete_configs")


def _level_order(name):
    for i, lv in enumerate(["LPIPS_king", "balanced_perceptual", "pareto_dual_king"]):
        if name.startswith(lv):
            return i
    return 99


def _srs_order(name):
    for i, s in enumerate(["1e-2", "1e-3", "1e-4"]):
        if name.endswith(s):
            return i
    return 99


_files = sorted(
    glob.glob(os.path.join(COMPLETE, "*.json")),
    key=lambda f: (_level_order(os.path.basename(f).replace(".json","")),
                   _srs_order(os.path.basename(f).replace(".json",""))),
)
assert len(_files) == 9, f"expected 9 complete configs, got {len(_files)}"

BASES = []
for f in _files:
    d = json.load(open(f))
    BASES.append((d["tag"], d["kw"]))


CONFIGS = []
for tag, kw_base in BASES:
    for fd in (False, True):
        for tr in (0.0, 1.0):
            kw = dict(kw_base)
            kw["final_denoise"] = fd
            kw["terminal_replace_weight"] = tr
            name = f"{tag}__fd{int(fd)}_tr{int(tr)}"
            CONFIGS.append((name, kw))


if __name__ == "__main__":
    print(f"Total: {len(CONFIGS)}")
    print(f"\nFirst 4 (LPIPS_king_srs1e-2 × {{fd, tr}}):")
    for n, kw in CONFIGS[:4]:
        print(f"  {n}")
        print(f"    fd={kw['final_denoise']}  tr={kw['terminal_replace_weight']}  "
              f"h_eps={kw['h_epsilon']}  L={kw['num_langevin']}  srs={kw['sigma_ref_sq']}")
    print(f"\nLast 4 (pareto_dual_king_srs1e-4 × {{fd, tr}}):")
    for n, kw in CONFIGS[-4:]:
        print(f"  {n}")
        print(f"    fd={kw['final_denoise']}  tr={kw['terminal_replace_weight']}  "
              f"h_eps={kw['h_epsilon']}  L={kw['num_langevin']}  srs={kw['sigma_ref_sq']}")
