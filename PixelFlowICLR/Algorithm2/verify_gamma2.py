"""复算 review 必改项 1 的统计口径：γ²=γ²_meas vs γ²=0 的 score error 差异。

口径：与图一致，对每个 (stage, tau) 网格点把 7 张图的 score_err 求均值后比较；
score solve 与任务无关，因此每个 (stage, tau, image) 只计一次。
"""
import csv, collections, os

csv_path = os.path.join(os.path.dirname(__file__), 'results', 'alg2_mse.csv')
rows = list(csv.DictReader(open(csv_path)))
g = collections.defaultdict(lambda: ([], []))
maxrel_img = 0.0
arg_img = None
seen = set()
for r in rows:
    if float(r['sigma_tau']) < 0.01:
        continue
    s0 = float(r['score_err'])
    s2 = float(r['score_err_g2'])
    if s0 == 0 and s2 == 0:
        continue
    key = (r['stage'], r['tau'], r['image'])
    if key in seen:
        continue
    seen.add(key)
    a, b = g[(r['stage'], r['tau'])]
    a.append(s0)
    b.append(s2)
    if s0 > 0:
        rel = abs(s2 - s0) / s0
        if rel > maxrel_img:
            maxrel_img = rel
            arg_img = (r['stage'], r['tau'], r['image'], s0, s2)

maxrel_mean = 0.0
arg = None
for k, (a, b) in sorted(g.items()):
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    rel = abs(mb - ma) / ma
    if rel > maxrel_mean:
        maxrel_mean = rel
        arg = (k, ma, mb, len(a))

print("max mean-curve rel diff: %.4f%% at %s" % (100 * maxrel_mean, arg))
print("max per-image rel diff: %.4f%% at %s" % (100 * maxrel_img, str(arg_img)))
