"""Framingham results: discrimination/calibration by method, external stratum, and site-Mondrian conformal sets."""
import glob, json, pickle
import numpy as np
from common import evaluate, RUNS, td_auc, brier_h, ece_h
from conformal import run_stream, summarise, label, label_known, scores
from analysis import holm

METHODS = ["cox", "rsf", "gru_central", "gru_fedavg", "gru_ditto", "twin_central", "twin_fed"]
KEYS = ["auc1", "auc2", "auc3", "c", "brier2", "ece2"]


def main():
    runs = {int(f.split("_s")[-1][:-4]): pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{RUNS}/fram_s*.pkl"))}
    R = dict(main={}, ext={}, conf={}, tests={})
    V = runs[0]["V"]
    for split, suf in (("main", ""), ("ext", "|ext")):
        for m in METHODS + (["twin_fed|ext0"] if split == "ext" else []):
            key = m + suf if not m.endswith("ext0") else m
            vals = []
            for sd, r in runs.items():
                if key in r["pred"]:
                    rows, risk = r["pred"][key]
                    vals.append(evaluate(risk, V["T"][rows], V["E"][rows]))
            if vals:
                R[split][m] = {k: (float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals]))) for k in KEYS}
    # paired bootstrap on seed 0
    r0 = runs[0]["pred"]
    for split, suf in (("main", ""), ("ext", "|ext")):
        rows_t, risk_t = r0["twin_fed" + suf]
        pos_t = {r: i for i, r in enumerate(rows_t)}
        pid = V["pid"][rows_t]
        up = np.unique(pid)
        groups = {u: np.where(pid == u)[0] for u in up}
        T, E = V["T"][rows_t], V["E"][rows_t]
        res = {}
        for metric in ("auc2", "ece2"):
            res[metric] = {}
            ps = []
            for m in METHODS:
                if m == "twin_fed" or (m + suf) not in r0:
                    continue
                rows_b, risk_b = r0[m + suf]
                pb = {r: i for i, r in enumerate(rows_b)}
                rb = risk_b[np.array([pb[r] for r in rows_t])]
                rng = np.random.default_rng(1)
                diffs = []
                for _ in range(300):
                    i = np.concatenate([groups[u] for u in rng.choice(up, len(up))])
                    f = (lambda r: td_auc(r[i, 1], T[i], E[i], 2.0)) if metric == "auc2" else (lambda r: ece_h(r[i, 1], T[i], E[i], 2.0))
                    diffs.append(f(risk_t) - f(rb))
                diffs = np.array(diffs)
                p = max(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1 / 300)
                res[metric][m] = dict(diff=float(diffs.mean()), lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)), p=float(p))
                ps.append(p)
            for (m, v), a in zip(res[metric].items(), holm(ps)):
                v["p_holm"] = float(a)
        R["tests"][split] = res
    # conformal (8-year = 2 units), site-Mondrian label-conditional static sets for twin and best GRU
    role = runs[0]["role"]
    for m in ("twin_fed", "gru_fedavg", "cox"):
        rows_c, risk_c = r0[m + "|cal"]
        Tc, Ec = V["T"][rows_c], V["E"][rows_c]
        kc = label_known(Tc, Ec)
        out = {}
        for split, suf in (("int", ""), ("ext", "|ext")):
            rows, risk = r0[m + suf]
            T, E = V["T"][rows], V["E"][rows]
            y, kn = label(T, E), label_known(T, E)
            site = V["site"][rows]
            sets = np.zeros((len(rows), 2), bool)
            pooled = []
            for s in np.unique(site):
                ms = site == s
                cs_mask = kc & (V["site"][rows_c] == s) if split == "int" else kc
                cs = scores(risk_c[cs_mask, 1], label(Tc, Ec)[cs_mask])
                cy = label(Tc, Ec)[cs_mask]
                st, df = run_stream(risk[ms, 1], y[ms], kn[ms], V["t"][rows][ms], V["t"][rows][ms] + 2, cs, cy, method="static")
                sets[ms] = st
            defer = sets.sum(1) != 1
            out[split] = summarise(sets, defer, y, kn)
        R["conf"][m] = out
        print("conf", m, out)
    json.dump(R, open(f"{RUNS}/fram_results.json", "w"), indent=1)
    for split in ("main", "ext"):
        print(split)
        for m, v in R[split].items():
            print(f"  {m:16s}", " ".join(f"{k}={v[k][0]:.3f}±{v[k][1]:.3f}" for k in KEYS))
    print(json.dumps(R["tests"], indent=0)[:1500])


if __name__ == "__main__":
    main()
