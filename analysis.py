"""Aggregate predictions into the numbers reported in the manuscript (writes runs/results.json)."""
import glob, json, pickle, sys
import numpy as np
from common import (load_sim, visit_table, evaluate, td_auc, brier_h, ece_h, harrell_c, RUNS, INTERNAL, EXTERNAL)

METHODS = ["cox", "rsf", "mlp_fedavg", "gru_local", "gru_fedavg", "gru_fedprox", "gru_ditto", "gru_central",
           "twin_noad", "twin_central", "twin_fed"]
KEYS = ["auc1", "auc2", "auc3", "c", "brier2", "ece2"]


def load_preds():
    P = {}
    for f in sorted(glob.glob(f"{RUNS}/preds_full_e0_s*.pkl")):
        seed = int(f.split("_s")[-1].split(".")[0])
        P[seed] = pickle.load(open(f, "rb"))
    C = pickle.load(open(f"{RUNS}/preds_classical.pkl", "rb"))
    return P, C


def metrics_for(d, V, mask_fn=None):
    rows, risk = d["row"], d["risk"]
    if mask_fn is not None:
        m = mask_fn(rows)
        rows, risk = rows[m], risk[m]
    return evaluate(risk, V["T"][rows], V["E"][rows]), rows, risk


def paired_boot(V, rows, r_a, r_b, metric, B=500, seed=0):
    """cluster bootstrap of metric(a) - metric(b) on identical rows."""
    rng = np.random.default_rng(seed)
    pid = V["pid"][rows]
    up = np.unique(pid)
    groups = {u: np.where(pid == u)[0] for u in up}
    T, E = V["T"][rows], V["E"][rows]
    f = {"auc2": lambda r, i: td_auc(r[i, 1], T[i], E[i], 2.0), "auc3": lambda r, i: td_auc(r[i, 2], T[i], E[i], 3.0),
         "auc1": lambda r, i: td_auc(r[i, 0], T[i], E[i], 1.0),
         "c": lambda r, i: harrell_c(r[i, 2], T[i], E[i], max_pairs=300000),
         "brier2": lambda r, i: brier_h(r[i, 1], T[i], E[i], 2.0), "ece2": lambda r, i: ece_h(r[i, 1], T[i], E[i], 2.0)}[metric]
    diffs, va, vb = [], [], []
    for _ in range(B):
        s = rng.choice(up, len(up))
        i = np.concatenate([groups[u] for u in s])
        a, b = f(r_a, i), f(r_b, i)
        diffs.append(a - b); va.append(a); vb.append(b)
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return dict(diff=float(np.mean(diffs)), lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)),
                p=float(max(p, 1 / B)), ci_a=(float(np.percentile(va, 2.5)), float(np.percentile(va, 97.5))))


def holm(ps):
    ps = np.asarray(ps)
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0
    for k, i in enumerate(o):
        run = max(run, (len(ps) - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def align(d_ref, d):
    """reorder d's predictions to d_ref's row order"""
    pos = {r: i for i, r in enumerate(d["row"])}
    idx = np.array([pos[r] for r in d_ref["row"]])
    return d["risk"][idx]


def main():
    pats, visits = load_sim()
    V = visit_table(pats, visits)
    P, C = load_preds()
    seeds = sorted(P)
    R = dict(main={}, ext={}, sites={}, tests={}, oracle={}, post={})
    for split, suf in (("main", ""), ("ext", "|ext")):
        for m in METHODS:
            vals = []
            if m in ("cox", "rsf"):
                vals.append(metrics_for(C[m + suf], V)[0])
            else:
                for sd in seeds:
                    if m + suf in P[sd]:
                        vals.append(metrics_for(P[sd][m + suf], V)[0])
            if not vals:
                continue
            R[split][m] = {k: (float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals]))) for k in KEYS}
            R[split][m]["per_seed_auc2"] = [float(v["auc2"]) for v in vals]
        if split == "ext":
            for extra in ("twin_fed|ext0",):
                vals = [metrics_for(P[sd][extra], V)[0] for sd in seeds if extra in P[sd]]
                R[split]["twin_fed_zeroshot"] = {k: (float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals]))) for k in KEYS}
    # oracle latent hazard (ranking ceiling)
    ref = P[seeds[0]]["twin_fed"]
    for split, d in (("main", P[seeds[0]]["twin_fed"]), ("ext", P[seeds[0]]["twin_fed|ext"])):
        rows = d["row"]
        lz = V["loghz"][rows]
        R["oracle"][split] = dict(auc1=td_auc(lz, V["T"][rows], V["E"][rows], 1.0), auc2=td_auc(lz, V["T"][rows], V["E"][rows], 2.0),
                                  auc3=td_auc(lz, V["T"][rows], V["E"][rows], 3.0), c=harrell_c(lz, V["T"][rows], V["E"][rows]))
    # per-site AUC2 / ECE2 (seed mean)
    for m in ["gru_fedavg", "gru_ditto", "gru_central", "twin_noad", "twin_fed"]:
        R["sites"][m] = {}
        for s in INTERNAL:
            vals = []
            for sd in seeds:
                if m not in P[sd]:
                    continue
                d = P[sd][m]
                msk = V["site"][d["row"]] == s
                vals.append(evaluate(d["risk"][msk], V["T"][d["row"][msk]], V["E"][d["row"][msk]]))
            R["sites"][m][s] = {k: float(np.mean([v[k] for v in vals])) for k in KEYS}
    # post-drift subsets (S3 after 4.0, S4 after 3.5) for all methods
    for m in METHODS:
        out = {}
        for s, t0 in ((2, 4.0), (3, 3.5)):
            vals = []
            src = [C[m]] if m in ("cox", "rsf") else [P[sd][m] for sd in seeds if m in P[sd]]
            for d in src:
                msk = (V["site"][d["row"]] == s) & (V["cal"][d["row"]] >= t0)
                vals.append(evaluate(d["risk"][msk], V["T"][d["row"][msk]], V["E"][d["row"][msk]]))
            out[s] = {k: float(np.mean([v[k] for v in vals])) for k in KEYS}
        R["post"][m] = out
    # paired bootstrap tests, seed 0, twin_fed vs each baseline
    sd = seeds[0]
    for split, suf in (("main", ""), ("ext", "|ext")):
        base = P[sd]["twin_fed" + suf]
        R["tests"][split] = {}
        for metric in ("auc2", "c", "ece2", "brier2"):
            res, ps = {}, []
            for m in METHODS:
                if m == "twin_fed":
                    continue
                src = C if m in ("cox", "rsf") else P[sd]
                if m + suf not in src:
                    continue
                rb = align(base, src[m + suf])
                res[m] = paired_boot(V, base["row"], base["risk"], rb, metric, B=300)
                ps.append(res[m]["p"])
            adj = holm(ps)
            for (m, v), a in zip(res.items(), adj):
                v["p_holm"] = float(a)
            R["tests"][split][metric] = res
            print(split, metric, {m: (round(v["diff"], 4), round(v["p_holm"], 3)) for m, v in res.items()}, flush=True)
    json.dump(R, open(f"{RUNS}/results.json", "w"), indent=1, default=float)
    for split in ("main", "ext"):
        print(split)
        for m, v in R[split].items():
            print(f"  {m:18s}", " ".join(f"{k}={v[k][0]:.3f}±{v[k][1]:.3f}" for k in KEYS))
    print("oracle", R["oracle"])


if __name__ == "__main__":
    main()


def ablation():
    pats, visits = load_sim()
    V = visit_table(pats, visits)
    A = {}

    def ev(d):
        return evaluate(d["risk"], V["T"][d["row"]], V["E"][d["row"]])

    P0 = pickle.load(open(f"{RUNS}/preds_full_e0_s0.pkl", "rb"))
    A["full"] = dict(int=ev(P0["twin_fed"]), ext=ev(P0["twin_fed|ext"]))
    A["no_adapter"] = dict(int=ev(P0["twin_noad"]), ext=ev(P0["twin_noad|ext"]))
    A["ext_zero_shot_adapter"] = dict(int=ev(P0["twin_fed"]), ext=ev(P0["twin_fed|ext0"]))
    for v in ("recon_only", "contrast_only", "supervised"):
        f = f"{RUNS}/preds_{v}_e0_s0.pkl"
        try:
            P = pickle.load(open(f, "rb"))
            A["enc_" + v] = dict(int=ev(P["twin_fed"]), ext=ev(P["twin_fed|ext"]))
        except FileNotFoundError:
            pass
    vals = []
    for e in (0, 1, 2):
        try:
            P = pickle.load(open(f"{RUNS}/preds_full_e{e}_s0.pkl", "rb"))
            vals.append(dict(int=ev(P["twin_fed"]), ext=ev(P["twin_fed|ext"])))
        except FileNotFoundError:
            pass
    A["encoder_seeds"] = {sp: {k: (float(np.mean([v[sp][k] for v in vals])), float(np.std([v[sp][k] for v in vals])))
                               for k in KEYS} for sp in ("int", "ext")}
    try:
        Ab = pickle.load(open(f"{RUNS}/preds_ablate_s0.pkl", "rb"))
        for k in ("abl_noobs", "abl_nomoment", "abl_nomemory"):
            A[k] = dict(int=ev(Ab[k]), ext=ev(Ab[k + "|ext"]))
    except Exception:
        pass
    json.dump(A, open(f"{RUNS}/ablation.json", "w"), indent=1, default=float)
    for k, v in A.items():
        if k == "encoder_seeds":
            print(k, {sp: {m: round(x[0], 3) for m, x in v[sp].items()} for sp in v})
        else:
            print(f"{k:22s}", "int", {m: round(x, 3) for m, x in v["int"].items()}, "ext", {m: round(x, 3) for m, x in v["ext"].items()})
