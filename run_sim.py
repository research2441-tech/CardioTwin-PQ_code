"""Experiment driver for the CardioTwinSim cohort.

python run_sim.py main <seed> [encoder_variant] [encoder_seed]
python run_sim.py classical
python run_sim.py ablate <seed>
"""
import sys, time, pickle
import numpy as np
import jax, jax.numpy as jnp
from dataset import build
from twin import (train_federated, infer, flatten, fit_adapter_labelfree, init_adapter, VMAX)
from baselines import train_seq, seq_risk
from common import ROUNDS, RUNS, INTERNAL, EXTERNAL, T_TRAIN_END


def collect_twin(p, ads, D, ext_adapter, tag, store, with_ext_zero=True):
    for s in INTERNAL:
        o = flatten(infer(p, ads.get(s, init_adapter()), D["te"][s]), D["te"][s])
        store.setdefault(tag, []).append(o)
    o = flatten(infer(p, ext_adapter, D["ext"]), D["ext"])
    store.setdefault(tag + "|ext", []).append(o)
    if with_ext_zero:
        o = flatten(infer(p, init_adapter(), D["ext"]), D["ext"])
        store.setdefault(tag + "|ext0", []).append(o)


def merge(lst):
    return {k: np.concatenate([d[k] for d in lst]) for k in lst[0]}


def seq_rows(B, risk):
    m = B["VM"] > 0
    return dict(row=B["ROW"][m], risk=risk[m])


def run_main(seed, variant="full", eseed=0, rounds=ROUNDS):
    t0 = time.time()
    D = build(variant, eseed)
    store = {}
    timing = {}
    # --- CardioTwin-PQ: federated twin with acquisition adapters
    ts = time.time()
    p, ads, hist = train_federated(D["tr"], D["va"], seed=seed, rounds=rounds)
    timing["twin_fed"] = time.time() - ts
    ext_ad = fit_adapter_labelfree(p, D["ext_pool"])
    collect_twin(p, ads, D, ext_ad, "twin_fed", store)
    pickle.dump(dict(p=jax.device_get(p), ads=jax.device_get(ads), ext_ad=jax.device_get(ext_ad), hist=hist),
                open(f"{RUNS}/twin_params_{variant}_e{eseed}_s{seed}.pkl", "wb"))
    print("twin fed", time.time() - t0, flush=True)
    if variant == "full" and eseed == 0 and seed == 0:
        # twin without adapters (plain FedAvg)
        p2, ads2, _ = train_federated(D["tr"], D["va"], seed=seed, rounds=rounds, adapters=False)
        collect_twin(p2, {}, D, init_adapter(), "twin_noad", store, with_ext_zero=False)
        # centralised twin (pooled data, single adapter)
        p3, ads3, _ = train_federated(D["tr"], D["va"], seed=seed, rounds=rounds, central=True)
        collect_twin(p3, {s: ads3[0] for s in INTERNAL}, D, fit_adapter_labelfree(p3, D["ext_pool"], ad0=ads3[0]),
                     "twin_central", store, with_ext_zero=False)
        print("twin variants", time.time() - t0, flush=True)
    if variant == "full" and eseed == 0:
        # --- neural baselines
        for name, mode, rec in [("gru_fedavg", "fedavg", True), ("gru_fedprox", "fedprox", True),
                                ("gru_ditto", "ditto", True), ("gru_local", "local", True),
                                ("gru_central", "central", True), ("mlp_fedavg", "fedavg", False)]:
            ts = time.time()
            m = train_seq(D["tr"], D["va"], mode=mode, recurrent=rec, seed=seed, rounds=rounds)
            timing[name] = time.time() - ts
            for s in INTERNAL:
                if mode in ("ditto", "local"):
                    r = seq_risk(m["local"][s], D["te"][s])
                else:
                    r = seq_risk(m["global_"], D["te"][s])
                store.setdefault(name, []).append(seq_rows(D["te"][s], r))
            if mode == "local":
                r = np.mean([seq_risk(m["local"][s], D["ext"]) for s in INTERNAL], 0)
            else:
                r = seq_risk(m["global_"], D["ext"])
            store.setdefault(name + "|ext", []).append(seq_rows(D["ext"], r))
            print(name, time.time() - t0, flush=True)
    out = {k: merge(v) for k, v in store.items()}
    out["timing"] = timing
    pickle.dump(out, open(f"{RUNS}/preds_{variant}_e{eseed}_s{seed}.pkl", "wb"))
    print("done", time.time() - t0)


def run_classical():
    from baselines import fit_classical, classical_risk
    from common import load_sim, make_splits, visit_table
    pats, visits = load_sim()
    V = visit_table(pats, visits)
    role = make_splits(pats)
    r = np.array([role[p] for p in V["pid"]])
    X = np.load(f"{RUNS}/features.npz")["X"]
    dev = V["cal"] < T_TRAIN_END
    tr = np.where((r == "train") & dev & (V["site"] != EXTERNAL))[0]
    med = np.nanmedian(X[tr], 0)
    mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0) + 1e-6
    Xi = np.where(np.isnan(X), med, X)
    miss = np.isnan(X[:, [0, 3, 4, 12]]).astype(float)   # ECG, PAT/PPG, labs missing indicators
    Xs = np.concatenate([(Xi - mu) / sd, miss], 1)
    te = np.where((r == "test") & (V["site"] != EXTERNAL))[0]
    early = V["cal"] - V["t"] < 2.5
    ex = np.where((V["site"] == EXTERNAL) & ~early)[0]
    out = {}
    for kind in ("cox", "rsf"):
        t0 = time.time()
        m = __import__("baselines").fit_classical(Xs[tr], V["Ttr"][tr], V["Etr"][tr], kind)
        out[kind] = dict(row=te, risk=classical_risk(m, Xs[te], kind))
        out[kind + "|ext"] = dict(row=ex, risk=classical_risk(m, Xs[ex], kind))
        print(kind, time.time() - t0, flush=True)
    pickle.dump(out, open(f"{RUNS}/preds_classical.pkl", "wb"))


def run_ablate(seed, rounds=ROUNDS):
    """Twin-side ablations on the full encoder, and encoder ablations with the full twin."""
    import twin as T
    store = {}
    D = build("full", 0)
    # (a) no innovation (observation) likelihood: risk-only training of the filter
    p, ads, _ = T.train_federated(D["tr"], D["va"], seed=seed, rounds=rounds, use_obs=False)
    collect_twin(p, ads, D, T.fit_adapter_labelfree(p, D["ext_pool"]), "abl_noobs", store, with_ext_zero=False)
    print("noobs done", flush=True)
    # (b) no moment matching: hazard from the posterior mean only
    orig = T.filter_patient
    T.MOMENT = False
    p, ads, _ = T.train_federated(D["tr"], D["va"], seed=seed, rounds=rounds)
    collect_twin(p, ads, D, T.fit_adapter_labelfree(p, D["ext_pool"]), "abl_nomoment", store, with_ext_zero=False)
    T.MOMENT = True
    print("nomoment done", flush=True)
    # (c) no filtering memory: each visit filtered from the prior (static twin)
    T.MEMORY = False
    p, ads, _ = T.train_federated(D["tr"], D["va"], seed=seed, rounds=rounds)
    collect_twin(p, ads, D, T.fit_adapter_labelfree(p, D["ext_pool"]), "abl_nomemory", store, with_ext_zero=False)
    T.MEMORY = True
    print("nomemory done", flush=True)
    out = {k: merge(v) for k, v in store.items()}
    pickle.dump(out, open(f"{RUNS}/preds_ablate_s{seed}.pkl", "wb"))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "main":
        run_main(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "full",
                 int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    elif cmd == "classical":
        run_classical()
    elif cmd == "ablate":
        run_ablate(int(sys.argv[2]))
